#!/usr/bin/env python3
"""Vocal Health Coach — Real-time WebSocket Backend

Browser streams microphone audio → server analyzes → streams results back.

Architecture:
  Browser → /ws (binary audio frames, Float32 PCM)
  Server  → /ws (JSON analysis events)
  Debug   → /debug/ws (all events + extra debug data + server logs)

Endpoints:
  GET  /          → singer UI
  GET  /debug     → Koda debug panel
  WS   /ws        → combined audio-in / results-out
  WS   /debug/ws  → debug event stream
  GET  /devices   → (local only) list audio devices

WebSocket event types (server → client):
  ears_frame    — 10Hz analysis: strain, pitch, all EARS dims
  phrase_start  — vocal phrase detected
  phrase_end    — phrase ended, summary + coaching hook
  session_update— strain history snapshot
  log           — server log entry (debug clients only)
  config        — sent on connect
"""

import asyncio
import collections
import json
import math
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml

import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from contextlib import asynccontextmanager

# EARS
sys.path.insert(0, str(Path(__file__).parent))
from mel_extractor import MelExtractor
from frequency_explorer import analyze_mel
try:
    from frequency_explorer import compute_emotion_properties, compute_tactile_properties
    _HAS_CROSSMODAL_BACKEND = True
except ImportError:
    _HAS_CROSSMODAL_BACKEND = False
    compute_emotion_properties = None
    compute_tactile_properties = None

try:
    import parselmouth
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False
    print("[WARN] praat-parselmouth not installed — HNR/shimmer disabled")

try:
    # Bypass broken 3D module (scipy.special.sph_harm removed in scipy 1.17)
    from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as _Scattering1D
    _scatter_transform = _Scattering1D(J=7, shape=4096, Q=1)
    SCATTER_AVAILABLE = True
except Exception as _e:
    _scatter_transform = None
    SCATTER_AVAILABLE = False
    print(f"[WARN] kymatio not available — wavelet scatter disabled: {_e}")

try:
    import joblib as _joblib
    _CLASSIFIER_PATH = Path(__file__).parent.parent / "models" / "phonation_classifier.joblib"
    if _CLASSIFIER_PATH.exists():
        _phon_bundle = _joblib.load(_CLASSIFIER_PATH)
        _phon_model       = _phon_bundle["model"]
        _phon_int_to_label = _phon_bundle["int_to_label"]
        _phon_strain_map   = _phon_bundle["strain_map"]
        CLASSIFIER_AVAILABLE = True
        print(f"[INFO] Phonation classifier loaded ({_CLASSIFIER_PATH.name}) "
              f"acc={_phon_bundle.get('accuracy', '?'):.3f}")
    else:
        CLASSIFIER_AVAILABLE = False
        _phon_model = None
        print(f"[WARN] Phonation classifier not found at {_CLASSIFIER_PATH} — pressed detection disabled")
except Exception as _ce:
    CLASSIFIER_AVAILABLE = False
    _phon_model = None
    print(f"[WARN] Phonation classifier load failed: {_ce}")

import base64
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARN] google-genai not installed — Gemini coaching disabled")

# ─── Config ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
CHUNK_SAMPLES = 4410            # 100ms per tick
EARS_WINDOW_SAMPLES = 44100     # 1s EARS window (full ring buffer)
PRAAT_WINDOW_SAMPLES = 8820     # 200ms parselmouth window — more stable shimmer (smooths phrase-onset spikes)

# Formula v8 — dual-signal: shimmer (rough phonation) + CPP (phonatory irregularity/tightness)
# CPP = Cepstral Peak Prominence: high CPP = healthy periodic voice, low CPP = strain/irregularity.
# Loudness-robust: loud+healthy → HIGHER CPP (more harmonic clarity); loud+strained → lower CPP.
# This eliminates the core volume false-positive problem — natural loud singing is not flagged.
#   shim_dev:  shimmer SPIKE above baseline → rough phonation
#   cpp_dev:   CPP DROP below baseline → phonatory irregularity, tightness, constriction
# Both gated on voiced_run < 3 (suppressed during onset, where metrics are unreliable).
# HNR deliberately excluded — rises for BOTH pressed strain AND relaxed high notes (untrustworthy)
# strain = min(1.0, max(shim_dev, cpp_dev))   ← either signal alone → strain
# Song validation: Runnin' Down → GREEN, Chris Young → YELLOW (both match expectations)
STRAIN_GREEN  = 0.25   # below = healthy phonation
STRAIN_YELLOW = 0.40   # above = RED — requires sustained push; Liza Jane green peaks 0.35-0.39 briefly, AC/DC sustains 0.40+

SILENCE_RMS = 0.003             # activity gate: below = no voice detected (lowered for browser mic gain)
LOW_ENERGY_RMS = 0.007          # phrase-tail gate: below = low amplitude (lowered to match browser mic)
SILENCE_HOLD_FRAMES = 3         # require 3 consecutive silent frames before going idle (prevents flicker)
MIN_PHRASE_S = 0.5
MIN_SILENCE_S = 0.4

LOG_BUFFER_SIZE = 500           # in-memory log entries for debug UI

# ─── Global state ─────────────────────────────────────────────────────────────

_audio_q: queue.Queue = queue.Queue(maxsize=200)
_ring = np.zeros(EARS_WINDOW_SAMPLES, dtype=np.float32)
_mel = MelExtractor(sample_rate=SAMPLE_RATE)
_executor = ThreadPoolExecutor(max_workers=2)
_main_loop: asyncio.AbstractEventLoop = None  # set in lifespan
_ema_strain: float = 0.0   # exponential moving average for smooth strain display
EMA_ALPHA = 0.40           # smoothing factor: ~2-3 frames to respond (was 0.25 — too slow, masked short-phrase strain)

# Session-adaptive baseline — continuous EMA adaptation from clean (relaxed) frames.
# Design goals:
#   1. Works immediately from session start (no warmup required) — seed is a valid baseline
#   2. Adapts to each singer's natural voice over the first few seconds of easy singing
#   3. Cannot be corrupted by strained frames — only clean frames (score < MAX_SCORE) contribute
#   4. Works correctly for beginners who strain from frame 1 — seed stays untouched if no
#      clean frames are detected; system still produces valid strain readings vs the seed
BASELINE_EMA_ALPHA = 0.05         # per clean frame: ~10 frames → 40% dialed in, ~50 → 92%
BASELINE_MAX_SCORE = 0.35         # gate: only frames this relaxed contribute to adaptation
                                   # 0.35 allows shimmer up to baseline+3.5% to update — covers normal singer variation
BASELINE_WARM_N    = 20           # clean frames before baseline is considered "warmed up"

# Pre-seeded from Daniel's Easy 2 anchor clip.
# Seed is intentionally conservative — a mid-range healthy voice. Other singers will
# refine it toward their own voice within seconds of easy phonation.
SEED_HNR_BASELINE  = 17.0   # dB  — Daniel easy singing HNR (for breathy detection)
SEED_SHIM_BASELINE = 7.0    # %   — typical healthy adult male singing shimmer (live/iPhone mic)
SEED_CPP_BASELINE  = 0.22   # CPP nepers — Daniel easy singing (Easy 2 mean=0.198, seed at 0.22)

_session_hnr_baseline:  float = SEED_HNR_BASELINE
_session_shim_baseline: float = SEED_SHIM_BASELINE
_session_cpp_baseline:  float = SEED_CPP_BASELINE
_baseline_clean_n: int = 0        # clean frames contributed so far this session
_voiced_run_count: int = 0        # consecutive voiced frames — onset gate for shimmer/CPP
_silence_frames: int = 0          # consecutive silent frames — hang-over counter for flicker prevention

# Deep-analysis derived signals (v11) — from Cohen's d ranking on Liza Jane ground truth.
# elastikos (d=-0.999): EARS touch.elastikos — energy envelope decay oscillation.
#   Strained voice → MORE elastic (higher value) — session-adaptive, deviates UP with strain.
# anharmonia (d=-0.727): EARS harmonic inharmonicity (std/mean of harmonic spacings).
#   Strained voice → MORE inharmonic (higher value) — session-adaptive.
# am_fast (d=-0.564): AM modulation power at 10-30Hz in 75-300Hz sub-band.
#   Strained voice → MORE fast amplitude modulation — Hilbert envelope method.
#   (Audio equivalent of Eulerian Video Magnification — amplify slow AM riding on carrier)
SEED_ELAST_BASELINE = 0.30   # seed: typical healthy-voice elastikos (adapts per session)
SEED_ANHAM_BASELINE = 0.15   # seed: typical healthy-voice anharmonia (adapts per session)
SEED_ALPHA_BASELINE = -8.0   # seed: alpha ratio dB for comfortable singing (~50Hz-1kHz dominant)
_session_elast_baseline: float = SEED_ELAST_BASELINE
_session_anham_baseline: float = SEED_ANHAM_BASELINE
_session_am_baseline: float = 0.0   # starts at 0, adapts upward from clean frames
_session_alpha_baseline: float = SEED_ALPHA_BASELINE
_rms_history: collections.deque = collections.deque(maxlen=200)  # rolling 10Hz RMS for 4-8Hz effort AM computation
_session_effort_am_baseline: float = 0.0

# v11 run count — resets on BOTH silence AND low_energy (unlike _voiced_run_count which only
# resets on silence). Low_energy frames contain amplitude transitions that contaminate ring
# buffer temporal features (elastikos/anharmonia). A fresh 20-frame gate ensures the ring
# buffer is filled with stable, full-amplitude voice before temporal features fire.
_v11_run_count: int = 0

# CPP 3-frame EMA — smooths phoneme-level CPP dips (100ms frames vary wildly with phoneme).
# Without smoothing, a single /i/ vowel or consonant frame can drop CPP by 0.3+, falsely
# triggering cpp_dev. With EMA, sustained dips (real strain) still register while transients wash out.
CPP_EMA_ALPHA = 0.33  # ~3-frame (300ms) time constant at 100ms per frame
_cpp_ema: float = SEED_CPP_BASELINE  # smoothed CPP per session

# Wavelet scattering v9 — session-adaptive scatter baseline.
# Builds within-session scatter baseline from v8-gated clean frames.
# Detects multi-scale amplitude modulation changes (strain → different modulation pattern).
# 2.7ms/frame overhead. Disabled until SCATTER_WARM_N clean frames collected.
SCATTER_WARM_N   = 15    # clean frames before scatter is active (~5-10s of easy singing)
SCATTER_EMA_ALPHA = 0.05  # same as v8 baseline EMA
SCATTER_FUSION_W  = 0.3   # fusion: 70% max + 30% mean — any single signal can carry strain
# Max-fusion: fuse = (1-w)*max(v8,scatter) + w*wavg(v8,scatter)
# 0.5 diluted single-signal detections below threshold; 0.3 preserves detection.

_scatter_baseline_feats: list = []  # accumulates clean frames during warmup
_scatter_mean: Optional[np.ndarray] = None   # set after SCATTER_WARM_N frames
_scatter_std:  Optional[np.ndarray] = None   # per-dim std for z-score normalization

# Pitch smoothing — EMA on Hz + note stability gate
PITCH_EMA_ALPHA = 0.15      # ~7-frame time constant (700ms) — smooths vibrato/breath noise
_pitch_ema_hz: float = 0.0  # EMA-smoothed fundamental frequency
_pitch_last_note: str = "—" # last candidate note (tracks what we're detecting)
_pitch_stable_note: str = "—" # last stable note (displayed after 2 consecutive frames)
_pitch_note_count: int = 0  # consecutive frames showing same candidate note

# Vibrato detection — ring buffer of raw Hz, detect periodic oscillation
VIBRATO_BUF_SIZE = 12       # 1.2 seconds at 10 Hz — need ~2 full vibrato cycles
VIBRATO_MIN_CENTS = 20      # minimum oscillation amplitude in cents (was 15 — reduced false positives)
VIBRATO_MIN_CROSSINGS = 4   # minimum zero-crossings (was 3 — requires more consistent periodicity)
VIBRATO_HOLD_FRAMES = 4     # hold vibrato=True for 400ms after last detection
_vibrato_hz_buf: list[float] = []  # recent raw Hz values
_vibrato_hold_count: int = 0  # countdown frames for vibrato hold

_singer_clients: list[WebSocket] = []   # browser singer UIs
_debug_clients: list[WebSocket] = []    # Koda debug panels

_session_history: list[float] = []
_phrase_history: collections.deque = collections.deque(maxlen=20)  # last 20 phrases
_log_buffer: collections.deque = collections.deque(maxlen=LOG_BUFFER_SIZE)
_session_start = time.time()
_frame_count = 0
_analysis_latencies: collections.deque = collections.deque(maxlen=50)

# Song-end praise tracking
_session_phrase_count: int = 0          # total phrases this session
_session_yellow_red_count: int = 0      # phrases that were yellow or red
_song_end_timer_task: Optional[asyncio.Task] = None  # pending song-end timer

# Vocal range mapper
_note_strain: dict = {}                 # "C4" → list of strain scores while that note was sung
RANGE_BROADCAST_EVERY = 15            # send range_update every 15 active frames (~1.5s)
MIN_FRAMES_PER_NOTE   = 3             # need ≥3 frames of a note before coloring it

# Feedback ladder — controls when Gemini speaks vs. shows visual-only cue
_consecutive_yellow: int = 0          # consecutive yellow phrases (resets on green or red)
YELLOW_VOICE_THRESHOLD = 1            # speak on first yellow (was 3 — lowered for demo visibility)

# ─── Gemini Vocal Coach ───────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")
GEMINI_MODEL   = "models/gemini-2.5-flash-native-audio-latest"

# ─── Load coaching config ─────────────────────────────────────────────────────

_COACHING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "coaching_responses.yaml"

def _load_coaching_config() -> dict:
    try:
        return yaml.safe_load(_COACHING_CONFIG_PATH.read_text())
    except Exception as e:
        print(f"[WARN] Could not load coaching_responses.yaml: {e}")
        return {}

_coaching_cfg = _load_coaching_config()

def _build_system_prompt(cfg: dict) -> str:
    pc = cfg.get("phrase_coaching", {})
    y_ex  = "\n".join(f'  "{e}"' for e in pc.get("yellow",  {}).get("examples", []))
    r_ex  = "\n".join(f'  "{e}"' for e in pc.get("red",     {}).get("examples", []))
    br_ex = "\n".join(f'  "{e}"' for e in pc.get("breathy", {}).get("examples", []))
    return f"""\
You are Koda, a warm real-time vocal coach who guides singers through warm-ups and practice. \
You're monitoring the singer's vocal strain using acoustic analysis — shimmer, cepstral peak \
prominence, harmonic-to-noise ratio, and perceptual features — all in real-time.

You just greeted the singer and invited them to warm up. Now you'll coach them through their \
singing with short technique cues at each breath point.

After each sung phrase you receive: zone (green/yellow/red), avg strain score (0–1), \
phrase duration in seconds, and recent phrase history.

Your job: one VERY SHORT coaching tip. Maximum 8 words — like a coach calling out during a performance.
- green: brief positive reinforcement or technique note. "Good placement." "Nice and easy." "That's it."
- yellow: quick technique cue. Style: {pc.get('yellow',{}).get('tone','')}
- red:    one urgent word or short phrase. Style: {pc.get('red',{}).get('tone','')}
- breathy: breath support reminder. Style: {pc.get('breathy',{}).get('tone','')}

Keep it under 8 words. Be warm but brief — the singer is actively performing.
Examples: "Ease up on the push." "Drop your jaw, more space." "Good, keep that placement." "Back off a touch."

Warm-up progression: In the first ~8 phrases, you're guiding a warm-up. Encourage easy singing first \
("Nice and easy", "Good placement"), then after 4-5 green phrases encourage the singer to push a \
little ("Now try taking it higher", "Push into the chorus"). After ~8-10 phrases say something like \
"You're warm — sing something you love!" to transition to free singing.

Yellow examples:
{y_ex}

Red examples:
{r_ex}

Breathy examples:
{br_ex}
"""

_COACH_SYSTEM_PROMPT = _build_system_prompt(_coaching_cfg)

# Song-end praise constants (from config or defaults)
_sp_cfg = _coaching_cfg.get("song_praise", {})
SONG_END_SILENCE_S    = 4.0   # seconds of silence before song-end praise fires
MIN_PHRASES_FOR_PRAISE = 8    # require a full song's worth of phrases (not just a verse)
_SONG_PRAISE_TIERS    = _sp_cfg.get("tiers", [
    {"threshold": 0,  "tier": "perfect",   "text": "You nailed that — perfectly clean the whole way through."},
    {"threshold": 5,  "tier": "excellent", "text": "Really strong session — almost entirely in the green."},
    {"threshold": 10, "tier": "solid",     "text": "Solid session — mostly green with just a little push at the edges."},
])

class GeminiVocalCoach:
    """Wraps a persistent Gemini Live session for phrase-boundary coaching."""

    def __init__(self):
        self._client = None
        self._session = None
        self._session_ctx = None
        self._lock = asyncio.Lock()
        self.connected = False

    async def connect(self):
        if not GEMINI_AVAILABLE:
            return
        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name="Kore")
                    )
                ),
                output_audio_transcription=genai_types.AudioTranscriptionConfig(),
                system_instruction=_COACH_SYSTEM_PROMPT,
            )
            self._session_ctx = self._client.aio.live.connect(model=GEMINI_MODEL, config=config)
            self._session = await self._session_ctx.__aenter__()
            self.connected = True
            print("[Gemini] Vocal coach connected")
        except Exception as e:
            print(f"[Gemini] Connect failed: {e}")

    async def disconnect(self):
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self.connected = False

    async def coach_phrase(self, zone: str, avg_strain: float, duration_s: float,
                           history: list, is_breathy: bool = False):
        """Send phrase data to Gemini, collect audio+text response, broadcast to clients."""
        if not GEMINI_AVAILABLE:
            return
        # Reconnect on demand — Live sessions time out after ~15 min idle
        if not self.connected:
            await self.connect()
        if not self.connected:
            return
        async with self._lock:
            try:
                recent = [p["zone"] for p in history[-5:]] if history else []
                breathy_note = " Note: the voice sounded BREATHY (too much air, not enough support)." if is_breathy else ""
                warmup_note = f" This is phrase #{_session_phrase_count} of the warm-up." if _session_phrase_count <= 10 else " Warm-up complete — free singing."
                prompt = (
                    f"Phrase just ended.{warmup_note} Zone: {zone.upper()}, "
                    f"strain: {avg_strain:.2f}, duration: {duration_s:.1f}s. "
                    f"Recent: {recent}.{breathy_note} Give your coaching cue now."
                )
                await self._session.send_client_content(
                    turns=genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)]),
                    turn_complete=True,
                )
                audio_chunks = []
                transcript = ""
                async for response in self._session.receive():
                    if response.data:
                        audio_chunks.append(response.data)
                    try:
                        sc = response.server_content
                        if sc:
                            if sc.output_transcription and sc.output_transcription.text:
                                transcript += sc.output_transcription.text
                            if sc.turn_complete:
                                break
                    except Exception:
                        pass

                audio_b64 = base64.b64encode(b"".join(audio_chunks)).decode() if audio_chunks else None
                await _broadcast_all({
                    "type": "gemini_coaching",
                    "zone": zone,
                    "transcript": transcript.strip(),
                    "audio_b64": audio_b64,
                })
                print(f"[Gemini] Coaching: '{transcript.strip()[:80]}'")

            except Exception as e:
                print(f"[Gemini] Coaching error (will reconnect next phrase): {e}")
                self.connected = False
                await self.disconnect()

    async def praise_song(self, tier: str, yellow_pct: float, phrase_count: int):
        """Send end-of-song praise when session was mostly green."""
        if not GEMINI_AVAILABLE:
            return
        if not self.connected:
            await self.connect()
        if not self.connected:
            return
        async with self._lock:
            try:
                # Find the matching tier text from config
                praise_text = ""
                for t in _SONG_PRAISE_TIERS:
                    if yellow_pct <= t["threshold"] or t["tier"] == tier:
                        praise_text = t.get("text", "")
                        break
                prompt = (
                    f"Song complete. The singer just finished {phrase_count} phrases "
                    f"with only {yellow_pct:.0f}% in yellow/red — a {tier} session. "
                    f"Deliver this end-of-song praise warmly, in your own words, "
                    f"keeping the spirit of: \"{praise_text}\""
                )
                await self._session.send_client_content(
                    turns=genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)]),
                    turn_complete=True,
                )
                audio_chunks = []
                transcript = ""
                async for response in self._session.receive():
                    if response.data:
                        audio_chunks.append(response.data)
                    try:
                        sc = response.server_content
                        if sc:
                            if sc.output_transcription and sc.output_transcription.text:
                                transcript += sc.output_transcription.text
                            if sc.turn_complete:
                                break
                    except Exception:
                        pass

                audio_b64 = base64.b64encode(b"".join(audio_chunks)).decode() if audio_chunks else None
                await _broadcast_all({
                    "type": "song_praise",
                    "tier": tier,
                    "yellow_pct": round(yellow_pct, 1),
                    "phrase_count": phrase_count,
                    "transcript": transcript.strip(),
                    "audio_b64": audio_b64,
                })
                print(f"[Gemini] Song praise ({tier}): '{transcript.strip()[:80]}'")

            except Exception as e:
                print(f"[Gemini] Song praise error: {e}")
                self.connected = False
                await self.disconnect()

_gemini_coach = GeminiVocalCoach()


async def _gemini_greet():
    """Short Gemini greeting when a singer connects — lets judges hear Koda immediately."""
    if not GEMINI_AVAILABLE:
        return
    if not _gemini_coach.connected:
        await _gemini_coach.connect()
    if not _gemini_coach.connected:
        return
    try:
        async with _gemini_coach._lock:
            await _gemini_coach._session.send_client_content(
                turns=genai_types.Content(role="user", parts=[genai_types.Part(
                    text="A singer just connected for a warm-up. Say warmly: 'Hey! Let's warm up your voice. Start whenever you're ready — nice and easy.' Keep it natural and under 15 words. Sound like a friendly vocal coach greeting a student."
                )]),
                turn_complete=True,
            )
            audio_chunks = []
            transcript = ""
            async for response in _gemini_coach._session.receive():
                if response.data:
                    audio_chunks.append(response.data)
                try:
                    sc = response.server_content
                    if sc:
                        if sc.output_transcription and sc.output_transcription.text:
                            transcript += sc.output_transcription.text
                        if sc.turn_complete:
                            break
                except Exception:
                    pass
            audio_b64 = base64.b64encode(b"".join(audio_chunks)).decode() if audio_chunks else None
            await _broadcast_all({
                "type": "gemini_coaching",
                "zone": "green",
                "transcript": transcript.strip(),
                "audio_b64": audio_b64,
            })
            print(f"[Gemini] Greeting: '{transcript.strip()[:60]}'")
    except Exception as e:
        print(f"[Gemini] Greeting error: {e}")


# ─── Logging ──────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "info"):
    entry = {
        "type": "log",
        "ts": round(time.time() - _session_start, 2),
        "level": level,
        "msg": msg,
    }
    _log_buffer.append(entry)
    print(f"[{level.upper()}] {msg}")
    # Fire-and-forget to debug clients
    try:
        _main_loop.call_soon_threadsafe(
            lambda e=entry: _main_loop.create_task(_broadcast_debug(e))
        )
    except RuntimeError:
        pass  # loop not running yet (startup) or already closed (shutdown)

# ─── Strain analysis ──────────────────────────────────────────────────────────

def _compute_cpp(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Cepstral Peak Prominence (CPP) — loudness-robust strain indicator.
    High CPP = healthy periodic phonation. Low CPP = strain/irregularity/tightness.
    Loud+healthy naturally scores HIGHER CPP, preventing volume false-positives.
    """
    try:
        N = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N // 2]
        q_min = int(sr / 600)
        q_max = int(sr / 75)
        if q_max >= len(cepstrum):
            return float('nan')
        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max + 1]))
        qs = np.arange(q_min, q_max + 1) / float(sr)
        cs = cepstrum[q_min:q_max + 1]
        coeffs = np.polyfit(qs, cs, 1)
        regression_at_peak = np.polyval(coeffs, peak_idx / float(sr))
        return float(cepstrum[peak_idx] - regression_at_peak)
    except Exception:
        return float('nan')


def _scatter_features(chunk: np.ndarray) -> Optional[np.ndarray]:
    """34-dim log-compressed wavelet scattering features from a 100ms audio chunk.
    RMS-normalized before scatter → loudness invariant.
    Log-compressed after scatter → equalizes 3-order-of-magnitude dynamic range across dims.
    Returns None if scatter unavailable or chunk too short/silent."""
    if not SCATTER_AVAILABLE:
        return None
    try:
        x = chunk.astype(np.float64)
        if len(x) < 4096:
            x = np.pad(x, (0, 4096 - len(x)))
        else:
            x = x[:4096]
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < 1e-8:
            return None
        x = x / rms
        Sx = _scatter_transform(x)        # (34, 32)
        feat = np.mean(Sx, axis=1)        # 34-dim mean across time
        return np.log(np.abs(feat) + 1e-10)   # log-compress
    except Exception:
        return None


def _scatter_strain_score(feat: np.ndarray) -> float:
    """Mean absolute z-score of scatter features vs session baseline.
    Returns 0.0 if baseline not yet warmed up. Range [0, 1]."""
    if _scatter_mean is None or _scatter_std is None:
        return 0.0
    z = (feat - _scatter_mean) / _scatter_std
    raw = float(np.mean(np.abs(z)))
    # raw ≈ 1.0 at baseline (by definition of std). Scale: (raw-1)/2 → 0 at baseline, 0.5 at 2×.
    return min(1.0, max(0.0, (raw - 1.0) / 2.0))


def _phonation_score(chunk: np.ndarray) -> float:
    """Phonation classifier: returns pressed_strain_score (overdrive+edge probability).
    Uses the same 34-dim log-compressed scatter features as the trained model.
    Returns 0.0 if classifier unavailable. Range [0, 1]."""
    if not CLASSIFIER_AVAILABLE or _phon_model is None:
        return 0.0
    if not SCATTER_AVAILABLE:
        return 0.0
    try:
        feat = _scatter_features(chunk)
        if feat is None:
            return 0.0
        proba = _phon_model.predict_proba(feat.reshape(1, -1))[0]
        # proba order: overdrive=0, edge=1, neutral=2, curbing=3
        return float(proba[0] + proba[1])   # overdrive + edge = pressed/strained
    except Exception:
        return 0.0


def _alpha_ratio(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Alpha ratio: log energy ratio (1-5kHz) / (50Hz-1kHz).
    Captures spectral tilt shift during pressed phonation.
    Published validation: Sol et al. 2023 — #1 feature for vocal mode classification (92% F1).
    Pressed/strained singing → energy tilts toward higher harmonics → HIGHER alpha ratio.
    Returns value in dB (typically -20 to +10 dB range). NaN if chunk is silent."""
    try:
        n_fft = 2048
        S = np.abs(np.fft.rfft(chunk.astype(np.float64) * np.hanning(len(chunk)), n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        low_mask  = (freqs >= 50)   & (freqs < 1000)
        high_mask = (freqs >= 1000) & (freqs < 5000)
        low_energy  = float(np.sum(S[low_mask]))
        high_energy = float(np.sum(S[high_mask]))
        if low_energy < 1e-20:
            return float('nan')
        return float(10.0 * np.log10(high_energy / low_energy + 1e-10))
    except Exception:
        return float('nan')


def _am_effort_depth(rms_history: list, sr_env: int = 10) -> float:
    """4-8 Hz AM depth in RMS envelope — the 'effort band'.
    Research: laryngeal muscle micro-fluctuations at 3.5-8 Hz correlate with vocal effort.
    Requires ~500ms history (50 frames at 10Hz). Returns 0 if insufficient history.
    Published basis: Drullman 1994 (8-10 Hz divides intelligibility); effort-band AM
    amplitudes ~3.5-5 Hz link to laryngeal muscle modulation during effortful phonation."""
    if len(rms_history) < 50:
        return 0.0
    try:
        from scipy.signal import butter, filtfilt
        arr = np.array(rms_history[-100:], dtype=np.float64)
        if np.std(arr) < 1e-8:
            return 0.0
        nyq = sr_env / 2.0
        b, a = butter(2, [4.0 / nyq, 8.0 / nyq], btype='band')
        filtered = filtfilt(b, a, arr)
        return float(np.std(filtered))
    except Exception:
        return 0.0


def _am_fast_power(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """AM modulation power (10-30Hz band) in 75-300Hz sub-band.
    Audio equivalent of Eulerian Video Magnification: measures fast amplitude fluctuations
    riding on the fundamental. Elevated in pressed/strained phonation (d=-0.564)."""
    try:
        from scipy.signal import butter, sosfilt
        from scipy.signal import hilbert as _hilbert
        sos = butter(4, [75, 300], btype='bandpass', fs=sr, output='sos')
        filtered = sosfilt(sos, chunk.astype(np.float64))
        envelope = np.abs(_hilbert(filtered))
        env_rms = float(np.sqrt(np.mean(envelope ** 2)))
        if env_rms < 1e-8:
            return 0.0
        envelope_norm = envelope / env_rms
        env_fft = np.fft.rfft(envelope_norm)
        env_freqs = np.fft.rfftfreq(len(envelope_norm), d=1.0 / sr)
        mask = (env_freqs >= 10.0) & (env_freqs < 30.0)
        return float(np.sum(np.abs(env_fft[mask]) ** 2)) / (len(env_fft) + 1e-9)
    except Exception:
        return 0.0


def _compute_strain(audio: np.ndarray) -> dict:
    """EARS + Parselmouth + wavelet scatter + phonation classifier combined strain.
    Returns idle dict if no voice detected."""
    global _voiced_run_count, _v11_run_count, _cpp_ema, _session_cpp_baseline
    global _silence_frames

    global _rms_history, _session_effort_am_baseline

    # Activity gate — only run heavy analysis if voice is present
    rms_val = float(np.sqrt(np.mean(audio[-CHUNK_SAMPLES:] ** 2)))
    _rms_history.append(rms_val)
    if rms_val < SILENCE_RMS:
        _silence_frames += 1
        if _silence_frames >= SILENCE_HOLD_FRAMES:
            # Confirmed silence — reset gates and return idle
            _voiced_run_count = 0
            _v11_run_count = 0
            # Reset CPP EMA to current session baseline so each new phrase starts neutral.
            # Prevents old-phrase high CPP from making healthy phrase-start CPP look like a drop.
            _cpp_ema = _session_cpp_baseline
            return {
                "active": False,
                "rms": round(rms_val, 5),
                "strain_score": 0.0,
                "zone": "idle",
                "tonos": 0.0, "thymos": 0.0, "trachytes": 0.0, "kallos": 0.0,
                "hnr_db": 0.0, "shimmer_pct": 0.0,
                "ears_score": 0.0,
            }
        # Brief dip (< SILENCE_HOLD_FRAMES) — treat as low_energy active to prevent flicker
        # Fall through with low_energy=True, voiced_run_count stays intact
    else:
        _silence_frames = 0   # reset hold counter on any active frame
    _voiced_run_count += 1
    onset_gated = _voiced_run_count < 3  # suppress unreliable onset measurements (300ms — physiological minimum for fold settling)
    # Low-energy gate: phrase tails (RMS just above silence floor) produce garbage shimmer/CPP.
    # At low amplitude, the shimmer estimator is dominated by noise, not glottal cycles.
    # Gate these frames identically to onset_gated — zero strain contribution.
    low_energy = rms_val < LOW_ENERGY_RMS

    # v11 run count — resets on low_energy in addition to silence.
    # L frames → active transitions create amplitude modulation that falsely triggers
    # elastikos/anharmonia. A fresh 20-frame gate ensures ring buffer is stable voice only.
    if low_energy:
        _v11_run_count = 0
    else:
        _v11_run_count += 1

    t0 = time.time()

    # ── EARS (fast path: emotion + touch only = 12x faster than full analyze_mel) ──
    tonos, thymos, trachytes, kallos = 0.5, 0.5, 0.0, 0.5
    ears_score = 0.5
    all_dims: dict = {}
    try:
        mel_frames = _mel.extract_from_audio(audio)

        # Fast path: only the two modalities we need for strain
        em = compute_emotion_properties(mel_frames) if _HAS_CROSSMODAL_BACKEND else {}
        tc = compute_tactile_properties(mel_frames) if _HAS_CROSSMODAL_BACKEND else {}
        def _safe(v, default=0.5):
            f = float(v) if v is not None else default
            return default if math.isnan(f) or math.isinf(f) else max(0.0, min(1.0, f))

        tonos     = _safe(em.get("tension"),  0.5)
        thymos    = _safe(em.get("arousal"),  0.5)
        kallos    = _safe(em.get("beauty"),   0.5)
        trachytes = _safe(tc.get("roughness"), 0.0)

        # Full analyze_mel — needed for temporal energy variance dims (primary strain signal).
        # 9ms vs 1ms fast path, but parselmouth already costs ~78ms so negligible.
        result = analyze_mel(mel_frames)
        for mod_name, mod_dict in result.get("modalities", {}).items():
            if isinstance(mod_dict, dict):
                for k, v in mod_dict.items():
                    if isinstance(v, (int, float)):
                        all_dims[f"{mod_name}.{k}"] = round(float(v), 4)

        ears_score = tonos  # kept for debug display; not used in strain formula
    except Exception as e:
        print(f"[EARS] error: {e}")

    # ── Parselmouth (HNR + shimmer) — gated on voiced onset ──────────────────────
    # onset_gated = True for first 2 voiced frames: vocal folds not yet in stable oscillation.
    # Both shimmer and CPP are unreliable at onset — suppress to eliminate phrase-boundary artifacts.
    hnr_db, shimmer_pct, cpp_val = 20.0, float('nan'), float('nan')
    if PRAAT_AVAILABLE:
        try:
            w = audio[-PRAAT_WINDOW_SAMPLES:] if len(audio) >= PRAAT_WINDOW_SAMPLES else audio
            snd = parselmouth.Sound(w.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
            harm = snd.to_harmonicity()
            vals = harm.values[0]
            valid = vals[vals > -200]
            hnr_db = float(np.mean(valid)) if len(valid) > 0 else 20.0
            if not onset_gated and not low_energy:
                pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
                shimmer_pct = parselmouth.praat.call(
                    [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
                ) * 100
        except Exception:
            pass

    if not onset_gated and not low_energy:
        chunk = audio[-PRAAT_WINDOW_SAMPLES:] if len(audio) >= PRAAT_WINDOW_SAMPLES else audio
        cpp_val = _compute_cpp(chunk.astype(np.float64))

    # ── Session-adaptive baseline — continuous EMA from clean frames ───────────
    # Tentative score gates whether this frame can update the baseline.
    # Clean frames (score < MAX_SCORE) slowly pull the baseline toward the singer's
    # actual relaxed voice. Strained frames are excluded — even a beginner who
    # pushes from the very first note won't corrupt the baseline (seed holds).
    # CPP baseline adapts UP freely (louder healthy singing → higher CPP is expected)
    # but adapts DOWN very slowly (prevents quiet/soft phrases from dropping sensitivity).
    global _baseline_clean_n
    global _session_hnr_baseline, _session_shim_baseline  # _session_cpp_baseline declared at function top
    global _session_elast_baseline, _session_anham_baseline, _session_am_baseline
    global _session_alpha_baseline
    shim_for_gate = shimmer_pct if not math.isnan(shimmer_pct) else _session_shim_baseline
    # CPP 3-frame EMA — smooth out phoneme-level CPP dips (raw 100ms CPP varies 0.1–0.6+ within
    # a phrase as phonemes change). Sustained dips = real strain; single-frame transients wash out.
    if not math.isnan(cpp_val):
        _cpp_ema = CPP_EMA_ALPHA * cpp_val + (1 - CPP_EMA_ALPHA) * _cpp_ema
    cpp_for_gate = _cpp_ema if not math.isnan(cpp_val) else _session_cpp_baseline
    _tent_shdev = max(0.0, shim_for_gate - _session_shim_baseline) / 7.0
    _tent_cdev  = max(0.0, _session_cpp_baseline - cpp_for_gate) / 0.35
    _tent_score = min(1.0, max(_tent_shdev, _tent_cdev))
    is_clean_frame = _tent_score < BASELINE_MAX_SCORE and not onset_gated
    if is_clean_frame:
        a = BASELINE_EMA_ALPHA
        _session_hnr_baseline  = (1 - a) * _session_hnr_baseline  + a * hnr_db
        if not math.isnan(shimmer_pct):
            # Asymmetric shimmer: adapts UP normally (increased shimmer = new normal),
            # DOWN very slowly (prevents drift when soft/clean frames over-tune sensitivity).
            if shimmer_pct >= _session_shim_baseline:
                _session_shim_baseline = (1 - a) * _session_shim_baseline + a * shimmer_pct
            else:
                _session_shim_baseline = (1 - 0.01) * _session_shim_baseline + 0.01 * shimmer_pct
        if not math.isnan(cpp_val):
            # Symmetric CPP adaptation — prevents baseline ratcheting from loud clean frames.
            # Asymmetric (UP=0.05, DOWN=0.01) caused +0.187 drift in Liza Jane, inverting verse/chorus.
            _session_cpp_baseline = (1 - 0.03) * _session_cpp_baseline + 0.03 * cpp_val
        _baseline_clean_n += 1

    # ── Deep-analysis v11 signals: elastikos, anharmonia, AM + life/temporal dims ──
    # All HIGHER in strained voice (confirmed by Cohen's d analysis on Liza Jane GT):
    #   elastikos (d=-0.999): energy envelope decay oscillation (touch modality)
    #   anharmonia (d=-0.727): harmonic spacing irregularity (harmonic modality)
    #   am_fast (d=-0.564): Hilbert AM power 10-30Hz in 75-300Hz band (motion amplification)
    #   metabole (d=-0.670): ZCR — temporal change rate (temporal modality)
    #   rhoe_mese: spectral flux mean (temporal modality) — tracks chorus stress
    #   life.metabolism: mel energy throughput rate — elevated in chorus pushing sections
    chunk_for_am = audio[-CHUNK_SAMPLES:] if len(audio) >= CHUNK_SAMPLES else audio
    _acoustic_gated = onset_gated or low_energy  # combined gate: shimmer/CPP/AM/alpha
    # EARS v11 and phonation both need ~2s of ring buffer filled with stable voice before
    # temporal features (elastikos, anharmonia) are reliable. The ring buffer is 1s, but
    # phrase-onset dynamics contaminate elastikos/anharmonia for the first ~1-2s after
    # the ring buffer fills. Use 20-frame (2s) gate.
    # 10-frame (1s) onset gate — sufficient for onset artifacts when combined with the
    # _v11_run_count reset on low_energy (which handles L→active transitions separately).
    # 20-frame gate over-suppresses strain detection in short phrases (< 2s).
    _v11_gated = _v11_run_count < 3   # matches voiced_run onset gate; both lift at 300ms
    elastikos_val  = _session_elast_baseline if _v11_gated else all_dims.get("touch.elastikos", _session_elast_baseline)
    anharmonia_val = _session_anham_baseline if _v11_gated else all_dims.get("harmonic.anharmonia", _session_anham_baseline)
    zcr_val        = 0.0 if _v11_gated else all_dims.get("temporal.metabole", 0.0)   # ZCR
    flux_val       = 0.0 if _v11_gated else all_dims.get("temporal.rhoe_mese", 0.0)  # spectral flux
    metab_val      = 0.0 if _v11_gated else all_dims.get("life.metabolism", 0.0)     # energy throughput
    am_fast_val = _am_fast_power(chunk_for_am) if not _v11_gated else 0.0
    # Alpha ratio: log spectral tilt (higher = pressed phonation). Research rank #1 for singing.
    alpha_val = _alpha_ratio(chunk_for_am) if not _v11_gated else float('nan')
    effort_am = _am_effort_depth(_rms_history) if not _v11_gated else 0.0

    # Baseline update for new signals (same clean-frame gate as v8)
    if is_clean_frame:
        a = BASELINE_EMA_ALPHA
        _session_elast_baseline = (1 - a) * _session_elast_baseline + a * elastikos_val
        _session_anham_baseline = (1 - a) * _session_anham_baseline + a * anharmonia_val
        if am_fast_val > 0:
            _session_am_baseline = (1 - a) * _session_am_baseline + a * am_fast_val
        if not math.isnan(alpha_val):
            _session_alpha_baseline = (1 - a) * _session_alpha_baseline + a * alpha_val
        if effort_am > 0:
            _session_effort_am_baseline = (1 - a) * _session_effort_am_baseline + a * effort_am

    # Score deviations (all increase with strain → deviation above baseline)
    elast_scale = max(0.1, _session_elast_baseline)
    anham_scale = max(0.05, _session_anham_baseline)
    am_scale    = max(1e-4, _session_am_baseline) if _session_am_baseline > 0 else 1e-4
    elast_dev = min(1.0, max(0.0, elastikos_val - _session_elast_baseline) / elast_scale)
    anham_dev = min(1.0, max(0.0, anharmonia_val - _session_anham_baseline) / anham_scale)
    am_dev    = min(1.0, max(0.0, am_fast_val - _session_am_baseline) / (am_scale * 2))
    # ZCR, spectral flux, metabolism — normalize to [0,1] relative change from baseline
    # (no pre-seeded baseline for these; use session EMA — starts at 0, adapts quickly)
    # These are already normalized floats from EARS (0–1 range typically)
    zcr_dev   = min(1.0, max(0.0, zcr_val  - 0.05) / 0.1)   # rough normalization
    flux_dev  = min(1.0, max(0.0, flux_val - 0.1)  / 0.3)
    metab_dev = min(1.0, max(0.0, metab_val - 0.05) / 0.2)
    # Alpha ratio strain score: deviation above session baseline / 6dB scaling
    # 6 dB above baseline = full strain (research: spectral tilt +6dB = clear pressed phonation)
    alpha_for_score = alpha_val if not math.isnan(alpha_val) else _session_alpha_baseline
    alpha_dev = min(1.0, max(0.0, alpha_for_score - _session_alpha_baseline) / 6.0)
    # Effort AM deviation (4-8 Hz laryngeal modulation band — research: diagnostic for vocal effort)
    effort_am_scale = max(1e-5, _session_effort_am_baseline) if _session_effort_am_baseline > 0 else 1e-5
    effort_am_dev   = min(1.0, max(0.0, effort_am - _session_effort_am_baseline) / (effort_am_scale * 2))
    # Combined EARS v11 signal: max of all new features, scaled to supporting role (0.7 cap)
    ears_v11 = min(1.0, max(
        elast_dev, anham_dev, am_dev, zcr_dev, flux_dev, metab_dev,
        alpha_dev, effort_am_dev
    )) * 0.7

    baseline_warm = _baseline_clean_n >= BASELINE_WARM_N

    # ── Strain formula v8 — dual-signal: max(shim_dev, cpp_dev) ────────────────
    # shim_dev: shimmer spike above baseline → rough phonation
    # cpp_dev:  CPP drop below baseline → phonatory irregularity / tightness / constriction
    #           Loudness-robust: loud+healthy → higher CPP → cpp_dev stays 0
    # HNR excluded — rises for BOTH pressed strain AND relaxed high notes → unreliable
    shim_dev = max(0.0, shim_for_gate - _session_shim_baseline) / 7.0   # was /10 — 7% above baseline = full strain (more sensitive)
    cpp_dev  = max(0.0, _session_cpp_baseline - cpp_for_gate) / 0.35   # was /0.5 — 0.35 CPP drop = full strain (more sensitive)
    v8_strain = min(1.0, max(shim_dev, cpp_dev))

    # ── Wavelet scatter v9 — session-adaptive multi-scale modulation baseline ──
    # Detects changes in amplitude modulation patterns across time scales (J=7 octaves).
    # Session-adaptive: baseline built from v8-gated clean frames within this session.
    # Falls back to v8-only until SCATTER_WARM_N clean frames collected (~5-10s).
    # Max-fusion: fuse = (1-w)*max(v8, scatter) + w*wavg(v8, scatter)
    # This ensures EITHER signal alone can detect strain, unlike pure weighted average.
    global _scatter_baseline_feats, _scatter_mean, _scatter_std
    scatter_score = 0.0
    scatter_feat = _scatter_features(audio[-CHUNK_SAMPLES:] if len(audio) >= CHUNK_SAMPLES else audio)
    if scatter_feat is not None:
        if _scatter_mean is not None and not _v11_gated:
            # Gate scatter with same 20-frame v11 gate: scatter wavelet features capture
            # multi-scale AM patterns that are contaminated by silence→voice ring buffer
            # transitions, identically to elastikos/anharmonia. Without this gate,
            # scatter fires at onset with scores 0.5-0.7 even on relaxed GREEN phrases.
            scatter_score = _scatter_strain_score(scatter_feat)
            # EMA update — only very clean scatter frames contribute
            if scatter_score < 0.3:
                _scatter_mean = (1 - SCATTER_EMA_ALPHA) * _scatter_mean + SCATTER_EMA_ALPHA * scatter_feat
        elif is_clean_frame:
            _scatter_baseline_feats.append(scatter_feat)
            if len(_scatter_baseline_feats) >= SCATTER_WARM_N:
                feats_arr = np.stack(_scatter_baseline_feats)
                _scatter_mean = np.mean(feats_arr, axis=0)
                _scatter_std  = np.std(feats_arr, axis=0) + 1e-8
                # Note: _log() uses asyncio.get_event_loop() which fails in executor threads
                # (Python 3.12+ raises RuntimeError in non-main threads). Use print() here.
                print(f"[INFO] Scatter baseline warm ({SCATTER_WARM_N} clean frames)")

    # ── Phonation classifier (v10) — trained on CVT dataset, detects pressed/hyperadducted phonation ──
    # Identifies overdrive/edge CVT modes (90.2% 4-class, 95.7% binary acc on 13K samples).
    # Fills the "hard push miss" gap: pressed voice has HIGH CPP/low shimmer → v8 won't flag it,
    # but the classifier learned modulation patterns that distinguish pressed from modal.
    # Only active when scatter features are available (same feature pipeline as training).
    chunk_now = audio[-CHUNK_SAMPLES:] if len(audio) >= CHUNK_SAMPLES else audio
    # Phonation classifier needs ~1s to see stable scatter features — the ring buffer
    # transitions from silence to voice over ~10 frames and the classifier fires false
    # positives during that window even after the general onset gate has lifted.
    phonation_gated = _voiced_run_count < 5 or low_energy
    phonation_score = 0.0 if phonation_gated else _phonation_score(chunk_now)

    # Max-blend fusion — v8 + scatter + phonation + EARS v11 (elastikos/anharmonia/AM)
    # Any signal alone → strain. ears_v11 capped at 0.7 to keep supporting role.
    if _scatter_mean is not None and scatter_feat is not None:
        w = SCATTER_FUSION_W
        max_s  = max(v8_strain, scatter_score, phonation_score, ears_v11)
        wavg_s = (v8_strain + scatter_score + phonation_score + ears_v11) / 4.0
        strain = min(1.0, (1 - w) * max_s + w * wavg_s)
    else:
        # Scatter not yet warmed — v8 + phonation + ears_v11
        strain = min(1.0, max(v8_strain, phonation_score * 0.7, ears_v11))

    zone = "green" if strain < STRAIN_GREEN else "yellow" if strain < STRAIN_YELLOW else "red"

    # ── Breathy detection (separate axis from strain) ──────────────────────────
    hnr_press   = max(0.0, hnr_db - _session_hnr_baseline) / 10.0  # debug only
    hnr_drop    = max(0.0, _session_hnr_baseline - hnr_db) / 10.0  # HNR below baseline
    breathy_score = hnr_drop * 0.7 + shim_dev * 0.3
    is_breathy    = breathy_score > 0.25 and strain < STRAIN_YELLOW

    latency_ms = round((time.time() - t0) * 1000)
    _analysis_latencies.append(latency_ms)

    return {
        "active": True,
        "rms": round(rms_val, 5),
        "strain_score": round(strain, 3),
        "zone": zone,
        "breathy_score": round(breathy_score, 3),
        "is_breathy": is_breathy,
        "tonos": round(tonos, 3),
        "thymos": round(thymos, 3),
        "trachytes": round(trachytes, 3),
        "kallos": round(kallos, 3),
        "hnr_db": round(hnr_db, 1),
        "shimmer_pct": round(shimmer_pct, 2) if not math.isnan(shimmer_pct) else 0.0,
        "hnr_press": round(hnr_press, 3),          # debug only
        "shim_dev": round(shim_dev, 3),            # rough phonation signal
        "cpp_db": round(cpp_for_gate, 3),          # CPP value (higher = healthier)
        "cpp_dev": round(cpp_dev, 3),              # tightness/irregularity signal
        "v8_strain": round(v8_strain, 3),          # v8-only score (debug comparison)
        "scatter_score": round(scatter_score, 3),  # wavelet scatter strain score
        "phonation_score": round(phonation_score, 3),  # classifier: pressed/overdrive probability
        "ears_v11": round(ears_v11, 3),            # EARS v11: elastikos/anharmonia/AM/ZCR/flux
        "elast_dev": round(elast_dev, 3),          # elastikos deviation (d=-0.999)
        "anham_dev": round(anham_dev, 3),          # anharmonia deviation (d=-0.727)
        "am_dev": round(am_dev, 3),                # AM fast modulation (d=-0.564)
        "zcr_dev": round(zcr_dev, 3),              # ZCR/metabole deviation (d=-0.670)
        "flux_dev": round(flux_dev, 3),            # spectral flux deviation
        "metab_dev": round(metab_dev, 3),          # life.metabolism deviation
        "alpha_dev": round(alpha_dev, 3),          # alpha ratio deviation (Sol 2023 #1 feature)
        "alpha_db": round(alpha_for_score, 2),     # raw alpha ratio in dB
        "effort_am": round(effort_am_dev, 3),      # 4-8Hz AM effort band (laryngeal modulation)
        "scatter_warm": _scatter_mean is not None, # True = scatter baseline active
        "onset_gated": onset_gated,               # True = first 2 frames, signals suppressed
        "low_energy": low_energy,                 # True = phrase tail, shimmer/CPP suppressed
        "voiced_run": _voiced_run_count,          # frames since last silence
        "hnr_baseline": round(_session_hnr_baseline, 1),
        "shim_baseline": round(_session_shim_baseline, 2),
        "cpp_baseline": round(_session_cpp_baseline, 3),
        "baseline_warm": baseline_warm,
        "baseline_clean_n": _baseline_clean_n,
        "ears_score": round(ears_score, 3),
        "latency_ms": latency_ms,
        "all_dims": all_dims,  # full dim snapshot for debug
    }


def _compute_pitch(audio: np.ndarray) -> dict:
    """YIN on last 250ms + EMA smoothing + note stability gate.
    EMA smooths frame-to-frame Hz jitter (vibrato, breath noise).
    Note gate requires 2 consecutive frames showing the same note before committing.
    """
    global _pitch_ema_hz, _pitch_last_note, _pitch_stable_note, _pitch_note_count, _vibrato_hz_buf, _vibrato_hold_count
    blank = {"pitch_hz": 0.0, "pitch_note": "—", "pitch_cents_off": 0, "vibrato": False}
    try:
        # Use last 500ms — longer window = more stable pitch estimate
        window = audio[-22050:] if len(audio) >= 22050 else audio
        f0 = librosa.yin(
            window,
            fmin=60.0,
            fmax=1200.0,
            sr=SAMPLE_RATE,
            hop_length=441,
        )
        # Filter out silence/noise frames (yin returns fmin for unvoiced)
        valid_f0 = f0[(f0 > 70) & (f0 < 1100)]
        if len(valid_f0) == 0:
            _pitch_ema_hz = 0.0
            _pitch_last_note = "—"
            _pitch_stable_note = "—"
            _pitch_note_count = 0
            _vibrato_hz_buf.clear()
            _vibrato_hold_count = 0
            return blank

        raw_hz = float(np.median(valid_f0))
        if raw_hz <= 0 or math.isnan(raw_hz):
            return blank

        # EMA smoothing — reduces Hz jitter from vibrato/breath noise
        if _pitch_ema_hz <= 0:
            _pitch_ema_hz = raw_hz   # cold start: seed immediately
        else:
            _pitch_ema_hz = PITCH_EMA_ALPHA * raw_hz + (1 - PITCH_EMA_ALPHA) * _pitch_ema_hz

        midi = 12 * math.log2(_pitch_ema_hz / 440.0) + 69
        nearest_midi = round(midi)
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        candidate_note = f"{note_names[nearest_midi % 12]}{(nearest_midi // 12) - 1}"
        cents_off = round((midi - nearest_midi) * 100)

        # Note stability gate — only commit to a new note after 3 consecutive frames (300ms)
        if candidate_note == _pitch_last_note:
            _pitch_note_count += 1
        else:
            _pitch_note_count = 1
        _pitch_last_note = candidate_note  # always track what we're seeing
        if _pitch_note_count >= 3:
            _pitch_stable_note = candidate_note  # commit: stable for 3+ frames
        else:
            candidate_note = _pitch_stable_note  # not yet stable — show last stable

        # Vibrato detection — periodic pitch oscillation around EMA
        # Use EMA-smoothed Hz for vibrato detection — raw Hz has jitter that causes false crossings
        _vibrato_hz_buf.append(_pitch_ema_hz)
        if len(_vibrato_hz_buf) > VIBRATO_BUF_SIZE:
            _vibrato_hz_buf.pop(0)

        detected_now = False
        if len(_vibrato_hz_buf) >= 8:  # need at least 800ms of data (~2 vibrato cycles)
            mean_hz = sum(_vibrato_hz_buf) / len(_vibrato_hz_buf)
            if mean_hz > 0:
                # Convert deviations to cents (1200 * log2(f/ref))
                deviations_cents = [1200 * math.log2(h / mean_hz) for h in _vibrato_hz_buf]
                amplitude = max(deviations_cents) - min(deviations_cents)
                # Count zero-crossings of deviation (sign changes = periodic oscillation)
                crossings = sum(
                    1 for i in range(1, len(deviations_cents))
                    if deviations_cents[i - 1] * deviations_cents[i] < 0
                )
                # Vibrato rate check: real vibrato is 4-7 Hz.
                # At 10Hz sample rate, crossings per second = crossings / (buf_len * 0.1) * 0.5
                # (each full cycle = 2 crossings). Rate = crossings / (2 * duration_s)
                buf_duration_s = len(_vibrato_hz_buf) * 0.1
                vibrato_rate_hz = crossings / (2.0 * buf_duration_s)
                rate_ok = 3.0 <= vibrato_rate_hz <= 8.0  # allow slight margin around 4-7 Hz
                detected_now = amplitude >= VIBRATO_MIN_CENTS and crossings >= VIBRATO_MIN_CROSSINGS and rate_ok

        # Hold vibrato state for VIBRATO_HOLD_FRAMES after last detection (prevents flicker)
        if detected_now:
            _vibrato_hold_count = VIBRATO_HOLD_FRAMES
        elif _vibrato_hold_count > 0:
            _vibrato_hold_count -= 1
        has_vibrato = _vibrato_hold_count > 0

        return {
            "pitch_hz": round(_pitch_ema_hz, 1),
            "pitch_note": candidate_note,
            "pitch_cents_off": cents_off,
            "vibrato": has_vibrato,
        }
    except Exception:
        return blank

# ─── Phrase detector ──────────────────────────────────────────────────────────

class PhraseDetector:
    def __init__(self):
        self._active = False
        self._start_t = 0.0
        self._silence_since = 0.0
        self._strain_buf: list[float] = []

    def update(self, is_active: bool, strain: float, now: float) -> Optional[str]:
        if not self._active and is_active:
            self._active = True
            self._start_t = now
            self._silence_since = 0.0
            self._strain_buf = [strain]
            return "phrase_start"

        if self._active:
            if not is_active:
                if self._silence_since == 0.0:
                    self._silence_since = now
                silence_dur = now - self._silence_since
                phrase_dur = now - self._start_t
                if silence_dur >= MIN_SILENCE_S and phrase_dur >= MIN_PHRASE_S:
                    self._active = False
                    self._silence_since = 0.0
                    return "phrase_end"
            else:
                self._silence_since = 0.0
                self._strain_buf.append(strain)
        return None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phrase_avg_strain(self) -> float:
        return float(np.mean(self._strain_buf)) if self._strain_buf else 0.0

    @property
    def phrase_duration(self) -> float:
        return time.time() - self._start_t if self._active else 0.0

# ─── Broadcast helpers ────────────────────────────────────────────────────────

async def _broadcast_singer(event: dict):
    """Send to all singer UI clients (strip debug-only fields)."""
    msg = {k: v for k, v in event.items() if k != "all_dims"}
    text = json.dumps(msg)
    dead = []
    for ws in list(_singer_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _singer_clients:
            _singer_clients.remove(ws)

async def _broadcast_debug(event: dict):
    """Send to all debug clients (full data including all_dims)."""
    text = json.dumps(event)
    dead = []
    for ws in list(_debug_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _debug_clients:
            _debug_clients.remove(ws)

async def _broadcast_all(event: dict):
    await asyncio.gather(
        _broadcast_singer(event),
        _broadcast_debug(event),
    )

# ─── Main analysis loop ───────────────────────────────────────────────────────

_phrase = PhraseDetector()

async def _song_end_timer():
    """Wait SONG_END_SILENCE_S then fire end-of-song praise if session was mostly green."""
    global _song_end_timer_task
    try:
        await asyncio.sleep(SONG_END_SILENCE_S)
        if _session_phrase_count < MIN_PHRASES_FOR_PRAISE:
            return
        yellow_pct = 100.0 * _session_yellow_red_count / _session_phrase_count
        tier = None
        for t in sorted(_SONG_PRAISE_TIERS, key=lambda x: x["threshold"]):
            if yellow_pct <= t["threshold"]:
                tier = t["tier"]
                break
        if tier is None:
            tier = "solid"  # fallback for high yellow/red sessions
        _log(f"Song end: {tier} ({yellow_pct:.0f}% yellow/red over {_session_phrase_count} phrases)")
        asyncio.create_task(
            _gemini_coach.praise_song(tier, yellow_pct, _session_phrase_count)
        )
    except asyncio.CancelledError:
        pass  # singing resumed — timer cancelled, no praise
    finally:
        _song_end_timer_task = None

async def _analysis_loop():
    global _ring, _session_history, _frame_count, _ema_strain
    global _session_phrase_count, _session_yellow_red_count, _song_end_timer_task
    global _consecutive_yellow

    chunk_acc = np.zeros(0, dtype=np.float32)
    filled = False

    _log("Analysis loop started — warming up (1s)")

    while True:
        chunks = []
        try:
            while True:
                chunks.append(_audio_q.get_nowait())
        except queue.Empty:
            pass

        if not chunks:
            await asyncio.sleep(0.005)
            continue

        chunk_acc = np.concatenate([chunk_acc] + chunks)

        while len(chunk_acc) >= CHUNK_SAMPLES:
            cur = chunk_acc[:CHUNK_SAMPLES]
            chunk_acc = chunk_acc[CHUNK_SAMPLES:]

            _ring = np.roll(_ring, -CHUNK_SAMPLES)
            _ring[-CHUNK_SAMPLES:] = cur
            _frame_count += 1

            if not filled:
                # Start analysis after 300ms (3 frames) — the Praat window only needs 200ms.
                if _frame_count >= 3:
                    filled = True
                    _log("Ring buffer full — analysis active")
                else:
                    continue

            audio_window = _ring.copy()
            now = time.time()

            loop = asyncio.get_running_loop()
            strain_task = loop.run_in_executor(_executor, _compute_strain, audio_window)
            pitch_task = loop.run_in_executor(_executor, _compute_pitch, audio_window)
            strain_metrics, pitch_metrics = await asyncio.gather(strain_task, pitch_task)

            voice_active = strain_metrics["active"]
            raw_strain = strain_metrics["strain_score"]

            # EMA smoothing — reduces per-frame jitter without lag artifacts
            if voice_active:
                _ema_strain = EMA_ALPHA * raw_strain + (1.0 - EMA_ALPHA) * _ema_strain
            else:
                _ema_strain = max(0.0, _ema_strain - 0.02)  # decay toward 0 during silence

            smoothed_strain = round(_ema_strain, 3)
            smoothed_zone = "green" if smoothed_strain < STRAIN_GREEN else "yellow" if smoothed_strain < STRAIN_YELLOW else "red"
            if not voice_active:
                smoothed_zone = "idle"

            phrase_event = _phrase.update(voice_active, smoothed_strain, now)

            # Session history (only active frames)
            if voice_active:
                _session_history.append(smoothed_strain)
                if len(_session_history) > 600:
                    _session_history = _session_history[-600:]

                # Range mapper: accumulate per-note strain
                note = pitch_metrics.get("pitch_note", "—")
                if note and note != "—":
                    _note_strain.setdefault(note, []).append(smoothed_strain)
                    _range_active_frames_local = len(_session_history)
                    if _range_active_frames_local % RANGE_BROADCAST_EVERY == 0:
                        range_data = {
                            n: {"avg": round(float(np.mean(s)), 3), "count": len(s)}
                            for n, s in _note_strain.items()
                            if len(s) >= MIN_FRAMES_PER_NOTE
                        }
                        await _broadcast_all({"type": "range_update", "notes": range_data})

            # Broadcast frame
            avg_latency = round(sum(_analysis_latencies) / len(_analysis_latencies)) if _analysis_latencies else 0
            frame_event = {
                "type": "ears_frame",
                "timestamp": now,
                "session_t": round(now - _session_start, 1),
                "frame": _frame_count,
                "phrase_active": _phrase.active,
                "avg_latency_ms": avg_latency,
                "singers_connected": len(_singer_clients),
                **strain_metrics,
                "strain_score": smoothed_strain,  # override with smoothed value
                "zone": smoothed_zone,
                "strain_raw": round(raw_strain, 3),
                **pitch_metrics,
            }
            await _broadcast_all(frame_event)

            # Phrase events
            if phrase_event == "phrase_start":
                evt = {"type": "phrase_start", "timestamp": now}
                await _broadcast_all(evt)
                _log("Phrase started")
                # Cancel any pending song-end timer — singing resumed
                if _song_end_timer_task and not _song_end_timer_task.done():
                    _song_end_timer_task.cancel()
                    _song_end_timer_task = None

            elif phrase_event == "phrase_end":
                avg = _phrase.phrase_avg_strain
                dur = _phrase.phrase_duration + MIN_SILENCE_S
                zone = "green" if avg < STRAIN_GREEN else "yellow" if avg < STRAIN_YELLOW else "red"
                was_breathy = strain_metrics.get("is_breathy", False)
                evt = {
                    "type": "phrase_end",
                    "timestamp": now,
                    "phrase_avg_strain": round(avg, 3),
                    "phrase_duration_s": round(dur, 1),
                    "zone": zone,
                    "was_breathy": was_breathy,
                }
                await _broadcast_all(evt)
                phrase_rec = {**evt, "ts": round(now - _session_start, 1)}
                _phrase_history.append(phrase_rec)
                # Track session stats for song-end praise
                _session_phrase_count += 1
                if zone in ("yellow", "red"):
                    _session_yellow_red_count += 1
                # ─── Feedback ladder ──────────────────────────────────────────
                # Tier 1 (silent):  green → reset counter, no output beyond zone meter
                # Tier 2 (visual):  1st or 2nd consecutive yellow → text cue only, no voice
                # Tier 3 (voice):   red, 3+ consecutive yellows, or breathy → Koda speaks
                if zone == "green":
                    _consecutive_yellow = 0
                    # Warm-up mode: active coaching on green phrases
                    # During warm-up (first 10 phrases): coach every 2nd green phrase
                    # After warm-up: coach every 4th green phrase
                    warmup_active = _session_phrase_count <= 10
                    green_interval = 2 if warmup_active else 4
                    if _session_phrase_count > 0 and _session_phrase_count % green_interval == 0:
                        asyncio.create_task(
                            _gemini_coach.coach_phrase("green", avg, dur, list(_phrase_history), False)
                        )
                elif zone == "yellow" and not was_breathy:
                    _consecutive_yellow += 1
                elif zone == "red":
                    _consecutive_yellow = 0  # red resets the counter

                # Demo mode: Gemini speaks on EVERY yellow or red phrase so judges hear it.
                # Short phrases keep Gemini responses brief and natural-sounding.
                use_voice = zone in ("yellow", "red") or was_breathy
                use_visual = False  # skip visual-only tier — always use voice for yellow+

                _log(f"Phrase ended: {zone.upper()} avg={avg:.3f} dur={dur:.1f}s "
                     f"consec_y={_consecutive_yellow} → {'VOICE' if use_voice else 'VISUAL' if use_visual else 'silent'}"
                     + (" [breathy]" if was_breathy else ""))

                if use_visual:
                    # Tier 2: visual-only hint — pick cue text based on strain level
                    cue = "Feeling some push there — soften the jaw and let it float."
                    if avg > STRAIN_YELLOW * 0.85:
                        cue = "Getting close to the edge — ease off the pressure before the next phrase."
                    await _broadcast_all({"type": "visual_cue", "zone": "yellow",
                                          "text": cue, "consecutive": _consecutive_yellow})

                if use_voice:
                    asyncio.create_task(
                        _gemini_coach.coach_phrase(zone, avg, dur, list(_phrase_history), was_breathy)
                    )
                # Start song-end timer (cancelled if singing resumes within SONG_END_SILENCE_S)
                if _song_end_timer_task and not _song_end_timer_task.done():
                    _song_end_timer_task.cancel()
                _song_end_timer_task = asyncio.create_task(_song_end_timer())

            # Session update every 10 active frames
            if len(_session_history) > 0 and len(_session_history) % 10 == 0:
                await _broadcast_all({
                    "type": "session_update",
                    "strain_history": [round(x, 3) for x in _session_history[-120:]],
                    "phrase_history": list(_phrase_history)[-10:],
                })

# ─── FastAPI ──────────────────────────────────────────────────────────────────

async def _safe_gemini_connect():
    """Connect to Gemini in background — don't block server startup."""
    try:
        await _gemini_coach.connect()
    except Exception as e:
        print(f"[WARN] Gemini connect failed on startup (will retry on first phrase): {e}")

@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    _log("Server starting up")
    task = asyncio.create_task(_analysis_loop())
    task.add_done_callback(lambda t: t.result() if not t.cancelled() else None)
    # Connect Gemini in background — don't block server startup (Cloud Run health check)
    asyncio.create_task(_safe_gemini_connect())
    yield
    _log("Server shutting down")
    await _gemini_coach.disconnect()

app = FastAPI(title="Vocal Health Coach", lifespan=lifespan)

# ─── Static pages ─────────────────────────────────────────────────────────────

@app.get("/")
async def singer_ui():
    p = Path(__file__).parent.parent / "frontend" / "index.html"
    return HTMLResponse(p.read_text() if p.exists() else "<h1>Singer UI not found</h1>")

@app.get("/debug")
async def debug_ui():
    p = Path(__file__).parent.parent / "frontend" / "debug.html"
    return HTMLResponse(p.read_text() if p.exists() else "<h1>Debug UI not found</h1>")

@app.get("/test")
async def test_ui():
    p = Path(__file__).parent.parent / "frontend" / "test.html"
    return HTMLResponse(p.read_text() if p.exists() else "<h1>Test UI not found</h1>")

@app.get("/recordings")
async def list_recordings():
    rec_dir = Path(__file__).parent.parent / "Vocal test recording sessions"
    files = []
    if rec_dir.exists():
        files = sorted([f.name for f in rec_dir.iterdir()
                        if f.suffix.lower() in ('.m4a', '.wav', '.mp3', '.aac', '.ogg')])
    return JSONResponse({"files": files})

@app.get("/recordings/{filename}")
async def serve_recording(filename: str):
    # Security: no path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    rec_dir = Path(__file__).parent.parent / "Vocal test recording sessions"
    path = rec_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(str(path))

# ─── Singer WebSocket (audio in + results out) ────────────────────────────────

@app.websocket("/ws")
async def singer_ws(websocket: WebSocket):
    global _ring, _ema_strain
    await websocket.accept()
    _singer_clients.append(websocket)
    # Clear ring buffer and reset adaptive baseline for new session
    global _ring, _ema_strain, _baseline_clean_n, _voiced_run_count, _v11_run_count
    global _session_hnr_baseline, _session_shim_baseline, _session_cpp_baseline
    global _scatter_baseline_feats, _scatter_mean, _scatter_std, _cpp_ema
    global _silence_frames, _pitch_ema_hz, _pitch_last_note, _pitch_note_count, _vibrato_hz_buf
    global _note_strain, _session_phrase_count, _session_yellow_red_count
    _ring = np.zeros(EARS_WINDOW_SAMPLES, dtype=np.float32)
    _ema_strain = 0.0
    _session_hnr_baseline  = SEED_HNR_BASELINE
    _session_shim_baseline = SEED_SHIM_BASELINE
    _session_cpp_baseline  = SEED_CPP_BASELINE
    _baseline_clean_n = 0
    _voiced_run_count = 0
    _v11_run_count = 0
    _silence_frames = 0
    _cpp_ema = SEED_CPP_BASELINE
    _pitch_ema_hz = 0.0
    _pitch_last_note = "—"
    _pitch_stable_note = "—"
    _pitch_note_count = 0
    _vibrato_hz_buf.clear()
    _vibrato_hold_count = 0
    _scatter_baseline_feats.clear()
    _scatter_mean = None
    _scatter_std  = None
    _note_strain.clear()
    _session_phrase_count = 0
    _session_yellow_red_count = 0
    _log(f"Singer connected — ring buffer + adaptive baseline reset. Total: {len(_singer_clients)}")

    await websocket.send_text(json.dumps({
        "type": "config",
        "sample_rate": SAMPLE_RATE,
        "praat_available": PRAAT_AVAILABLE,
        "strain_thresholds": {"green": STRAIN_GREEN, "yellow": STRAIN_YELLOW},
        "silence_rms": SILENCE_RMS,
    }))

    # Gemini greeting — let the user hear Koda immediately on connect
    asyncio.create_task(_gemini_greet())

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                # Audio data: Float32 PCM from browser
                if len(message["bytes"]) > 352800:  # cap at 2s of 44100Hz float32
                    continue
                audio_chunk = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                try:
                    _audio_q.put_nowait(audio_chunk)
                except queue.Full:
                    _audio_q.get_nowait()  # drop oldest
                    _audio_q.put_nowait(audio_chunk)
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif msg.get("type") == "set_sample_rate":
                        pass  # future: dynamic resampling
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERROR] Singer WS error: {e}")
    finally:
        if websocket in _singer_clients:
            _singer_clients.remove(websocket)
        _log(f"Singer disconnected — total: {len(_singer_clients)}")

# ─── Audio Injection WebSocket (for Reel pipeline — no session reset) ─────────

@app.websocket("/inject/ws")
async def inject_ws(websocket: WebSocket):
    """Audio-only injection endpoint. Does NOT reset session state or join singer_clients.
    Used by Reel pipeline: browser connects via /ws (gets events), inject script
    connects here (sends audio only). No interference between the two.
    """
    await websocket.accept()
    _log("Audio injection client connected")
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                if len(message["bytes"]) > 352800:
                    continue
                audio_chunk = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                try:
                    _audio_q.put_nowait(audio_chunk)
                except queue.Full:
                    _audio_q.get_nowait()
                    _audio_q.put_nowait(audio_chunk)
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERROR] Inject WS error: {e}")
    finally:
        _log("Audio injection client disconnected")

# ─── Debug WebSocket (read-only, receives all events + logs) ──────────────────

@app.websocket("/debug/ws")
async def debug_ws(websocket: WebSocket):
    await websocket.accept()
    _debug_clients.append(websocket)
    _log(f"Debug client connected — total: {len(_debug_clients)}")

    # Send current state snapshot on connect
    await websocket.send_text(json.dumps({
        "type": "snapshot",
        "session_history": _session_history[-120:],
        "phrase_history": list(_phrase_history),
        "log_buffer": list(_log_buffer)[-50:],
        "uptime_s": round(time.time() - _session_start, 1),
        "frame_count": _frame_count,
        "singers_connected": len(_singer_clients),
        "praat_available": PRAAT_AVAILABLE,
    }))

    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=60)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        if websocket in _debug_clients:
            _debug_clients.remove(websocket)

# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _session_start, 1),
        "frames": _frame_count,
        "singers": len(_singer_clients),
        "debug_clients": len(_debug_clients),
        "praat": PRAAT_AVAILABLE,
        "gemini": _gemini_coach.connected,
    }

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vocal Health Coach Backend")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                ws_ping_interval=20, ws_ping_timeout=10)
