"""
Entry 1: Musician/Room Perception Agent
Gemini Live Agent Challenge 2026

"Gemini hears your words. EARS hears you."

Architecture:
  - Gemini Live API: native audio understanding + conversation
  - EARS (proprietary): 172-dimension acoustic perception
    - Room model: RT60, DRR, material signatures
    - Player dimensions: tonos, trachytes, kallos, cheimon, thymos, etc.
  - EARS exposed as a Gemini tool — agent calls it proactively

Usage:
  python src/agent_entry1.py                    # live mic
  python src/agent_entry1.py --file audio.wav   # file mode (testing)
"""

import asyncio
import sys
import os
import json
import time
import threading
import queue
import wave
import argparse
import numpy as np
from pathlib import Path

# ─── Path setup ───
PROJECT_ROOT = Path(__file__).parent.parent
EARS_PATH = PROJECT_ROOT / "ears"
AUDIO_PERCEPTION_SRC = Path("/Users/daniel/Documents/projects/audio-perception/src")

# EARS imports — try local ears/ first (for deployment), fall back to dev path
if EARS_PATH.exists() and any(EARS_PATH.iterdir()):
    sys.path.insert(0, str(EARS_PATH))
else:
    sys.path.insert(0, str(AUDIO_PERCEPTION_SRC))

# Also add frequency explorer scripts
SCRIPTS_PATH = Path("/Users/daniel/Documents/projects/audio-perception/scripts")
AUDIO_PERCEPTION_ROOT = Path("/Users/daniel/Documents/projects/audio-perception")
sys.path.insert(0, str(AUDIO_PERCEPTION_ROOT))

from google import genai
from google.genai import types

# ─── Config ───
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")
MODEL = "models/gemini-2.5-flash-native-audio-latest"
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100          # 100ms audio chunks to Gemini
CHUNK_SAMPLES = SAMPLE_RATE // 10
EARS_UPDATE_INTERVAL = 2.0       # seconds between EARS analyses
OUTPUT_SAMPLE_RATE = 24000       # Gemini outputs at 24kHz

SYSTEM_PROMPT = """You are a musician and room perception coach powered by EARS — a proprietary 172-dimension audio perception system.

You have access to a tool called `get_ears_perception` that returns precise acoustic measurements of:
- The room: reverberation time (RT60), direct-to-reverberant ratio (DRR), confidence
- The player's sound: tonos (tension/pitch center), trachytes (surface roughness/clarity), kallos (beauty/harmonic richness), cheimon (storm energy/intensity), thymos (life force), and other Greek-named perceptual dimensions

Unlike other tools that only transcribe or label audio semantically, EARS measures the physics of sound — giving you exact distances and directions, not just right/wrong judgments.

Your job:
1. Listen to the musician playing
2. Call get_ears_perception() to get precise dimensional data about what you hear
3. Give specific, honest, useful feedback — use the actual numbers
4. Distinguish between what the PLAYER is doing vs what the ROOM is adding
5. Speak naturally and conversationally — you're a coach, not a report generator

Example response style: "Your tone is genuinely beautiful right now — kallos at 0.74, which means rich harmonics with good sustain. The room is adding about 0.3 seconds of reverb at mid frequencies, which is actually working in your favor here. I notice your trachytes is very low at 0.04 — that's exceptionally clean attack. What are you going for with this piece?"

Always use get_ears_perception() before making specific claims about the sound."""


# ─── EARS Bridge ───

class EARSBridge:
    """Wraps EARS streaming pipeline + frequency explorer for Gemini tool calls."""

    def __init__(self):
        self._audio_buffer = []  # recent audio chunks (numpy arrays)
        self._buffer_lock = threading.Lock()
        self._last_perception = {}
        self._room_model = None
        self._init_ears()

    def _init_ears(self):
        """Initialize EARS components."""
        try:
            from room_model import RoomModelEstimator
            self._room_model = RoomModelEstimator(sample_rate=SAMPLE_RATE)
            print("[EARS] Room model initialized")
        except ImportError as e:
            print(f"[EARS] Warning: room model not available ({e})")

    def push_audio(self, chunk: np.ndarray):
        """Add audio chunk to buffer. Called by audio capture thread."""
        with self._buffer_lock:
            self._audio_buffer.append(chunk.copy())
            # Keep ~4 seconds
            max_chunks = int(4.0 * SAMPLE_RATE / len(chunk))
            if len(self._audio_buffer) > max_chunks:
                self._audio_buffer = self._audio_buffer[-max_chunks:]

    def get_buffered_audio(self, seconds: float = 2.0) -> np.ndarray:
        """Get last N seconds of audio as a single array."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
            all_audio = np.concatenate(self._audio_buffer)
            n_samples = int(seconds * SAMPLE_RATE)
            if len(all_audio) >= n_samples:
                return all_audio[-n_samples:]
            return all_audio

    def analyze(self) -> dict:
        """Run EARS analysis on current audio buffer. Returns structured perception dict."""
        import warnings
        warnings.filterwarnings("ignore")

        audio = self.get_buffered_audio(seconds=2.0)
        result = {
            "timestamp": time.time(),
            "room": {},
            "player": {},
            "confidence": 0.0,
        }

        # ─── Room model ───
        if self._room_model is not None and len(audio) > SAMPLE_RATE // 2:
            try:
                room_state, _ = self._room_model.update(audio)  # returns (RoomState, timestamp)
                if room_state is not None:
                    rd = room_state.to_dict()
                    rt60_vals = [rd['rt60'].get(k, 0.3) for k in ['250Hz', '500Hz', '1000Hz', '2000Hz']]
                    result["room"] = {
                        "rt60_mid_s": round(float(np.mean(rt60_vals)), 3),
                        "rt60_by_band": rd['rt60'],
                        "drr_db": rd['drr_db'],
                        "confidence": rd['confidence'],
                        "interpretation": _interpret_room(rd),
                    }
            except Exception as e:
                result["room"]["error"] = str(e)

        # ─── Greek dimensional analysis (frequency_explorer) ───
        try:
            from scripts.frequency_explorer import analyze_mel

            mel = _audio_to_mel(audio)
            if mel is not None and mel.shape[0] > 5:
                analysis = analyze_mel(mel)
                mods = analysis.get("modalities", {})
                fp = analysis.get("fingerprint", {})

                def g(d, k, default=0.0):
                    return round(float(d.get(k, default)), 4)

                emotion = mods.get("emotion", {})
                touch = mods.get("touch", {})
                weather = mods.get("weather", {})
                life = mods.get("life", {})
                acoustic = mods.get("acoustic", {})
                harmonic = mods.get("harmonic", {})

                result["player"] = {
                    # Core tonal dimensions
                    "tonos": g(emotion, "tonos"),          # tension/tone
                    "kallos": g(emotion, "kallos"),        # beauty/harmonic richness
                    "thymos": g(emotion, "thymos"),        # spirit/arousal
                    "kratos": g(emotion, "kratos"),        # power/dominance
                    "hedone": g(emotion, "hedone"),        # pleasure/valence
                    # Touch dimensions
                    "trachytes": g(touch, "trachytes"),    # roughness/clarity
                    "sklerotes": g(touch, "sklerotes"),    # hardness/attack
                    "baros": g(touch, "baros"),            # weight/density
                    # Weather/energy
                    "cheimon": g(weather, "cheimon"),      # storm intensity
                    "anemos": g(weather, "anemos"),        # wind/movement
                    # Acoustic
                    "kentron": g(acoustic, "kentron"),     # spectral center
                    "metabole": g(acoustic, "metabole"),   # flux/change rate
                    "harmonia": g(harmonic, "harmonia"),   # harmonic quality
                    # Interpretations
                    "interpretations": {
                        "trachytes": _interpret_trachytes(touch.get("trachytes", 0.5)),
                        "kallos": _interpret_kallos(emotion.get("kallos", 0.5)),
                        "cheimon": _interpret_cheimon(weather.get("cheimon", 0.5)),
                    }
                }
                result["confidence"] = 0.85
        except Exception as e:
            result["player"]["error"] = str(e)
            result["confidence"] = 0.0

        self._last_perception = result
        return result


def _audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio to log-mel spectrogram (T, 80)."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=80,
            n_fft=2048, hop_length=160, fmax=8000
        )
        log_mel = librosa.power_to_db(mel + 1e-8, ref=np.max)
        return log_mel.T  # (T, 80)
    except Exception:
        return None


def _interpret_room(room_dict: dict) -> str:
    rt60 = room_dict.get('rt60', {})
    mid = np.mean([rt60.get(k, 0.3) for k in ['250Hz', '500Hz', '1000Hz', '2000Hz']])
    drr = room_dict.get('drr_db', 5.0)
    if mid < 0.2:
        size = "very small/dead room (studio treatment or small closet)"
    elif mid < 0.4:
        size = "small room with some absorption (bedroom, small practice space)"
    elif mid < 0.8:
        size = "medium room (living room, rehearsal space)"
    else:
        size = "large or reverberant space (hall, church, large room)"
    proximity = "close-miked" if drr > 8 else ("moderate distance" if drr > 2 else "distant")
    return f"{size}, {proximity} mic placement"


def _interpret_trachytes(v: float) -> str:
    if v < 0.1: return "extremely smooth/clean — no noise, pure tone"
    if v < 0.3: return "clean with subtle texture — good clarity"
    if v < 0.6: return "moderate texture — normal acoustic playing"
    return "high roughness — lots of overtones, noise, or distortion"


def _interpret_kallos(v: float) -> str:
    if v > 0.7: return "high harmonic richness — beautiful, complex tone"
    if v > 0.4: return "moderate harmonic content — pleasant"
    return "sparse harmonics — simple or muted tone"


def _interpret_cheimon(v: float) -> str:
    if v > 0.8: return "storm-level energy — intense, dynamic playing"
    if v > 0.5: return "moderate energy — engaged playing"
    return "calm energy — gentle or soft playing"


# ─── Audio I/O ───

def load_wav_as_float(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load WAV file as float32 at target sample rate."""
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def audio_chunk_generator_file(wav_path: str):
    """Yield 100ms audio chunks from a WAV file (for testing)."""
    audio = load_wav_as_float(wav_path)
    n_chunks = len(audio) // CHUNK_SAMPLES
    for i in range(n_chunks):
        yield audio[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
    # Yield remainder
    remainder = audio[n_chunks * CHUNK_SAMPLES:]
    if len(remainder) > 0:
        yield remainder


def play_audio_bytes(audio_bytes: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE):
    """Play PCM16 audio bytes through speakers."""
    try:
        import sounddevice as sd
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio_array, sample_rate, blocking=False)
    except Exception as e:
        print(f"[Audio playback error: {e}]")


# ─── Main Agent ───

async def run_agent(wav_file: str = None, device_id: int = None):
    """Main entry point. wav_file=None for live mic mode."""
    client = genai.Client(api_key=API_KEY)
    ears = EARSBridge()

    # ─── EARS tool declaration (explicit FunctionDeclaration for Live API) ───
    ears_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_ears_perception",
                description=(
                    "Get EARS 172-dimension acoustic perception data. Returns: "
                    "room acoustics (RT60 reverberation time, DRR direct-to-reverberant ratio) "
                    "and player tonal dimensions: tonos (tension/tone center), kallos (beauty/harmonic richness), "
                    "thymos (spirit/arousal), trachytes (roughness — low = clean), cheimon (storm energy/intensity). "
                    "Call this BEFORE giving any specific acoustic feedback."
                ),
                parameters=types.Schema(type="OBJECT", properties={})
            )
        ]
    )

    # ─── Session config ───
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[ears_tool],
        system_instruction=SYSTEM_PROMPT,
    )

    print(f"[Agent] Connecting to {MODEL}...")
    print("[Agent] EARS tool registered")
    print("[Agent] Type Ctrl+C to stop\n")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("[Agent] Connected. Session open.\n")

        if wav_file:
            await _run_file_mode(session, ears, wav_file)
        else:
            await _run_live_mode(session, ears, device_id=device_id)

    print("\n[Agent] Session closed.")


async def _run_receive_loop(session, ears):
    """Handle all responses from Gemini: tool calls, audio, transcription."""
    response_audio = []

    async for response in session.receive():
        if response.data:
            response_audio.append(response.data)
            play_audio_bytes(response.data)

        # Tool call handling
        if response.tool_call:
            for fc in response.tool_call.function_calls:
                print(f"\n[EARS tool called]")
                data = ears.analyze()
                data_str = json.dumps(data)
                # Print key dims
                player = data.get("player", {})
                print(f"  tonos={player.get('tonos', '?')} kallos={player.get('kallos', '?')} "
                      f"trachytes={player.get('trachytes', '?')} cheimon={player.get('cheimon', '?')}")
                await session.send_tool_response(
                    function_responses=[types.FunctionResponse(
                        id=fc.id, name=fc.name, response={"result": data_str}
                    )]
                )

        try:
            sc = response.server_content
            if sc:
                if sc.output_transcription and sc.output_transcription.text:
                    print(sc.output_transcription.text, end="", flush=True)
                if sc.turn_complete:
                    print()
                    break
        except Exception:
            pass

    return response_audio


async def _run_file_mode(session, ears, wav_file: str):
    """Send a WAV file to Gemini and get coach feedback."""
    import librosa, warnings
    warnings.filterwarnings("ignore")

    print(f"[File mode: {wav_file}]")
    audio, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    print(f"[Sending {len(audio)/SAMPLE_RATE:.1f}s of audio...]")

    # Feed audio to EARS buffer too
    ears.push_audio(audio)

    # Stream to Gemini in 100ms chunks
    chunk_size = CHUNK_SAMPLES
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        pcm = (chunk * 32767).astype(np.int16).tobytes()
        await session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
        )
        await asyncio.sleep(0.05)  # yield to event loop

    await session.send_realtime_input(audio_stream_end=True)

    # Ask Gemini to analyze
    await session.send_client_content(
        turns=types.Content(
            role="user",
            parts=[types.Part(text="Please use get_ears_perception and give me specific feedback on my playing and the room acoustics.")]
        ),
        turn_complete=True
    )

    print("[Gemini is listening...]\n")
    await _run_receive_loop(session, ears)


async def _run_live_mode(session, ears, device_id: int = None):
    """Live mic mode: continuous streaming + coaching."""
    import sounddevice as sd

    # Auto-select iPhone mic if available and no device specified
    if device_id is None:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if 'iphone' in d['name'].lower() and d['max_input_channels'] > 0:
                device_id = i
                break

    device_name = sd.query_devices(device_id)['name'] if device_id is not None else sd.query_devices(kind='input')['name']
    print(f"[Live mic mode — using: {device_name}]")
    print("[Play your instrument. Gemini will coach every 10 seconds. Ctrl+C to stop.]\n")

    mic_q = queue.Queue()

    def mic_cb(indata, frames, time_info, status):
        mic_q.put(indata[:, 0].copy().astype(np.float32))

    last_ask = time.time()
    ask_interval = 10.0

    async def send_loop():
        nonlocal last_ask
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=CHUNK_SAMPLES, device=device_id, callback=mic_cb):
            while True:
                try:
                    chunk = mic_q.get(timeout=0.5)
                    ears.push_audio(chunk)
                    pcm = (chunk * 32767).astype(np.int16).tobytes()
                    await session.send_realtime_input(
                        audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                    )

                    # Periodically ask for coaching
                    if time.time() - last_ask > ask_interval:
                        last_ask = time.time()
                        await session.send_client_content(
                            turns=types.Content(role="user", parts=[
                                types.Part(text="Call get_ears_perception and give me a quick 2-sentence coaching note.")
                            ]),
                            turn_complete=True
                        )
                except queue.Empty:
                    await asyncio.sleep(0.1)

    # Run send and receive concurrently
    sender = asyncio.create_task(send_loop())
    try:
        async for response in session.receive():
            if response.data:
                play_audio_bytes(response.data)
            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    print(f"\n[EARS tool called]")
                    data = ears.analyze()
                    await session.send_tool_response(
                        function_responses=[types.FunctionResponse(
                            id=fc.id, name=fc.name,
                            response={"result": json.dumps(data)}
                        )]
                    )
            try:
                sc = response.server_content
                if sc:
                    if sc.output_transcription and sc.output_transcription.text:
                        print(sc.output_transcription.text, end="", flush=True)
                    if sc.turn_complete:
                        print()
            except Exception:
                pass
    except KeyboardInterrupt:
        print("\n[Stopping...]")
    finally:
        sender.cancel()


def main():
    parser = argparse.ArgumentParser(description="EARS + Gemini Musician/Room Perception Agent")
    parser.add_argument("--file", "-f", help="WAV file to analyze (default: live mic)")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Audio input device ID (run with --list-devices to see options)")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("Input devices:")
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch, {d['default_samplerate']}Hz)")
        return

    asyncio.run(run_agent(wav_file=args.file, device_id=args.device))


if __name__ == "__main__":
    main()
