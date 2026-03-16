# Koda Vocal Health Coach

**Real-time vocal strain detection and coaching for singers, powered by Gemini Live.**

Koda listens while you sing, flags vocal strain as it happens, and gives you spoken technique cues between phrases. It runs in a browser — no install, no special hardware.

---

## The Problem

Self-taught singers don't get feedback on vocal health. A voice clinician with a laryngoscope can see strain. A singing teacher can hear it. But when you're practicing alone in your bedroom, nobody tells you that you've been pushing too hard for the last twenty minutes until your voice is hoarse the next morning.

Clinical tools exist (MDVP, Praat analysis suites) but they're designed for speech pathology labs. Consumer singing apps focus on pitch accuracy. Neither solves the problem of real-time strain feedback during practice.

## How It Works

Sing into your phone or laptop mic. Koda runs acoustic analysis 10 times per second and shows strain on a color-coded gauge:

- **Green** — healthy phonation
- **Yellow** — elevated effort, consider adjusting technique
- **Red** — significant strain detected, take a break

When you pause to breathe between phrases, Gemini Live speaks a short coaching cue — technique reminders like *"Ease up on the push"* or *"Drop your jaw, more space."* After you finish a song, Koda gives a brief spoken summary of how the session went.

---

## Architecture

See [`docs/architecture.png`](docs/architecture.png) for the visual diagram.

```
Browser (phone, laptop, tablet)
  │
  ├── getUserMedia() captures mic → Float32 PCM at 44.1kHz
  ├── 100ms audio chunks sent as WebSocket binary frames
  │
  ▼
FastAPI Backend (Cloud Run)
  │
  ├── Ring buffer — 1s sliding window (44,100 samples)
  ├── Parselmouth (Praat) — HNR, shimmer, pitch tracking (~78ms)
  ├── CPP — cepstral peak prominence, loudness-robust strain signal (~1ms)
  ├── EARS — mel spectrogram perceptual features (~9ms)
  ├── Wavelet scattering — 34-dim multi-scale modulation analysis (~3ms)
  │
  ├── Session-adaptive baseline — EMA per singer, calibrated from clean frames
  ├── Strain fusion — v8 (shimmer + CPP) primary, v11 (8 EARS signals) supporting
  │
  ▼
Gemini 2.5 Flash (Live API, persistent session)
  │
  ├── Phrase coaching — spoken cue at each breath point
  ├── Song praise — summary after 4s silence, interruptible
  └── Audio streamed back to browser via WebSocket
```

---

## Gemini Integration

Koda uses **Gemini 2.5 Flash** through the **Live API** with native audio output.

The Live API was the right fit here for a few specific reasons:

**Session persistence.** Gemini holds context across the entire singing session. If a singer has been mostly green and then hits two yellow phrases in a row, the coaching reflects that pattern. A stateless API call would treat every phrase in isolation.

**Native voice.** Gemini generates speech directly (voice: Kore) rather than routing text through a separate TTS service. This matters for latency — the singer hears the cue before starting their next phrase.

**Phrase-boundary timing.** The system sends coaching prompts at natural pause points (when the singer breathes), not mid-phrase. Each prompt includes the zone, strain score, phrase duration, and the last 5 phrase results, so Gemini has enough context to give a relevant tip.

**Coaching flow:**
```
Singer finishes phrase → silence detected → phrase metrics computed
  → Gemini receives structured prompt with zone, score, duration, history
  → Gemini speaks a short coaching cue (target: ≤8 words)
  → Audio sent to browser via WebSocket → singer hears it before next phrase
```

Coaching tone is anchored through `config/coaching_responses.yaml` — examples and style guides per zone that Gemini uses as reference, not scripts.

---

## Technical Approach

### Strain Detection

The central difficulty is separating loud singing from strained singing. Volume alone is not a useful signal — a singer belting cleanly is loud, and a singer with a constricted throat can be quiet.

We use two primary signals:

**Shimmer** measures amplitude variation between consecutive glottal cycles. When vocal fold closure is incomplete or irregular (rough/breathy phonation), shimmer spikes.

**CPP (Cepstral Peak Prominence)** measures how periodic the voice signal is. A healthy voice with good breath support produces a strong, clear fundamental — high CPP. Strain from constriction or pressing disrupts that periodicity, and CPP drops. Published research confirms CPP increases with loudness in healthy voices (Helou et al., 2020), which gives it natural immunity to volume-based false positives.

Both signals are measured as deviation from the singer's own session baseline, not absolute thresholds. The baseline adapts continuously via EMA from relaxed frames only — strained frames are excluded from the update, so the baseline can't drift toward the singer's strained voice.

**Fusion:** `strain = max(shimmer_dev, cpp_dev)` — either signal alone is enough to flag strain. Eight additional EARS features (validated via Cohen's d against ground-truth labels) and wavelet scattering z-scores provide supporting corroboration.

### What We Validated

Tested against labeled recordings from a single male voice with known strain regions:

| Clip | Expected | Predicted | Notes |
|------|----------|-----------|-------|
| Easy singing (anchor) | GREEN | GREEN | Neither signal elevated |
| Rough phonation (anchor) | RED | RED | Shimmer spike from forced rasp |
| Runnin' Down a Dream (full song) | GREEN | GREEN | Comfortable range, no push |
| Liza Jane (full song) | Verses=GREEN, Chorus=YELLOW/RED | Matches pattern | P80=0.44, verse/chorus separation visible |

Ground-truth test (28 labeled 2-second segments from Liza Jane): 43% exact zone match. Over-detection dominates under-detection — the system flags more than necessary rather than missing real strain.

**Honest limitations:**
- Moderate pressed phonation (tight but not yet rough) is not reliably detected. This is an open problem in voice science.
- Validated on one voice only. Female voices, trained singers, and non-English singing are untested.
- 43% exact accuracy is not clinical-grade. This is a practice tool, not a diagnostic.
- Browser mic quality varies a lot between devices and affects results.

### Real-Time Pipeline

| Stage | Time | Method |
|-------|------|--------|
| Audio capture | — | Browser getUserMedia, Float32 PCM |
| WebSocket transfer | ~5ms | Binary frames, 100ms chunks |
| EARS analysis | ~9ms | Mel spectrogram, emotion/touch/temporal |
| Parselmouth | ~78ms | HNR + shimmer via Praat |
| CPP computation | ~1ms | Custom cepstral analysis |
| Wavelet scattering | ~3ms | kymatio Scattering1D, 34-dim |
| **Total per frame** | **~96ms** | **Fits in the 100ms frame budget** |

---

## Running Locally

```bash
git clone https://github.com/RedFox485/koda-vocal-coach.git
cd koda-vocal-coach
python3 -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-server.txt

export GEMINI_API_KEY="your-key-here"
python src/vocal_health_backend.py --port 8080
```

Open `http://localhost:8080` in Chrome, grant mic access, and sing.

Requires Python 3.11+, a microphone, and a Gemini API key.

---

## Deploying to Google Cloud Run

```bash
export GEMINI_API_KEY="your-key-here"
./deploy-gcp.sh
```

The script enables GCP services, builds via Cloud Build, and deploys to Cloud Run with WebSocket support and session affinity. About 5 minutes end to end.

---

## Project Structure

```
src/
  vocal_health_backend.py    # FastAPI backend — analysis + WebSocket + Gemini coaching
  mel_extractor.py           # Mel spectrogram extraction (EARS)
  frequency_explorer.py      # Perceptual feature computation (EARS)
frontend/
  index.html                 # Single-file UI — strain gauge, pitch, range map, coaching panel
config/
  coaching_responses.yaml    # Coaching tone anchors and examples per zone
scripts/                     # Calibration, validation, and offline analysis tools
docs/
  architecture.png           # System architecture diagram
  TECHNICAL-INNOVATIONS.md   # Detailed research notes and validation data
Dockerfile                   # Cloud Run container (Python 3.11, CPU-only PyTorch)
deploy-gcp.sh                # One-command GCP deployment
fly.toml                     # Fly.io config (alternative deploy target)
```

---

## Built With

- [Gemini 2.5 Flash](https://ai.google.dev/) — Live API, native audio output
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — async WebSocket backend
- [Parselmouth](https://parselmouth.readthedocs.io/) — Praat acoustic analysis
- [kymatio](https://www.kymat.io/) — wavelet scattering
- [librosa](https://librosa.org/) — audio processing
- Google Cloud Run — deployment

---

## License

MIT
