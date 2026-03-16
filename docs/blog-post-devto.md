# Building a Real-Time Vocal Coach with Gemini Live and Signal Processing

*Built for the [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/) hackathon.*

Hundreds of millions of people sing. Almost none of them get professional feedback. A voice lesson runs $50-90/hour in the US, and in most of the developing world, trained vocal coaches simply don't exist. YouTube has filled part of the gap — one vocal coaching video has 240 million views — but video can't hear you sing. It can't tell you your throat is tightening before it becomes a nodule.

I built Koda to close that gap: a real-time vocal health coach that listens to you sing, detects strain as it happens, and speaks technique cues at every breath point — powered by a persistent Gemini Live session that improvises coaching based on what it actually hears.

This post covers how the signal processing pipeline works, why Gemini Live was the right integration point, and the key technical decisions that made a 96ms analysis loop possible.

## Architecture Overview

The system is straightforward in structure, complex in execution:

```
Browser (mic capture)
    → WebSocket (Float32 PCM, 10Hz)
        → Cloud Run backend (FastAPI)
            → Parallel analyzers (Parselmouth, CPP, perceptual engine, phonation classifier)
            → Strain fusion engine
            → Phrase boundary detection
                → Gemini Live (persistent session, native audio generation)
        ← JSON analysis events (ears_frame, phrase_end, coaching cues)
    ← Canvas rendering (strain gauge, pitch tuner, session graph)
```

The browser captures microphone audio using the Web Audio API and sends raw Float32 PCM frames over a WebSocket at 10Hz. The backend runs multiple acoustic analyzers in parallel on every frame, fuses their outputs into a single strain score, and streams results back as JSON events. When a phrase boundary is detected (singer pauses to breathe), the accumulated strain data is sent to a Gemini Live session that speaks a coaching cue in natural voice.

The entire analysis pipeline runs in ~96ms per frame. At 10Hz, that leaves comfortable headroom — the singer sees the strain gauge respond in real-time with no perceptible lag.

## Deep Dive: The Signal Processing Pipeline

The core challenge is distinguishing genuine vocal strain from normal loud singing. Volume alone is useless — a healthy belt at full volume should read green, while a quiet but tightly constricted phrase should read yellow. This required multiple independent signals, each capturing a different physical dimension of vocal fold behavior.

### Shimmer via Parselmouth

Shimmer measures the cycle-to-cycle variation in amplitude of vocal fold vibration. Healthy phonation produces consistent amplitudes; strained or fatigued vocal folds vibrate irregularly, spiking shimmer.

I use Parselmouth (the Python wrapper for Praat, the gold standard in phonetics research) to extract shimmer from a 200ms analysis window. The key insight is that shimmer must be measured against a session-adaptive baseline, not a fixed threshold. Every voice is different — a trained soprano and an untrained baritone have completely different "normal" shimmer values.

The baseline adapts using an exponential moving average that only accepts clean frames:

```python
BASELINE_EMA_ALPHA = 0.05    # ~10 frames to 40% adapted, ~50 to 92%
BASELINE_MAX_SCORE = 0.35    # only relaxed frames contribute

if is_clean_frame:
    a = BASELINE_EMA_ALPHA
    _session_shim_baseline = (1 - a) * _session_shim_baseline + a * shimmer_pct
```

This means the system works immediately from session start (seeded from a conservative baseline), adapts to each singer's natural voice within seconds of easy singing, and cannot be corrupted by strained frames — only clean frames pass the gate.

### Cepstral Peak Prominence (CPP)

CPP is the signal I'm most proud of in this pipeline. It measures how "periodic" the voice is by looking at the prominence of the fundamental frequency peak in the cepstral domain. Healthy phonation produces a strong, clear peak. Strain, constriction, and vocal fold irregularity flatten it.

The critical property: CPP is loudness-robust. A loud, healthy voice actually scores *higher* CPP because the harmonics are cleaner and more energetic. A loud, strained voice scores lower. This eliminates the core false-positive problem that plagues amplitude-based strain detection.

```python
def _compute_cpp(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    N = len(chunk)
    pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
    win = np.hanning(N)
    spec = np.fft.rfft(pre * win, n=N)
    log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
    cepstrum = np.real(np.fft.irfft(log_pow))[:N // 2]
    q_min = int(sr / 600)   # ~73 samples (max F0 = 600Hz)
    q_max = int(sr / 75)    # ~588 samples (min F0 = 75Hz)
    peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max + 1]))
    # Regression line across quefrency range
    qs = np.arange(q_min, q_max + 1) / float(sr)
    cs = cepstrum[q_min:q_max + 1]
    coeffs = np.polyfit(qs, cs, 1)
    regression_at_peak = np.polyval(coeffs, peak_idx / float(sr))
    return float(cepstrum[peak_idx] - regression_at_peak)
```

The implementation computes the real cepstrum (inverse FFT of the log power spectrum), isolates the quefrency range corresponding to human voice pitch (75-600Hz), and measures how far the dominant peak rises above the regression line. Pre-emphasis at 0.97 flattens the spectral tilt before analysis, which improves peak detection for lower voices.

CPP drops are smoothed with a 3-frame EMA to prevent phoneme-level transients (a single /i/ vowel can dip CPP by 0.3+) from triggering false positives, while sustained drops from real strain still register clearly.

### The Perceptual Strain Engine

Beyond Parselmouth and CPP, I run a perceptual analysis engine that extracts multiple dimensions from each audio frame: energy envelope decay patterns, harmonic spacing irregularity, fast amplitude modulation in the 75-300Hz band (think of it as audio-domain motion magnification — amplifying micro-fluctuations riding on the fundamental), spectral tilt via alpha ratio, and 4-8Hz effort-band modulation depth linked to laryngeal muscle micro-fluctuations.

Each dimension was validated against ground-truth annotations using Cohen's d effect sizes on labeled singing data. The strongest signal — energy envelope decay oscillation — achieved d=-0.999, meaning it separates strained from relaxed singing by a full standard deviation.

### Strain Fusion

The final strain score fuses all signals using a max-blend strategy:

```python
# Primary signals: shimmer spike + CPP drop
shim_dev = max(0.0, shimmer - _session_shim_baseline) / 7.0
cpp_dev  = max(0.0, _session_cpp_baseline - cpp) / 0.35
v8_strain = min(1.0, max(shim_dev, cpp_dev))

# Perceptual engine (capped at 0.7 — supporting role)
ears_v11 = min(1.0, max(elast_dev, anham_dev, am_dev, ...)) * 0.7

# Final fusion: any signal alone can trigger strain
strain = min(1.0, max(v8_strain, phonation_score * 0.7, ears_v11))
```

The design principle is that any single analyzer detecting strain is enough to raise the score. This is deliberate — different types of strain manifest in different signals. Pressed phonation with tight vocal fold closure might not spike shimmer at all (the folds are vibrating regularly, just too hard), but the perceptual engine catches the spectral tilt change and the CPP detects the constriction. Max-blend ensures nothing slips through.

## The Gemini Live Integration

This is where the project shifts from signal processing to something genuinely new. The analyzers produce numbers. Gemini turns those numbers into coaching.

### Why Gemini Live Specifically

Three properties of Gemini Live made this possible in a way no other API could:

1. **Persistent session.** A single Gemini Live connection stays open for the entire singing session. Every phrase, every strain spike, every zone transition stays in context. When Gemini delivers a coaching cue after the fifth phrase, it knows what happened in phrases one through four. No re-prompting, no context assembly, no stateless request/response.

2. **Native audio generation.** Gemini Live generates voice natively — it's not text piped through a TTS engine. The coaching cues sound natural, with appropriate pacing and emphasis. When Gemini says "ease off the push," it sounds like a coach, not a robot reading a transcript.

3. **Streaming input/output.** The connection accepts audio input and produces audio output in a continuous stream. This is critical for the breath-point timing model — Gemini needs to perceive the pause and respond within the natural breath window (~1-2 seconds) before the singer starts the next phrase.

### How Coaching Cues Work

The system detects phrase boundaries using silence duration and RMS thresholds. When a singer pauses (MIN_SILENCE_S = 0.4 seconds of sub-threshold RMS), the phrase ends and the backend assembles a coaching payload: strain score, phrase duration, vocal zone (green/yellow/red), shimmer and CPP deviations, and the perceptual engine's dimensional breakdown.

This payload is sent to the Gemini Live session along with a style directive: technique-anchored, concise, specific to what just happened. Gemini reads the strain data, considers the last several phrases of context, and speaks a coaching cue.

The cues are not scripted. Two consecutive yellow-zone phrases might produce "Drop your jaw — find the note with less effort" followed by "Better placement that time, but I'm still hearing tension on the vowel." Gemini improvises within the technique framework, adapting to what it observes changing across phrases.

After a sustained silence (indicating the song is over), Gemini delivers a full session summary referencing specific moments — which phrases showed strain, where the singer corrected, what to work on next time.

## Key Technical Decisions

**Why FFmpeg for audio framing:** The browser sends raw PCM, but the backend needs consistent frame boundaries aligned to the analysis window. FFmpeg handles resampling and frame alignment with sub-millisecond precision, which matters when you're computing cycle-to-cycle shimmer on 100ms windows.

**Why Cloud Run:** The backend is stateful per WebSocket connection but stateless across sessions. Cloud Run's container model fits perfectly — each connection gets its own instance with its own session-adaptive baselines. In production benchmarking (50 sequential requests), Koda averaged 119ms response time with zero failures and zero cold starts — even after 5 minutes of idle, the first request returned in 111ms. Cloud Run's min-instances=1 keeps the container warm so judges and users never wait.

**Why WebSocket over WebRTC:** WebRTC is designed for peer-to-peer audio/video with its own codec pipeline. I don't need codec negotiation or NAT traversal — I need raw PCM frames delivered reliably in order. WebSocket gives me exactly that with simpler debugging. The tradeoff is slightly higher latency (~10-20ms more than WebRTC's data channel), but at 96ms total pipeline latency, that's irrelevant.

**Why adaptive baselines instead of fixed thresholds:** This was the single most important design decision. Fixed thresholds for shimmer and CPP would make the system usable only for voices similar to whoever calibrated it. An untrained singer with naturally higher shimmer would show perpetual yellow. A trained singer with exceptionally clean phonation would never trigger a warning even when straining. Adaptive baselines solve both cases — the system learns what "normal" means for each individual voice within seconds.

## Results

Koda is deployed on Cloud Run and working end-to-end. The full loop — sing into the browser, see the strain gauge respond, hear Gemini speak a coaching cue at the breath point — runs in real-time with no perceptible delay.

The strain detection reliably distinguishes comfortable singing (green) from pushed technique (yellow/red) across different voice types and singing styles. The adaptive baselines mean a first-time user gets meaningful feedback within the first few phrases, with accuracy improving as the session continues.

Gemini's coaching cues are contextual and specific — not generic "sing better" advice, but targeted observations about what changed in the last phrase and what to try differently.

## The Access Angle

46% of singers report voice disorders. 58% of vocal injuries are nodules — mostly preventable with proper technique. A single vocal coaching video on YouTube has 240 million views, because hundreds of millions of people want to learn but can't access a teacher.

Koda doesn't replace vocal coaches. It reaches the people who will never have one. A teenager in Lagos singing along to Afrobeats on her phone. A choir member in rural Iowa with no voice teacher within 100 miles. A bedroom singer who was told they "can't sing" twenty years ago and hasn't tried since.

Every person has a voice. Not everyone has a teacher. Gemini Live made it possible to change that.

---

## Built With

- **Python / FastAPI** — WebSocket backend, async event loop
- **Gemini Live API** — persistent coaching session, native audio generation
- **Parselmouth / Praat** — shimmer, HNR, jitter extraction (phonetics research standard)
- **Cepstral Peak Prominence (CPP)** — custom implementation for vocal fold closure quality
- **Cloud Run** — stateful-per-connection container hosting, scales to zero
- **WebSocket** — 10Hz bidirectional audio/analysis stream
- **Canvas API** — real-time strain gauge, pitch tuner, session graph rendering
- **librosa / NumPy / SciPy** — spectral analysis, filtering, signal processing primitives
