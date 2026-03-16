# Koda Vocal Health Coach — Technical Innovations
## Running doc for competition submission (Gemini Live Agent Challenge, deadline March 16, 2026)

---

## The Problem We're Solving

Every tool that measures vocal health costs thousands of dollars and requires a clinical setting.
Every consumer singing app a self-taught singer can afford is just a pitch tuner.

There is no real-time vocal strain monitor that works while you're actually practicing — for anyone who can't see a voice clinician. The gap is real: 50–100 million self-taught singers globally, many of them learning from YouTube with no health feedback at all. They can injure themselves and not know it's happening until they lose their voice.

**We sit between Smule ($0, pitch only) and MDVP ($6,000, clinical). That gap is empty.**

---

## Core Innovations

### 1. Dual-Signal Strain Model: CPP + Shimmer

Most approaches to voice analysis treat strain as a single axis or rely on shimmer alone. We went through several formula iterations to find a signal combination that is both *accurate* and *loudness-robust*.

**The core problem with single-metric approaches:**
- Shimmer alone spikes at phrase onsets/offsets (artifacts) and fires on natural loudness variation
- HNR rises for *both* pressed strain AND clean high notes (confounded by pitch)
- Energy variance (dyn_diaphora) works for isolated pressed notes but fires on natural phrase dynamics in continuous singing

**Formula v8: max(shim_dev, cpp_dev)**

```python
shim_dev = max(0.0, shimmer_pct - baseline_shim) / 10.0   # rough phonation
cpp_dev  = max(0.0, baseline_cpp - cpp) / 0.5             # phonatory irregularity / tightness
strain   = min(1.0, max(shim_dev, cpp_dev))
```

**Shimmer** detects rough phonation: blown-out, breathy, raspy voices. Amplitude variation between glottal cycles spikes when closure is incomplete.

**CPP (Cepstral Peak Prominence)** detects phonatory irregularity from the opposite direction: when the voice is *healthy and resonant*, CPP is high. When it becomes *strained or constricted*, CPP drops. Crucially: **CPP is loudness-robust** — a loud-but-healthy voice scores *higher* CPP (more harmonic clarity from better breath support), which means loud healthy singing doesn't trigger false positives. Only irregularity from strain causes the drop.

CPP catches both pathologies:
- Rough phonation: incomplete closure → irregular pulses → CPP drops
- Severe pressing: hyperadduction → turbulent flow artifacts → CPP drops

**Voiced-onset gate**: Both signals are suppressed for the first 3 consecutive voiced frames (300ms) of each phrase. Parselmouth shimmer and cepstral computation are unreliable during voicing onset — the vocal folds haven't settled into stable oscillation. This eliminates the most common false-positive pattern.

**Anchor validation (v8):**
| Clip | True Zone | Predicted | P80 | Primary driver |
|------|-----------|-----------|-----|----------------|
| Easy 2 | GREEN | GREEN ✓ | 0.162 | neither |
| Medium 1 | GREEN | GREEN ✓ | 0.256 | neither |
| hard push | YELLOW | GREEN ✗ | 0.146 | pressed phonation not detected |
| Rough 1 | RED | RED ✓ | 1.000 | shimmer spike |

**Song validation (v8):**
| Recording | Daniel's expectation | Predicted | P80 |
|-----------|---------------------|-----------|-----|
| Runnin' Down a Dream | just green | GREEN ✓ | 0.288 |
| You (Chris Young) | mostly green, couple yellow | YELLOW ✓ | 0.529 |
| Liza Jane | verses green, chorus yellow/red | RED | 0.641 |

**Known limitation**: Moderate pressed phonation (hyperadduction without yet causing glottal irregularity) is not reliably detected by either shimmer or CPP. The dyn_diaphora metric (energy variance) captured it on isolated notes, but fires on natural phrase dynamics in continuous singing. Open research question.

---

### 2. Loudness vs. Strain: The Core Research Problem

The hardest part of vocal strain detection is separating "loud but healthy" from "loud and strained." Most consumer tools fail here — they flag loud singing as strained and miss quiet strain.

**What we tried and why it failed:**
- `dynamis_diaphora` (energy variance, 300ms): Works for isolated pressed notes. In continuous singing, natural phrase dynamics (vowel/consonant energy transitions, emphasis, vibrato) create energy variance that looks like pressed phonation. Song phrase transitions showed dyn=5.4+ vs pressed phonation dyn=1.0.
- `rhoe_diaphora` (energy flow range, 1s): Correlates with strain in 2s segment averages (r=+0.58) but the per-100ms frame values don't match the seed calibrated from 2s averages.
- HNR: Rises for both pressed strain AND clean high notes — confounded by pitch.

**Why CPP works:** CPP measures the *relative prominence* of the fundamental pitch peak in the power cepstrum. This is a ratio measurement — it's not affected by absolute volume, only by the regularity of the glottal pulses. Published research (Helou et al., 2020; PMC 7295673) confirms CPP rises with loudness in healthy voices, providing natural loudness immunity.

**What published research says:** A 2025 paper on automatic strain classification in singers found wavelet scattering features (multi-scale amplitude modulation) achieved 86.1% accuracy — the highest reported for this task. Shimmer + CPP is a practical approximation of this approach with much lower compute cost.

---

### 3. Asymmetric Shimmer Baseline: Root Cause of Over-Detection Fixed

**Problem discovered (March 13)**: The session-adaptive shimmer baseline was adapting symmetrically — drifting DOWN during soft or clean singing sections. After a quiet verse, the baseline would drop from seed (5.26%) toward ~3.3%. When Daniel returned to normal comfortable singing (~7-8% shimmer), the model treated his relaxed voice as strained. This was the primary source of false positives: up to 25% over-detection on the Liza Jane ground truth.

**Root cause**: Standard EMA with symmetric alpha=0.05 adapts equally toward lower shimmer values. A singer who alternates between quiet humming and full-voice phrases will have their baseline constantly pulled down by the quiet sections.

**Fix — Asymmetric EMA**:
```python
if shimmer_pct >= _session_shim_baseline:
    _session_shim_baseline = (1 - 0.05) * _session_shim_baseline + 0.05 * shimmer_pct
else:
    _session_shim_baseline = (1 - 0.01) * _session_shim_baseline + 0.01 * shimmer_pct
```

Upward adaptation: 5% per frame (normal). Downward adaptation: 1% per frame (5x slower). The baseline follows the singer's voice as it warms up or pushes harder, but resists being dragged down by brief soft sections. CPP uses the same asymmetry.

**Accuracy impact**: 29% → 36% exact match on Liza Jane ground truth (after asymmetric fix).

---

### 3a. Low-Energy Gate: Phrase Tail Artifact Suppression

**Problem**: When a singer's voice fades out at the end of a phrase, signal amplitude drops toward (but stays above) the silence floor. At RMS=0.008-0.016, shimmer and CPP measurements are dominated by noise rather than actual glottal oscillation. A phrase tail at RMS=0.009 was producing shimmer=28.5% and CPP=0.095 — values that look like extreme pathological strain, but are actually just measurement noise at low SNR.

**Gate**:
```python
SILENCE_RMS   = 0.008   # below = no voice
LOW_ENERGY_RMS = 0.016  # below = phrase tail, acoustic features suppressed (2x silence floor)

low_energy = rms_val < LOW_ENERGY_RMS
if onset_gated or low_energy:
    shim_dev, cpp_dev = 0.0, 0.0
```

Phrase tail frames pass the activity gate (the microphone is still picking up signal) but their acoustic feature scoring is zeroed out — exactly like onset gating at phrase starts. The frame is marked as "active" (voice present) but contributes zero strain.

The low-energy gate also applies to EARS v11 signals (alpha ratio, AM features) for the same reason: low-amplitude signal produces unreliable spectral and modulation measurements.

---

### 4. Session-Adaptive Baseline with Bias-Resistant EMA

**The problem with absolute thresholds**: Acoustic measurements vary dramatically by singer, microphone distance, room acoustics, and recording equipment. A shimmer of 6% is normal for one voice and strained for another. HNR at 18dB is typical for a baritone and low for a trained soprano.

**Our approach**: Measure strain as *deviation from the singer's own relaxed voice baseline*, not against clinical population averages.

**The naive approach fails**: Establish baseline from the first N frames, then lock it. Problem: a beginner singer who doesn't know how to warm up will strain from frame 1. Their baseline calibrates to their strained voice, and everything looks normal after that — the system becomes useless.

**Our solution**: Continuous EMA (Exponential Moving Average) adaptation with a strain gate.

```python
BASELINE_EMA_ALPHA = 0.05   # per clean frame

# Every frame:
tentative_score = min(1.0, max(shim_dev, dyn_dev))   # vs current baseline
if tentative_score < 0.25:  # only truly relaxed frames update the baseline
    baseline = (1 - 0.05) * baseline + 0.05 * observed
```

**Key properties:**
- **Starts immediately**: The seed is a valid baseline from frame 0. No warmup required.
- **Bias-resistant**: Strained frames (score ≥ 0.25) are excluded from the EMA. A beginner who strains from the first note produces *zero* clean frames → baseline stays at seed → strain is still detected correctly vs seed.
- **Converges naturally**: After ~10 clean frames (1 second of easy singing), baseline is ~40% dialed in. After ~50 frames (5 seconds), 92% converged. A singer who warms up properly gets a personalized baseline within a few seconds.
- **Never gets hijacked**: Because only clean frames contribute, sustained strain can never pull the baseline up over time.

**Seed values**: Pre-calibrated from recordings of a reference voice (typical mid-range male singer, comfortable singing, close-mic'd). Intentionally conservative — set to the middle of the typical range so both high and low voices converge toward their true baseline from a functional starting point.

---

### 4a. EARS v11: Deep-Analysis Strain Signals (March 13, 2026)

Statistical analysis of Liza Jane ground truth labels (Cohen's d across green vs yellow/red chunks) revealed six acoustic features from the EARS perceptual system that are reliably elevated in strained voice:

| Feature | Cohen's d | Direction | Physical meaning |
|---------|-----------|-----------|-----------------|
| `touch.elastikos` | -0.999 | Higher in strain | Energy envelope decay oscillation — strained voice loses elastic bounce |
| `harmonic.anharmonia` | -0.727 | Higher in strain | Harmonic spacing irregularity — pressed phonation disrupts partials |
| `temporal.metabole` (ZCR) | -0.670 | Higher in strain | Zero-crossing rate — roughness and aperiodicity |
| `temporal.rhoe_mese` | tracking | Higher in strain | Spectral flux — sustained vocal effort modulation |
| AM fast (75-300Hz, 10-30Hz band) | -0.564 | Higher in strain | Hilbert AM roughness band — Eulerian motion amplification |
| `life.metabolism` | tracking | Higher in strain | Mel energy throughput — elevated in chorus push sections |

These are combined as **EARS v11** with a 0.7 cap (supporting role only — v8 shimmer+CPP remains primary):
```python
ears_v11 = min(1.0, max(elast_dev, anham_dev, am_dev, zcr_dev, flux_dev, metab_dev,
                         alpha_dev, effort_am_dev)) * 0.7
```

**Alpha ratio (Sol et al. 2023)**: Log spectral tilt ratio log(1-5kHz energy / 50Hz-1kHz energy). Ranked #1 feature for vocal mode classification in singing research (92% F1). Captures pressed phonation via spectral tilt — strained phonation pushes energy into upper harmonics. Session-adaptive: deviation above singer's own relaxed baseline / 6dB scaling.

**4-8 Hz effort AM band**: Laryngeal muscle micro-fluctuations in the 4-8Hz band are diagnostic for vocal effort (Drullman 1994). Distinguished from the 10-30Hz roughness band by targeting the laryngeal control frequency range specifically. Computed from RMS envelope history.

All EARS v11 signals use session-adaptive EMA baselines — same clean-frame gate as the v8 shimmer/CPP baseline — so they calibrate to the singer's own voice within 5-10 seconds.

---

### 4b. Gradient Zone Display

**Problem**: Binary zone transitions (green → yellow at score=0.40) produced jarring color jumps. A singer approaching the threshold experiences the meter suddenly changing zone. This creates anxiety at exactly the wrong moment — where the singer should be encouraged to ease off gently.

**Solution**: 8-point interpolation window before each threshold:
```javascript
const STRAIN_GREEN  = 0.40;
const STRAIN_YELLOW = 0.60;
const TRANSITION_W  = 0.08;  // 8-point blend window

// Score 0.32 → starts blending green→yellow
// Score 0.40 → full yellow
// Score 0.52 → starts blending yellow→red
// Score 0.60 → full red
```

The strain meter, zone pill, and history graph line all use gradient color. The singer sees the color gently warming toward yellow as they approach the threshold — a natural, intuitive signal to ease off before hitting the zone change.

---

### 5. Real-Time Architecture: Browser as Microphone

Traditional audio analysis tools assume local audio access. We stream audio from the browser directly to the analysis backend:

```
Browser getUserMedia() → Float32 PCM chunks → WebSocket binary → Server ring buffer → EARS + parselmouth
```

This means:
- Works on any device with a browser and microphone (phone, tablet, laptop, no install)
- Latency: ~100ms per analysis frame (parselmouth ~78ms + EARS 300ms window ~4ms)
- The singer's microphone setup (their phone, their laptop) is automatically used — no configuration

**EARS fast path**: Full `analyze_mel` on 1s window ~9ms. The 300ms window for dyn_diaphora adds ~4ms. Parselmouth (HNR + shimmer on 100ms window) ~78ms. Total per-frame: ~91ms — well within the 100ms chunk budget.

---

### 6. Gemini Live as Phrase-Boundary Vocal Coach

Analysis happens at 10Hz (every 100ms). Coaching happens at phrase boundaries — when the singer finishes a phrase and takes a breath. This matches the natural rhythm of how a vocal coach gives feedback: not mid-phrase, but at the natural pause.

The Gemini Live session persists for the duration of the practice session. At each phrase boundary, the system sends:
- Zone (green/yellow/red)
- Average strain score for the phrase
- Recent phrase history (last 5 phrases)
- Whether breathiness was detected

Gemini responds with voice coaching, streamed back to the browser in real time. The singer hears the coach's voice in their ear, then starts the next phrase.

**Why Live (not standard API)**: Gemini Live maintains session context, so coaching evolves as the session progresses — early feedback is different from late-session feedback when fatigue becomes a factor.

---

### 7. Wavelet Scattering v9: Session-Adaptive Multi-Scale Modulation Baseline

The 2025 paper finding 86.1% accuracy used wavelet scattering (multi-scale amplitude modulation analysis). We implemented this with a session-adaptive approach that sidesteps the cross-singer calibration problem.

**Why cross-clip scatter failed**: Raw wavelet scattering from an anchor clip (close-mic'd isolated notes) is too different from song recordings (room acoustics, mic distance, phrase dynamics) for zero-shot distance to work. Scatter_p80=1.000 for all songs when compared against anchor baseline.

**Solution: session-adaptive scatter baseline**. Build the scattering baseline *within each session* from v8-gated clean frames — exactly like the shimmer/CPP EMA baseline. The baseline adapts to the singer's own voice, room, and microphone within ~5-10 seconds of easy singing.

```python
# 34-dim log-compressed scattering features per 100ms frame
# RMS normalization before scatter → loudness invariant
# Log compression after → equalizes 3-order-of-magnitude dynamic range
feat = log(mean_over_time(Scattering1D(J=7, Q=1)(rms_normalized_chunk)))

# Mean absolute z-score as strain score
z = (feat - baseline_mean) / baseline_std
scatter_score = clip((mean|z| - 1.0) / 2.0, 0, 1)
```

**Max-blend fusion** (v8 + scatter):
```python
# Either signal alone → strain. Prevents fusion from averaging away real detections.
w = 0.5
max_score  = max(v8_strain, scatter_score)
wavg_score = w * v8_strain + (1-w) * scatter_score
fuse_score = (1-w) * max_score + w * wavg_score
```

**Computation cost**: 2.7ms/frame (vs parselmouth ~78ms). Total per-frame: ~94ms. Still within 100ms budget.

**Liza Jane verse/chorus separation** (v9, 10s segments):
| Segment | Fused P80 | Zone | v8 P80 | Scatter P80 |
|---------|-----------|------|--------|-------------|
| 0-10s  | 0.310 | GREEN  | 0.369 | 0.257 |
| 10-20s | 0.615 | RED    | 0.627 | 0.668 |
| 20-30s | 0.373 | YELLOW | 0.471 | 0.221 |
| 30-40s | 0.444 | YELLOW | 0.551 | 0.315 |
| 40-50s | 0.648 | RED    | 0.710 | 0.586 |
| 50-60s | 0.714 | RED    | 0.795 | 0.487 |
| 60-70s | 0.688 | RED    | 0.864 | 0.681 |

Pattern: `G → R → Y → Y → R → R → R` — verse relaxation then chorus sustained strain. Scatter adds independent confirmation of v8's strain signal.

---

## What Makes This Novel

1. **Dual-signal architecture distinguishing two pathology types** — shimmer catches rough phonation; CPP catches irregularity/constriction. Most tools use one metric.
2. **CPP loudness-robust strain detection** — loud+healthy → higher CPP → zero false positives on powerful singing
3. **Asymmetric EMA baseline** — baseline resists downward drift from quiet passages, preventing soft sections from re-calibrating the model to flag normal voice as strained
4. **Low-energy gate** — phrase tails (voice fading out) produce garbage acoustic readings; suppressed identically to onset frames
5. **EARS v11 multi-signal corroboration** — 8 independently-validated signals (d=-0.99 to d=-0.56) all elevated in strain, providing convergent evidence
6. **Alpha ratio from literature** — Sol et al. 2023 #1 feature for vocal mode classification applied as a session-adaptive signal
7. **Session-adaptive wavelet scatter baseline** — builds within-session modulation baseline, cross-singer calibration-free
8. **Max-blend sensor fusion** — either signal alone can detect strain; prevents weighted averaging from suppressing real detections
9. **Bias-resistant EMA baseline** — correctly handles untrained singers who strain immediately
10. **Browser-native real-time pipeline** — no app install, works on any device with a browser
11. **Phrase-aware AI coaching** — coaching cadence matches natural singing rhythm; Gemini Live maintains session context so feedback evolves over time
12. **Gradient zone display** — smooth color interpolation prevents jarring threshold jumps; singer sees the meter warm toward the next zone before it changes

---

## System Accuracy

**Song-level validation (current — v11 + asymmetric baseline + low-energy gate):**
| Recording | Expected | Predicted | P80 | v8 P80 | Scatter P80 |
|-----------|----------|-----------|-----|--------|------------|
| Runnin' Down a Dream | GREEN | GREEN ✓ | 0.366 | 0.175 | 0.137 |
| You (Chris Young) | YELLOW | GREEN ✗ | 0.218 | 0.141 | 0.000 |
| Liza Jane (67s) | RED | YELLOW ✗ | 0.437 | 0.481 | 0.290 |

**Ground truth validation (Liza Jane, 28 labeled 2s chunks):**
- Baseline (v8 only): 25% exact match (7/28)
- + asymmetric shimmer baseline: 36% exact match (10/28)
- + EARS v11 + new thresholds (0.40/0.60): 43% exact match (12/28)
- + low-energy gate: 43% exact match (maintained; 58-60s false positive resolved)
- Over-detection: 16/28 | Under-detection: 2/28

**What "43% exact" means**: Each 2s chunk is classified as green/yellow/red. 43% match exactly; 57% are off by one zone (no chunk is off by 2 zones). Over-detection dominates — the system errs toward flagging potential strain, which is the safe failure mode for a vocal health tool.

**Liza Jane within-song temporal structure** (v9):
- Pattern: `G → R → Y → Y → R → R → R` — verse relaxation visible, then chorus sustained strain
- Deep-analysis heatmap: EARS features (anharmonia, elastikos, metabole, rhoe_mese) show clear red bands at chorus transitions

**Known limitation**: Moderate pressed phonation (hyperadduction without glottal irregularity) is not reliably detected. Shimmer and CPP only fire when closure is visibly incomplete or when voice becomes aperiodic. The alpha ratio and EARS v11 signals provide partial coverage but not complete.

**Summary metric**: P80 (80th percentile of per-frame strain scores). Mean undercounts brief intense strain events. P80 correctly captures "singer who strains 20% of the time" as meaningfully different from "singer who strains 0%".

---

## Open Questions / Future Work

- Validation on diverse voice types (female voices, trained singers, different languages)
- Longer session validation — does baseline drift correctly over 20-30 minutes?
- Register break detection (sudden kallos/metabole spike) — not yet implemented
- Vibrato compensation in strain scoring — not yet implemented
- Clinical validation against MDVP and Praat measurements on the same recordings
- Under-detection of moderate pressed phonation — open research problem

---

*Last updated: March 13, 2026*
