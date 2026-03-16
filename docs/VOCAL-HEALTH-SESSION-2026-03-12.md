# Vocal Health Coach — Session State March 12, 2026 (~1:37pm)

## Decision: PIVOT — Entry 1 is now Vocal Health Coach

**Original Entry 1**: Musician/Room Perception (still works, still being built)
**New Entry 1 (PRIMARY)**: Real-time vocal strain detector / vocal health coach

### One-line pitch
"Every tool that measures vocal health costs thousands and requires a clinician.
Every app a singer can afford is just a pitch tuner.
We're the first real-time vocal health monitor anyone can use while they're actually practicing."

### The kids angle
- Self-taught singers learning on YouTube have NO health feedback
- Kids who can't afford lessons → injury risk with no warning
- No existing consumer tool fills this gap — confirmed by competitive analysis
- Urgent need: singers physically injure themselves. Musicians don't (same way).

---

## Strain Detection Theory — VALIDATED March 12

### What happens acoustically when a voice strains:
1. Laryngeal tension → tonos↑, thymos↓, kallos drops (harmonics thin out)
2. Pressed phonation → trachytes↑↑, kallos↓↓, jitter↑ (noise floor rises)
3. Register break → metabole spike, sudden kallos drop

### Test: Daniel's three recordings (Macedonia St 27/29/30)
- 27: comfortable mid ahh (baseline)
- 29: pushed high chest voice (strained)
- 30: head voice same high note (healthy)

### Results:
```
                   27 (easy)   29 (strained)   30 (head voice)
tonos (tension)    0.39        0.78 ↑↑          0.46
thymos (spirit)    0.70        0.30 ↓↓          0.66
cheimon (energy)   1.35        0.39 ↓↓          1.27
kallos (beauty)    0.69        0.70 →           0.72
katharotes_db      124         94 ↓             100
```

### Validated strain formula:
```
strain_score = (tonos × 0.6) + ((1 - thymos) × 0.4)

27 (easy):     0.35  → GREEN ✅
29 (strained): 0.75  → RED ✅
30 (head):     0.41  → GREEN ✅
```

### Key finding: trachytes = 0.0 across all three
- trachytes doesn't capture vocal roughness at EARS sensitivity levels
- Need to add HNR (harmonics-to-noise ratio) + jitter from librosa
- This is the next build task

---

## Competitive Analysis — KEY FINDINGS

**No consumer app provides real-time vocal strain monitoring during live singing. Zero.**

| Tier | Product | What they do |
|------|---------|-------------|
| Consumer | Smule, Yousician, Vanido, Sing Sharp | Pitch accuracy only |
| Closest competitor | OperaVox (iPhone) | Jitter/shimmer, POST-RECORDING only |
| Clinical (real-time) | Visi-Pitch | Real-time but $2.5-4K, 1990s design |
| Clinical (post) | MDVP, Praat, CSL | $4-8K, clinical use only |
| Wearable | Northwestern patch (2025) | Vocal LOAD (usage amount), not quality |

**We sit between Smule ($0, pitch only) and MDVP ($6,000, clinical). That gap is empty.**

### Market size
- 50-100M self-taught singers globally
- 10% willing to pay = 5-10M users
- $5-10/month = $300M-1.2B annual addressable market

---

## Product Architecture (decided)

### Features (priority order):
| Priority | Feature | Status |
|----------|---------|--------|
| CORE | Real-time strain meter (green/yellow/red) | Building next |
| CORE | HNR + jitter for roughness dimension | Building next |
| CORE | Pitch needle (vibrato/slide compensated) | Queued |
| CORE | Gemini coaching at phrase boundaries | Working demo exists |
| CORE | Session strain graph (timeline at bottom) | Queued |
| IMPORTANT | Phrase detection (structural, not just silence) | Queued |
| IMPORTANT | Timing feedback (rushing/dragging) | Queued |
| LATER | Beautiful UI | Last 4 hours |
| LATER | Key/scale pitch mode | Post-competition |

### Architecture principles:
- Backend/frontend completely separated by WebSocket contract
- Any UI can plug in — prototype HTML now, beautiful UI later
- Gemini coaching happens at phrase boundaries (not mid-phrase)
- Three coaching modes: voice+text (competition default), voice-only, text-only

### WebSocket event contract:
```json
{"type": "ears_frame", "strain_score": 0.75, "tonos": 0.78, "thymos": 0.30,
 "trachytes": 0.0, "kallos": 0.70, "hnr_db": 12.3, "jitter_pct": 2.1,
 "pitch_hz": 523.2, "pitch_note": "C5", "pitch_cents_off": -8,
 "vibrato_detected": true, "timestamp": 1234567890.0}

{"type": "gemini_coaching", "transcript": "...", "audio_playing": true}

{"type": "session_update", "strain_history": [0.35, 0.38, 0.75, 0.72, 0.40]}
```

### Tech stack:
- Backend: Python FastAPI + WebSocket
- Audio: sounddevice (iPhone mic, device [4])
- EARS: streaming_pipeline + frequency_explorer
- Vocal metrics: librosa (HNR, jitter, pyin pitch)
- Gemini: Live API (gemini-2.5-flash-native-audio-latest) — CONFIRMED WORKING
- Frontend: minimal HTML/JS prototype now, beautiful later

---

## Gemini Live API — CONFIRMED WORKING
- Model: gemini-2.5-flash-native-audio-latest (only model with bidiGenerateContent)
- EARS as explicit FunctionDeclaration tool — tool call confirmed working
- Response: 29s of voice coaching with specific EARS dimension values used
- Key: needs `await asyncio.sleep(0.05)` in send loop to yield event loop
- Audio output at 24kHz PCM16 mono

---

## Build Progress (March 12, afternoon session)

### COMPLETED:
1. [x] HNR + shimmer via parselmouth (validated on Daniel's 3 recordings)
2. [x] Real-time WebSocket backend (`src/vocal_health_backend.py`)
   - Browser streams PCM audio → server analyzes → streams results back
   - Activity gate: parselmouth + EARS only run when voice detected (fixes idle-yellow bug)
   - 12.6x EARS speedup: fast path (emotion+touch only) = 1ms vs 9ms full analyze_mel
   - Full analyze_mel only when debug clients connected
3. [x] Pitch detector with vibrato/slide compensation (librosa pyin)
4. [x] Phrase boundary detector (energy-based, fires phrase_end coaching hook)
5. [x] Singer UI (`frontend/index.html`) — strain meter + pitch needle + session graph
6. [x] Koda debug panel (`frontend/debug.html`) — all EARS dims, signal chain, event log, phrase history
7. [x] Fly.io deployment — **LIVE at https://koda-vocal-coach.fly.dev/**
   - 2 dedicated perf CPUs, 4GB RAM, Dallas region
   - Auto-stop when idle (free), auto-wake on connection (~2s)
   - `https://koda-vocal-coach.fly.dev/debug` = Koda panel
   - `https://koda-vocal-coach.fly.dev/health` = status check

### Key architectural decisions:
- Browser getUserMedia → PCM Float32 → WebSocket binary → server ring buffer
- No sounddevice on server (browser IS the mic)
- EARS as engine: products (this app) consume it. Future: EARS API service.
- Python `performance-2x` Fly machine (not shared, not GPU — CPU is the right call)

### Still to build:
- [ ] Wire Gemini Live coaching to phrase_end events (voice coaching at phrase boundaries)
- [ ] Stream Gemini audio back to browser over WebSocket
- [ ] Feedback loop iteration using Daniel's recorded vocal sessions
- [ ] arXiv preprint (due before March 16)
- [ ] Demo video recording

## Recorded Vocal Sessions (for feedback loop — March 12 afternoon):
- Daniel recorded 2 vocal sessions for iteration
- Goal: listen to strain score accuracy vs felt strain, adjust formula weights
- Feedback loop plan: TBD next session

---

## Files Built Today:
- `/Users/daniel/Documents/projects/koda-comp-gemini/src/agent_entry1.py` — working Gemini Live + EARS demo
- `/Users/daniel/Documents/projects/koda-comp-gemini/docs/GEMINI-VS-EARS-GAP-ANALYSIS.md`
- `/Users/daniel/Documents/projects/koda-comp-gemini/docs/SESSION-STATE-2026-03-12.md`
- `/Users/daniel/Documents/projects/audio-perception/data/Macedonia St 27/29/30.m4a` — strain test recordings

## Test recordings (Daniel's voice):
- Macedonia St 27: easy mid ahh → strain_score 0.35 (GREEN)
- Macedonia St 29: strained high chest → strain_score 0.75 (RED)
- Macedonia St 30: head voice high → strain_score 0.41 (GREEN)

---

## Clinical Metrics Validation (March 12, ~1:45pm)

### Parselmouth (Praat) installed — clinical-grade metrics now available
`.venv/bin/pip install praat-parselmouth` ✅

### Results on Macedonia St recordings:

```
Clinical norms: Jitter < 1.0%  Shimmer < 3.0%  HNR > 20 dB

27 (easy):     F0=164Hz  Jitter=0.614%  Shimmer=3.83%  HNR=20.2dB
29 (strained): F0=262Hz  Jitter=0.265%  Shimmer=5.10%  HNR=19.5dB  ← HNR↓ Shimmer↑
30 (head):     F0=284Hz  Jitter=0.522%  Shimmer=4.29%  HNR=21.7dB  ← cleanest
```

### Key findings:
- HNR correctly orders: head > easy > strained ✅
- Shimmer correctly highest on strained ✅
- Jitter LOWER on strained (tension-type = rigidity = less pitch wobble) — skip jitter for now
- EARS is doing 70% of the work; clinical metrics refine

### Final combined strain formula:
```python
ears_score = (tonos * 0.6) + ((1 - thymos) * 0.4)
hnr_norm = max(0, min(1, (20 - hnr_db) / 30))      # 20dB=0, -10dB=1.0
shimmer_norm = min(1.0, shimmer_pct / 10.0)          # 0%=0, 10%+=1.0

strain_score = ears_score * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10

# Zones:
# < 0.40: GREEN (healthy)
# 0.40 - 0.65: YELLOW (approaching strain)
# > 0.65: RED (strain detected)
```

### Combined scores on test data:
- 27 (easy):     0.27 🟢 GREEN ✅
- 29 (strained): 0.54 🟡 YELLOW (conservative — short clip, phone quality)
- 30 (head):     0.31 🟢 GREEN ✅

### Dependencies added to competition project:
- `praat-parselmouth` for HNR + shimmer

### Real-time architecture:
- EARS (tonos/thymos): 100Hz continuous
- Clinical metrics (HNR, shimmer): 10Hz on 500ms windows via parselmouth
- Combined into single strain_score per WebSocket frame

---

## Afternoon/Evening Session (March 12, ~4–8pm)

### PYIN → YIN fix (most critical performance fix)

**Problem**: librosa.pyin on 1s window = 2716ms/frame. At 4x playback speed (chunks every 25ms), only 6 EARS frames total for a 23.2s recording.

**Root cause**: analysis loop `await`-ed pitch + strain in parallel but pitch dominated.

**Fix**: Replaced pyin with `librosa.yin` on last 250ms of ring buffer = 3ms. Reduced Parselmouth window from 500ms → 100ms = 78ms. Total per-frame ~82ms.

**Result**: 6 frames → 213 frames at 4x speed. 35x improvement.

```python
# Before: librosa.pyin on 1s = 2716ms
# After: librosa.yin on 250ms (last 11025 samples) = 3ms
window = audio[-11025:] if len(audio) >= 11025 else audio
f0 = librosa.yin(window, fmin=60.0, fmax=1200.0, sr=SAMPLE_RATE, hop_length=441)
valid_f0 = f0[(f0 > 70) & (f0 < 1100)]  # exclude boundary noise
```

### Threshold recalibration
- **Current thresholds**: GREEN < 0.50, YELLOW 0.50–0.68, RED > 0.68
- **Note**: Thresholds were updated from original (0.40/0.65) to (0.50/0.68) to match MelExtractor tonos range
- **Key insight from MelExtractor**: `compute_emotion_properties` returns tonos in a different scale than original formula assumed. Shifted thresholds accordingly.

### EMA smoothing + ring buffer reset
```python
EMA_ALPHA = 0.35
# In singer_ws (on new client connect):
_ring = np.zeros(EARS_WINDOW_SAMPLES, dtype=np.float32)
_ema_strain = 0.0
# In _analysis_loop:
_ema_strain = EMA_ALPHA * raw_strain + (1.0 - EMA_ALPHA) * _ema_strain
```

### Infrastructure
- **fly.toml**: `min_machines_running = 0 → 1` — eliminates 20s cold start
- **start.sh**: wrapper script for Fly deployment
- **GET /test**: serves frontend/test.html (file streaming test page)
- **GET /recordings**: returns JSON list of `.m4a` files from Vocal test recording sessions/
- **GET /recordings/{filename}**: serves audio file (path traversal protected)

### Test infrastructure built
- **`frontend/test.html`**: Pre-loads all recordings, streams to WebSocket at 1x/2x/4x/8x speed. Simulates real mic for offline iteration. Fixed `t=?` bug (ev.session_t not ev.t).
- **`scripts/watch_ui.py`**: Playwright screenshot loop — opens test page, streams, takes screenshots every N seconds. Saves to `/tmp/koda_watch/` for Koda to observe UI changes.

### CLI annotation tool (`scripts/annotate.py`)
Real-time annotation tool: plays recordings + labels strain as you listen.

**Final model: set-and-persist**
- Press G/Y/R once → zone persists, samples at 100ms continuously until changed
- SPACE pauses/resumes sampling (intentional gap)
- Q quits and saves
- Summary shows % of total song duration (not % of annotated time)

**Output format** (`docs/annotations/<name>.json`):
```json
{
  "file": "Danny - Chris Young R1.m4a",
  "duration": 178.32,
  "annotations": [
    {"t": 1.23, "zone": "green", "type": "start"},
    {"t": 45.67, "zone": "yellow", "type": "change"},
    {"t": 45.70, "zone": "yellow", "type": "auto"},
    ...
  ]
}
```
`type` values: `auto` (100ms continuous sample), `start` (first key), `change` (zone shift), `pause`, `resume`

### Comparison chart tool (`scripts/compare.py`)
3-panel matplotlib chart:
1. **Top**: EARS strain timeline with smoothed line + zone bands + avg/peak stats
2. **Middle**: Human annotation zones as colored spans + step-line + change markers
3. **Bottom**: Agreement bar (green = zone match, red = mismatch)

```
python scripts/compare.py                           # pick from annotated files
python scripts/compare.py "Danny - Chris Young R1"  # by name
```

### Annotation results — First pass (rough, take with grain of salt)
Daniel annotated all 3 recordings by ear during real-time playback.

| Recording | Duration | Labeled | G | Y | R |
|-----------|----------|---------|---|---|---|
| Chris Young R1 | ~3min | 10.2s | 0% | 100% | 0% |
| Liza Jane R1 | ~2min | 33.0s | 0% | 91% | 9% |
| Runnin Down R1 | ~3min | 4.3s | 0% | 100% | 0% |

**Known issues with this data:**
- Pressed YELLOW too often (yellow-by-default bias — uncertainty → middle zone)
- Was late on presses (200–500ms annotation lag vs felt experience)
- Low coverage: mostly pressed during noticed strain events, not throughout
- Runnin Down coverage especially thin (4.3s labeled of ~180s total)

**RED spike in Liza Jane**: correctly caught chorus at ~47–55s (confirmed strained moment)

### EARS vs human agreement (comparison charts)
| Recording | Agreement |
|-----------|-----------|
| Danny - Chris Young R1 | 68% |
| Danny - Liza Jane R1 (longer) | 44% |

**Pattern**: EARS over-calls GREEN what Daniel labels YELLOW. Suggests:
- STRAIN_GREEN threshold (0.50) may be too high — shift toward ~0.40–0.45
- EARS misses RED moments Daniel felt in Liza Jane chorus → investigate formula reweighting
- BUT: Daniel's annotations are biased toward YELLOW anyway (see above) — hard to know ground truth yet

### Known open bug
**Ring buffer "filled" flag**: When `_ring` is reset on new client connect, the `filled` local variable in `_analysis_loop` is NOT reset. First few frames post-reconnect may run on stale buffer assumption. Not yet fixed.

---

## Calibration Session (March 12, ~8-9pm)

### Anchor clips recorded
`Vocal test recording sessions/Anchors/`:
- Ahh easy 1.m4a → easy/relaxed
- Ahh pushed 1.m4a → pushed
- Ahh pushed 2 note.m4a → pushed
- Ahh relaxed 2 note.m4a → relaxed

HNR on anchors: easy=18.0dB, pushed1=22.5dB, pushed2=24.9dB, relaxed=20.9dB
Shimmer on anchors: easy=7.3%, pushed1=3.7%, pushed2=3.2%, relaxed=5.1%
→ Pressed phonation confirmed: pushed = higher HNR + lower shimmer

### Dim scan — all EARS dimensions vs annotations
`scripts/dim_scan.py` runs full analyze_mel on all reviewed annotation files.

**Top signals (r > 0.50):**
- `temporal.rhoe_diaphora` r=+0.583 — energy flow range within window
- `temporal.rhoe_megiste` r=+0.577 — max energy flow
- `life.metabole` r=+0.558 — rate of acoustic change
- `temporal.dynamis_diaphora` r=+0.534 — power variance

BUT: these temporal dims measure phrase-level energy dynamics, not fold tension. Work for songs (transitions/dynamics), don't fire on isolated sustained notes in anchors.

**Key insight: session-adaptive baseline required.**
HNR varies by 6-10 dB between recording conditions (mic distance, room, recording method). Absolute thresholds don't work. Session-relative formula does.

### Final formula (session-adaptive, Mar 12)
```python
# First BASELINE_FRAMES (20) voiced frames → establish session baseline
# Then measure deviation:
hnr_strain  = max(0, (hnr_db - session_hnr_baseline) / 8.0)          # 55% weight
shim_strain = max(0, (session_shim_baseline - shimmer_pct) / 5.0)     # 30% weight
dyn_strain  = max(0, (raw_dynamis/session_dyn_baseline - 1.0) / 3.0)  # 15% weight
strain = hnr_strain * 0.55 + shim_strain * 0.30 + dyn_strain * 0.15

# Thresholds (anchor-validated):
# GREEN < 0.35, YELLOW 0.35-0.55, RED > 0.55
```

Validated on anchors: easy→0.000 🟢, pushed1→0.45 🟡, pushed2→0.69 🔴 ✓

### Calibration tools built
- `scripts/review.py` — segment-by-segment annotation (2s chunks, replay, skip)
- `scripts/calibrate.py` — formula accuracy analysis per zone
- `scripts/dim_scan.py` — ranks ALL EARS dims by correlation with annotations
- `scripts/compare.py` — now uses annotation filename (not audio filename) for output

### YouTube vocal instructor pipeline (Mar 12 evening)
`scripts/yt_harvest.py` — downloads vocal instructor videos, extracts strain-labeled segments:
1. yt-dlp downloads audio + auto-captions
2. Scan captions for keywords: strained/pushed/relaxed/easy/supported/tight/etc
3. Extract 3s audio around each keyword timestamp
4. Label automatically from keyword context
5. Output annotation JSON + audio segments for review.py refinement

## Formula v5 — VALIDATED (later March 12 session)

### Dual-signal: `strain = min(1.0, max(shim_dev, dyn_dev))`

**The HNR problem**: HNR rises for BOTH pressed strain AND relaxed high notes. On Medium 1, singing a high note naturally → HNR 21-24dB, same as hard push. HNR excluded from formula.

**Two signals, two pathologies**:
- `shim_dev = (shimmer - baseline) / 10.0` — catches rough phonation (Rough 1: shimmer 16-19%)
- `dyn_dev = (temporal.dynamis_diaphora_300ms - baseline) / 1.0` — catches pressed phonation (hard push: dyn 1.007 vs relaxed 0.13-0.18)

**300ms EARS window**: `temporal.dynamis_diaphora` = energy variance. Brief squeeze/push creates high variance. Hard push 300ms = 1.007 vs easy 0.131-0.175 (5-7x separation). Dilutes on 1s window.

**P80 summary metric**: Instead of "dominant zone" (frame vote count), use 80th percentile score.
- Rationale: if top 20% of frames reach yellow/red, the clip qualifies as that zone
- Hard push: P80=0.357 → YELLOW ✓ (was incorrectly GREEN with dominant-zone method)
- Rough 1: P80=0.836 → RED ✓

**Final anchor scorecard**: 4/4 ✓ (Easy2=GREEN, Medium1=GREEN, hard push=YELLOW, Rough1=RED)

### Files updated:
- `src/vocal_health_backend.py` — formula v5, SEED_DYN_BASELINE, 300ms EARS window, dyn_dev signal
- `scripts/live_sim.py` — EARS imports, ears_dyn_300ms(), strain_v5(), P80 summary

---

## NEXT SESSION priorities (Gemini integration — target 10pm March 12)
1. Wire `phrase_end` events → Gemini Live API (voice coaching at phrase boundaries)
2. Stream Gemini audio back to browser over WebSocket
3. Improve annotation ground truth (see annotation scoring improvements below)
4. arXiv preprint draft
5. Demo video recording

## Annotation scoring — identified improvements needed
Real-time annotation has 3 known failure modes:
1. **Yellow bias**: uncertain → yellow (middle) by default
2. **Annotation lag**: press 200–500ms after felt strain
3. **Attention split**: singing/listening + labeling at same time

**Best next approach**: Segment-by-segment post-hoc annotation
- Auto-chop audio into 2–3s segments
- Play each segment, pause → press G/Y/R → replay until confident → confirm
- No time pressure, can replay ambiguous segments
- Eliminates lag: you're rating PAST audio, not predicting
- Expected quality: significantly higher — worth building before calibrating formula weights

**Alternative**: Comparison-based (AXB)
- "Is clip A or B more strained?" pairwise comparison
- Psychoacoustically much easier than absolute G/Y/R judgment
- Produces ranked order → maps to zones
- More work to build, but gold-standard for perceptual annotation

**Files**:
- Charts: `docs/strain-charts/*_comparison.png`
- Annotations: `docs/annotations/Danny - *.json`
