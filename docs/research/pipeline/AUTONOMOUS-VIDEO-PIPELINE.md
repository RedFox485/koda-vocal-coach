# Autonomous Demo Video Pipeline — Architecture

**Goal:** Given a script, a singing audio file, and optionally a DJI clip, Claude autonomously produces a complete, polished, production-grade demo video with zero human intervention during the render phase.

**Philosophy:** Every janky demo video you've seen at a hackathon is janky because it was screen-recorded live with things going wrong. We eliminate that entirely — every frame is controlled, every audio track is mixed with surgical precision, every timing decision is made in code.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUTS (Human provides)                   │
├─────────────────────────────────────────────────────────────┤
│  1. Script markdown (shot list + voiceover text)            │
│  2. Singing audio file (.wav) — Daniel or sourced           │
│  3. DJI clip (.mp4) — 3s hook shot (optional)               │
│  4. Voice clone sample (.wav) — for ElevenLabs cloning      │
│  5. Pipeline config (resolution, timing overrides, etc.)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 1: AUDIO PRODUCTION                    │
├─────────────────────────────────────────────────────────────┤
│  a) Parse script → extract voiceover blocks                  │
│  b) Generate voiceover clips via ElevenLabs API              │
│     - Per-block emotion/pacing settings                      │
│     - Save as: vo_01.wav, vo_02.wav, ...                     │
│  c) Measure each clip duration → build master timeline       │
│  d) Generate Gemini coaching cues (TTS or pre-recorded)      │
│  e) Optional: generate ambient pad / background music        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 2: UI SIMULATION                        │
├─────────────────────────────────────────────────────────────┤
│  a) Launch Koda frontend in Playwright (headless Chrome)     │
│  b) Inject singing audio via WebSocket proxy                 │
│     - Backend processes it as if live mic input              │
│     - UI responds: strain gauge, pitch tuner, range map      │
│  c) Playwright captures video at 1080p60 or 4K              │
│     OR: frame-by-frame screenshot → ffmpeg encode            │
│  d) Orchestrate UI interactions:                             │
│     - Click "Start" at correct timestamp                     │
│     - Wait for Gemini greeting                               │
│     - Feed singing audio through WebSocket                   │
│     - Pause audio at phrase boundaries (coaching moments)    │
│     - Resume after coaching cue timing window                │
│  e) Capture coaching panel text animations                   │
│  f) Export: raw UI video (no audio — we mix separately)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 3: COMPOSITION                         │
├─────────────────────────────────────────────────────────────┤
│  a) Build master timeline from Phase 1 measurements          │
│  b) Layer video tracks:                                      │
│     - Track V1: DJI clip (0:00–0:03)                         │
│     - Track V2: Playwright UI capture (0:03–end)             │
│     - Track V3: Text overlays (burned in via ffmpeg)         │
│     - Track V4: Architecture diagram (static image, Shot 7)  │
│     - Track V5: Title card (Shot 8 end)                      │
│  c) Layer audio tracks:                                      │
│     - Track A1: Singing audio (ducked during voiceover)      │
│     - Track A2: Voiceover clips (placed at exact timestamps) │
│     - Track A3: Gemini coaching audio (at phrase boundaries)  │
│     - Track A4: Gemini greeting + song summary               │
│     - Track A5: Optional ambient pad (low volume)            │
│  d) Apply audio mixing rules:                                │
│     - Singing: 100% when alone, 40% under voiceover          │
│     - Gemini voice: always 100%, never ducked                 │
│     - Voiceover: 100%, never overlaps Gemini                  │
│  e) Apply transitions:                                       │
│     - Hard cuts between all shots                             │
│     - Optional 0.2s cross-dissolve: DJI → screen             │
│  f) Encode final output: H.264, 1080p, YouTube-optimized     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 4: QUALITY CHECK                        │
├─────────────────────────────────────────────────────────────┤
│  a) Extract keyframes at shot boundaries                     │
│  b) Claude vision reviews each keyframe:                     │
│     - Is the UI visible and clean?                           │
│     - Are text overlays readable?                            │
│     - Is the strain gauge in the right zone for this shot?   │
│  c) Check audio alignment:                                   │
│     - Voiceover-to-visual sync                               │
│     - No overlap between voiceover and Gemini                │
│  d) Verify total runtime                                     │
│  e) Generate QA report                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                     │
├─────────────────────────────────────────────────────────────┤
│  final_demo.mp4 — YouTube-ready, production-grade            │
│  qa_report.md — keyframe screenshots + timing verification   │
│  timeline.json — full timing manifest for re-renders         │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Deep Dives

### 1. WebSocket Audio Injection (The Core Trick)

The Koda frontend captures mic audio via `getUserMedia` and sends it over WebSocket to the backend. To simulate this programmatically:

**Option A: WebSocket Proxy Script**
```python
# Concept: Python script that reads a WAV file and sends chunks
# over WebSocket exactly as the browser would — 100ms chunks at 10Hz

import asyncio
import websockets
import numpy as np
import soundfile as sf

async def inject_audio(ws_url, audio_path, chunk_ms=100):
    audio, sr = sf.read(audio_path)
    chunk_size = int(sr * chunk_ms / 1000)

    async with websockets.connect(ws_url) as ws:
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size].astype(np.float32)
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_ms / 1000)  # Real-time pacing
```

This means the backend processes the audio identically to a live session — strain gauge, pitch tuner, everything responds exactly as it would with a real singer.

**Option B: Modify Frontend to Accept File Input**
Add a hidden "file injection" mode to the frontend JS that reads a WAV file instead of mic input. Advantage: captures the full round-trip including browser-side rendering. Disadvantage: harder to control timing.

**Recommendation: Option A** — cleaner separation, backend doesn't know the difference, and we control timing precisely from the orchestrator script.

**Pause/Resume for Coaching Moments:**
The orchestrator needs to pause audio injection at specific timestamps (phrase boundaries) to create the natural breath pauses where Gemini would coach. These timestamps come from the script/timeline config:

```json
{
  "coaching_pauses": [
    {"at_seconds": 45, "pause_duration": 8, "label": "coaching_cue_1"},
    {"at_seconds": 62, "pause_duration": 8, "label": "coaching_cue_2"},
    {"at_seconds": 80, "pause_duration": 14, "label": "song_end_summary"}
  ]
}
```

During each pause, the orchestrator:
1. Stops sending audio chunks
2. Waits for the UI to show the coaching trigger
3. Waits the configured pause duration
4. Resumes audio injection

### 2. Playwright Video Capture

**Native Recording:**
```python
context = browser.new_context(
    record_video_dir="./captures/",
    record_video_size={"width": 1920, "height": 1080},
    viewport={"width": 1920, "height": 1080},
    device_scale_factor=2  # 2x for crisp text
)
```

Playwright's native video uses VP8 codec in WebM. Quality is decent but not 60fps — typically 25-30fps.

**Frame-by-Frame Capture (Higher Quality):**
```python
# Take screenshots at controlled intervals
# Then compose with ffmpeg for guaranteed smooth output

frames = []
for frame_num in range(total_frames):
    # Advance the simulation state
    await orchestrate_next_frame(page, frame_num)

    # Capture frame
    await page.screenshot(path=f"frames/{frame_num:06d}.png")

# Compose at exact 60fps
# ffmpeg -framerate 60 -i frames/%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

**Advantage of frame-by-frame:** We can decouple real-time from capture-time. We can take 200ms per frame if needed (for heavy pages) and still output perfect 60fps. No dropped frames, ever.

**Hybrid Approach (Recommended):**
- Use Playwright's native recording for the "live" UI segments (singing + gauge moving)
- Use frame-by-frame for static moments (architecture diagram, title card)
- Compose all segments in ffmpeg

### 3. Gemini Coaching Audio Strategy

**Problem:** We want perfect coaching cues at perfect times, but real Gemini responses are unpredictable in timing and content.

**Solution: Record coaching cues separately, overlay in post.**

Approach:
1. Run Koda normally, sing the song, let Gemini coach naturally
2. Record each coaching cue as a separate audio file
3. OR: Use the Gemini API directly to generate coaching responses to specific prompts, capture the audio output
4. In the final composition, place these audio clips at exact timestamps

**Generating coaching cues on demand:**
```python
# Use Gemini Live API to generate a coaching response
# Feed it a specific scenario: "The singer just completed a yellow-zone phrase,
# shimmer at 0.15, CPP deviation -0.08, phrase duration 3.2 seconds"
# Capture the audio response
```

This gives us:
- Perfect content (relevant to what's on screen)
- Perfect timing (placed at exact frame in the timeline)
- Multiple takes (generate 3-5 options, pick the best)

### 4. Audio Mixing with FFmpeg

```bash
# Layer all audio tracks with precise timing and volume control
ffmpeg \
  -i ui_capture.mp4 \           # Video track (from Playwright)
  -i singing.wav \               # Singing audio
  -i vo_01.wav \                 # Voiceover clip 1
  -i vo_02.wav \                 # Voiceover clip 2
  -i coaching_cue_1.wav \        # Gemini coaching cue
  -filter_complex "
    [1:a]volume=1.0[singing];
    [2:a]adelay=3000|3000[vo1];          # Voice clip 1 starts at 3s
    [3:a]adelay=12000|12000[vo2];        # Voice clip 2 starts at 12s
    [4:a]adelay=65000|65000[coach1];     # Coaching cue at 65s

    # Duck singing under voiceover
    [singing][vo1]sidechaincompress=threshold=0.01:ratio=4[singing_ducked1];

    # Mix all tracks
    [singing_ducked1][vo1][vo2][coach1]amix=inputs=4
  " \
  -c:v copy \
  output.mp4
```

For more complex mixing, MoviePy might be cleaner:

```python
from moviepy.editor import *

# Load clips
ui_video = VideoFileClip("ui_capture.mp4")
dji_clip = VideoFileClip("dji_hook.mp4").subclip(0, 3)

# Load audio
singing = AudioFileClip("singing.wav")
vo_clips = [AudioFileClip(f"vo_{i:02d}.wav") for i in range(1, 10)]
coaching = AudioFileClip("coaching_cue_1.wav")

# Place voiceover at timestamps from timeline
vo_placed = [clip.set_start(t) for clip, t in zip(vo_clips, vo_timestamps)]

# Composite
final_audio = CompositeAudioClip([singing, *vo_placed, coaching])
final_video = concatenate_videoclips([dji_clip, ui_video])
final_video = final_video.set_audio(final_audio)
final_video.write_videofile("final_demo.mp4", fps=60, codec="libx264")
```

### 5. Text Overlay System

**Option A: In-Browser (preferred for Koda)**
Add a hidden overlay layer to the Koda frontend that the orchestrator controls:
```javascript
// Orchestrator sends commands via page.evaluate()
showOverlay("Parselmouth · Shimmer · HNR", "lower-third", 2000);
```
Advantage: Text renders with the page, captured naturally by Playwright.

**Option B: FFmpeg Post-Processing**
```bash
ffmpeg -i video.mp4 \
  -vf "drawtext=text='96ms Pipeline Latency':fontsize=24:fontcolor=white:
       x=100:y=h-100:enable='between(t,25,28)'" \
  output.mp4
```
Advantage: No frontend modification needed. Disadvantage: Less control over styling.

**Recommendation: Option A** — we already control the frontend, and in-browser text overlays match the UI aesthetic naturally.

---

## The Singing Audio Question

### Option 1: Daniel Records Fresh
**Pros:** Authentic, personal connection, matches the "I built this for myself" narrative
**Cons:** Recording quality depends on room/mic, may need multiple takes

### Option 2: Use a Professional Vocal Sample
**Pros:** Guaranteed quality, clear strain contrast between easy/hard sections
**Cons:** Less authentic, licensing questions, "who is this singer?"
**Sources for royalty-free vocal stems:**
- Cambridge-MT multitrack library (free, academic use)
- MUSDB18 dataset (free for research)
- Splice (paid, high quality stems)
- YouTube Audio Library (limited vocal content)

### Option 3: Use Daniel's Existing Test Recordings
**Pros:** Already recorded, proven to trigger strain gauge correctly
**Cons:** May not be highest quality
**Files we have:**
- `calibration_AB.wav` — 52s, normal → strained singing
- Liza Jane recordings (from earlier testing)

### Recommendation: Daniel records fresh with intent
Record specifically FOR the video — choose a song with clear verse (easy) / chorus (hard) sections, sing it 3 times, pick the best take. The singing quality matters because it's audible in the final video. Use iPhone mic in a quiet room — surprisingly good quality for this purpose.

**Alternative for future pipeline runs:** Source a professional vocal stem and feed it through. This makes the pipeline fully autonomous (no human recording needed). Good for portfolio demos of other projects.

---

## Orchestrator Script Design

The main pipeline script (`scripts/render_demo.py`) orchestrates everything:

```python
"""
Demo Video Renderer — Autonomous production pipeline

Usage:
    python scripts/render_demo.py \
        --script docs/DEMO-VIDEO-PLAN.md \
        --singing audio/singing_take_best.wav \
        --dji video/dji_hook.mp4 \
        --voice-id <elevenlabs_voice_id> \
        --output output/final_demo.mp4
"""

class DemoRenderer:
    def __init__(self, config):
        self.config = config
        self.timeline = Timeline()

    async def render(self):
        # Phase 1: Audio Production
        vo_clips = await self.generate_voiceover()
        coaching_clips = await self.generate_coaching_cues()
        self.timeline.build(vo_clips, coaching_clips)

        # Phase 2: UI Simulation
        ui_video = await self.capture_ui_simulation()

        # Phase 3: Composition
        final = await self.compose(ui_video, vo_clips, coaching_clips)

        # Phase 4: QA
        report = await self.quality_check(final)

        return final, report
```

---

## Reusability for Other Projects

This pipeline is project-agnostic with the right abstractions:

| Component | Koda-Specific | Reusable |
|-----------|--------------|----------|
| Script parser | Shot list format | Yes — any markdown shot list |
| Voiceover generation | Voice ID, emotion map | Yes — pass any voice + script |
| WebSocket injection | Koda audio format | Partially — protocol varies |
| Playwright capture | Koda URL + interactions | Yes — any web app URL |
| FFmpeg composition | Track count, timing | Yes — generic audio/video layers |
| Text overlays | Koda styling | Yes — configurable styles |
| QA vision check | Koda-specific pass criteria | Yes — configurable criteria |

**To use for a different project:**
1. Write a shot list markdown in the same format
2. Provide the web app URL and interaction script
3. Swap the voice ID and emotion settings
4. Run the pipeline

---

## Implementation Priority

For the current Koda competition (deadline: today):

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| **P0** | WebSocket audio injection script | 2 hrs | Core — nothing works without this |
| **P0** | Voiceover generation (ElevenLabs API) | 1 hr | Script → audio clips |
| **P1** | Playwright UI capture | 2 hrs | Smooth video vs janky screen recording |
| **P1** | FFmpeg composition script | 2 hrs | Layers everything together |
| **P2** | Text overlay system | 1 hr | Polish — can skip for v1 |
| **P2** | QA vision check | 1 hr | Nice to have — manual review works |
| **P3** | Gemini coaching cue generation | 2 hrs | Can use live recordings instead |
| **P3** | Full orchestrator with config | 3 hrs | For future competitions |

**Minimum viable pipeline (6 hours):** WebSocket injection + voiceover generation + Playwright capture + FFmpeg composition = a complete rendered video with no screen recording.

**Full pipeline (13 hours):** Everything above = fully autonomous, reusable for any project.

---

## File Structure

```
scripts/
  render_demo.py          # Main orchestrator
  inject_audio.py         # WebSocket audio injection
  generate_voiceover.py   # ElevenLabs API wrapper
  capture_ui.py           # Playwright video capture
  compose_video.py        # FFmpeg/MoviePy composition
  qa_check.py             # Vision-based QA

config/
  demo_timeline.json      # Shot timestamps, audio placements
  voice_settings.json     # ElevenLabs voice ID, per-shot emotion

audio/
  singing/                # Input singing audio files
  voiceover/              # Generated voiceover clips
  coaching/               # Gemini coaching cue recordings
  mixed/                  # Intermediate mixed audio

video/
  dji/                    # DJI Osmo 2 clips
  captures/               # Playwright video captures
  frames/                 # Frame-by-frame screenshots (if used)
  overlays/               # Text overlay renders

output/
  final_demo.mp4          # Final rendered video
  qa_report.md            # Quality check report
  timeline.json           # Realized timeline for debugging
```
