# Video Production Tools Research

**Date:** 2026-03-16
**Purpose:** Evaluate programmatic video production tools for Koda competition video pipeline.

---

## 1. MoviePy

**Current Version:** 2.1.x (v2.0 was a major rewrite)
**PyPI:** https://pypi.org/project/moviepy/
**GitHub:** https://github.com/Zulko/moviepy

### What Changed in v2.0
- Dropped Python 2; requires Python 3.7+
- Effects are now classes, not functions (breaking API change)
- Removed dependencies on ImageMagick, PyGame, OpenCV, scipy, scikit
- Now relies primarily on Pillow for image manipulation
- More structured, object-oriented API

### Performance (Critical Issue)
- **v2.x is 10x slower than v1.x** for rendering. A 3-minute video took 18m32s on v2.1.2 vs 1m39s on v1.0.3.
- v2.1.2 further regressed from v2.1.1 (10 iter/s vs 23 iter/s).
- For 1080p60 production, MoviePy v2 is likely too slow for iterative workflows.
- **Alternative:** [MovieLite](https://github.com/francozanardi/movielite) — performance-focused fork using Numba.

### Audio Mixing
- Supports `CompositeAudioClip` for layering multiple audio tracks.
- Basic volume control per clip. No built-in sidechain/ducking.
- Adequate for simple layering; complex mixing should go through FFmpeg directly.

### Text Overlays & Transitions
- `TextClip` for text overlays (now Pillow-based, was ImageMagick).
- Built-in crossfade transitions between clips.
- Custom effects possible via the new class-based system.

### Maintenance Status
- Actively seeking maintainers. Maintained by a small volunteer group with inconsistent bandwidth.
- Documentation updated January 2025.
- **Risk:** Not dead, but not thriving. Breaking performance regressions in point releases.

### Verdict
**Use for:** Quick prototyping, simple compositing, text overlays.
**Avoid for:** Final 1080p60 renders (too slow). Use FFmpeg directly for final encode.

---

## 2. FFmpeg Python Wrappers

### ffmpeg-python (kkroening/ffmpeg-python)
- **GitHub:** https://github.com/kkroening/ffmpeg-python
- **Last release:** 2019 (0.2.0). No releases in 5+ years.
- **Status:** Effectively abandoned. Issues still being opened, PRs unmerged.
- **Strengths:** Elegant API for building complex filter graphs. Good docs.
- **Weakness:** Stale. Doesn't support newer FFmpeg features/filters.

### pyffmpeg (modern, on PyPI)
- **PyPI:** https://pypi.org/project/pyffmpeg/
- **Latest:** v2.5.2.3.2 (February 2026) — actively maintained.
- **Key feature:** Bundles its own FFmpeg binary. No system install needed.
- **Good for:** Simple operations. Less flexible for complex filter graphs.

### python-ffmpeg
- Newer library, actively maintained. Smaller community. Fewer methods.
- Promising but less battle-tested.

### PyAV
- Direct Python bindings to FFmpeg's libav* libraries (not a CLI wrapper).
- Actively maintained. Used by Manim internally (as of v0.19.0).
- Good for frame-level access and streaming pipelines.
- More complex API — you work with packets/frames, not CLI flags.

### Recommendation
**For this project:** Use `subprocess` to call FFmpeg directly for final renders (full control, no wrapper lag). Use `ffmpeg-python` for prototyping filter graphs (its API is still the best for building complex graphs, even if unmaintained). Consider `PyAV` if we need frame-level manipulation in Python.

---

## 3. Remotion

**Website:** https://www.remotion.dev/
**GitHub:** https://github.com/remotion-dev/remotion
**Language:** TypeScript/React

### Overview
- Compose videos using React components. Full access to CSS, Canvas, SVG, WebGL.
- Render to MP4/WebM. Server-side rendering supported.
- Very high quality output — pixel-perfect since it's browser-rendered.

### Python Integration
- Official Python SDK exists for triggering renders via Remotion Lambda (AWS).
- `RemotionClient` — construct with region, serve URL, function name, then trigger renders.
- Requires a Node.js project for the video template; Python just triggers renders.

### Pricing
- **Free:** Individuals and small teams.
- **Company:** $100/month (teams of 4+).
- **Enterprise:** $500/month.
- Lambda rendering costs extra (AWS charges).

### Verdict
**Overkill for this project.** Remotion shines for template-based batch rendering (personalized videos at scale). For a one-off competition video, the React/Node.js setup overhead isn't justified. The Python integration is just a render trigger, not a composition tool.

**Consider if:** We want browser-quality motion graphics or animated data visualizations that would be painful in FFmpeg.

---

## 4. Manim (3Blue1Brown's Tool)

**Website:** https://www.manim.community/
**GitHub:** https://github.com/ManimCommunity/manim
**Current Version:** 0.20.1 (community edition)

### Capabilities
- Python framework for creating mathematical/explanatory animations.
- Produces high-quality vector animations: graphs, diagrams, code walkthroughs, architecture visualizations.
- Output: MP4 or individual frames (PNG sequence).

### Recent Changes
- **v0.19.0:** Replaced external FFmpeg dependency with PyAV (bundled). Simpler install.
- Currently undergoing a **major refactor** — new features not being accepted during this period.
- Active community, Discord server.

### Generative Manim
- AI-powered tool that generates Manim animations from natural language prompts.
- Uses GPT-4o and Claude Sonnet to generate animation code.
- Could be useful for rapid prototyping of diagram animations.

### Use for Architecture Segment
**Yes — this is the right tool for animated architecture diagrams.** Manim excels at:
- Showing data flow between components (arrows, highlights, transforms)
- Building up diagrams piece by piece with animations
- Code/text animations
- Clean, professional look (the 3Blue1Brown aesthetic)

### Integration with Pipeline
- Render Manim scenes to MP4 or PNG sequences.
- Composite into final video using FFmpeg or MoviePy.
- Can control resolution and frame rate.

### Verdict
**Use for:** Architecture diagrams, system flow animations, any "explainer" visuals.
**Not for:** Full video composition, live footage editing, audio mixing.

---

## 5. Editly

**GitHub:** https://github.com/mifi/editly
**npm:** https://www.npmjs.com/package/editly
**Language:** Node.js

### Overview
- Declarative, JSON-driven video editing. Define clips, transitions, overlays in JSON/JSON5.
- Built on FFmpeg and headless GL (for transitions).
- Supports: clips, images, titles, custom HTML (via Puppeteer), GL transitions, audio overlay.

### Maintenance
- Maintained by a single developer (mifi, author of LosslessCut).
- Still active but solo maintenance = risk.

### Capabilities
- Built-in transitions (fade, dissolve, GL-based).
- Title cards, subtitle overlays.
- Audio mixing (background music + per-clip audio).
- Custom HTML/CSS rendered via Puppeteer for complex overlays.

### Verdict
**Interesting but niche.** The JSON-driven approach is appealing for reproducibility, but:
- Node.js dependency (not Python-native).
- Single maintainer risk.
- For our use case, FFmpeg + MoviePy gives more control.

---

## 6. Shotstack API

**Website:** https://shotstack.io/
**Pricing:** https://shotstack.io/pricing/

### Pricing Model
- **Free tier:** 20 minutes/month, 100 images/month.
- **Pay-as-you-go:** $0.40/minute rendered.
- **Subscription:** $0.20/minute (lower at volume).
- **Overage:** 30% premium on plan rate.
- **Rollover:** Up to 3x monthly budget.
- 1 credit = 1 minute of video, regardless of resolution.

### Capabilities
- JSON-based timeline API. Define clips, text, audio, transitions.
- Cloud rendering (no local compute needed).
- Merge fields for templated/personalized videos.

### Verdict
**Not worth it for a one-off video.** The free tier (20 min) might cover iterating on a short video, but you lose control over encoding quality. Better for batch/template use cases. Local FFmpeg rendering is free and gives full control.

---

## 7. AI Avatar Generators (Synthesia / HeyGen / D-ID)

### HeyGen
- **Pricing:** Creator plan $24/month (annual). Unlimited video generation on paid plans.
- **API:** Starts from $5.
- **Quality:** Avatar IV is the most realistic — sophisticated motion capture, natural eye movements, fluid hand gestures. Best lip-sync of the three.
- **Languages:** 175+ languages with real-time translation.
- **Best for:** Marketing/training videos with a talking head.

### Synthesia
- **Pricing:** Starter $29/month ($18/year). Creator $89/month ($64/year).
- **Limits:** 10 minutes/month on lowest paid plan (strict caps).
- **Quality:** Professional-grade, accurate lip-sync. Not quite HeyGen Avatar IV level.
- **Strength:** Enterprise features (SOC 2 Type II). Fortune 500 clientele.

### D-ID
- **API Pricing:** Build tier $18/month — 32 min streaming or 16 min regular video.
- **Credits:** 1 credit = 15 seconds. 40-second video = 3 credits.
- **Limit:** Max 5-minute videos.
- **Quality:** Good for photo-to-video (animate a still image). Less natural than HeyGen/Synthesia for full avatars.

### For Competition Video
**Probably not needed.** A talking head avatar would feel generic for a technical demo. Better to use:
- Screen recordings with voiceover
- Animated diagrams (Manim)
- Text overlays and motion graphics

**If we do want a presenter:** HeyGen offers the best quality-to-price ratio. The $24/month plan with unlimited generation would let us iterate.

---

## 8. Audio Ducking Techniques

### FFmpeg sidechaincompress
The built-in approach for programmatic ducking:

```bash
ffmpeg -i voiceover.wav -i music.wav \
  -filter_complex \
  "[1:a]asplit=2[sc][mix]; \
   [0:a][sc]sidechaincompress=threshold=0.03:ratio=4:attack=200:release=1000[compr]; \
   [compr][mix]amix=duration=first" \
  -c:a aac -b:a 320k ducked_output.mp4
```

**Parameters:**
- `threshold`: Level at which ducking kicks in (lower = more sensitive). Default 0.125.
- `ratio`: How much to reduce. 4:1 means 4dB over threshold becomes 1dB.
- `attack`: How fast ducking engages (ms). 200ms is smooth.
- `release`: How fast ducking releases (ms). 1000ms prevents pumping.

### Manual Volume Envelopes
For more control, use FFmpeg's `volume` filter with expression-based automation:

```bash
ffmpeg -i music.wav -af "volume='if(between(t,5,15),0.2,1)':eval=frame" output.wav
```

This drops music to 20% volume between seconds 5-15. Can chain multiple conditions for segment-based control.

### Recommendation
**Use sidechaincompress for voiceover-over-music** — it's automatic and sounds natural. For precise control (e.g., specific singing passages), use manual volume envelopes calculated from voiceover timestamps.

### Python Integration
Build the FFmpeg command string in Python using timestamps from your script/voiceover file. The voiceover audio acts as the sidechain signal.

---

## 9. YouTube Upload Optimization

### Recommended Settings (Official + Best Practice)

| Setting | Value |
|---------|-------|
| **Container** | MP4 (.mp4) |
| **Video Codec** | H.264 (libx264) |
| **Audio Codec** | AAC-LC |
| **Resolution** | 1920x1080 (or higher) |
| **Frame Rate** | Match source (60fps if source is 60fps) |
| **Video Bitrate (1080p60)** | 12-15 Mbps |
| **Audio Bitrate** | 320 kbps |
| **Audio Sample Rate** | 48 kHz |
| **H.264 Profile** | High |
| **H.264 Level** | 4.0 (or 4.2 for 60fps) |
| **Pixel Format** | yuv420p |
| **Color Space** | Rec. 709 (SDR) |
| **Keyframe Interval** | 2 seconds (GOP = framerate * 2) |

### FFmpeg Command for YouTube 1080p60

```bash
ffmpeg -i input.mov \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -profile:v high \
  -level 4.2 \
  -pix_fmt yuv420p \
  -r 60 \
  -g 120 \
  -bf 2 \
  -c:a aac \
  -ar 48000 \
  -b:a 320k \
  -movflags +faststart \
  output.mp4
```

**Key flags:**
- `-crf 18`: High quality (lower = better, 18 is visually lossless). YouTube re-encodes anyway, so upload high quality.
- `-preset slow`: Better compression efficiency. Use `medium` if time-constrained.
- `-movflags +faststart`: Moves moov atom to start of file for streaming.
- `-g 120`: Keyframe every 2 seconds at 60fps.

### Codec Strategy
- **Upload as H.264.** YouTube re-encodes everything to VP9 (for 1080p+) and AV1 (for popular videos). Uploading VP9/AV1 yourself doesn't skip re-encoding.
- **Upload at higher quality than target.** YouTube's re-encoding is lossy, so start with CRF 18 or lower.
- **1440p trick:** Upload at 1440p even if content is 1080p — YouTube assigns higher bitrate VP9 streams to 1440p+ uploads, resulting in better perceived quality even when viewed at 1080p.

---

## 10. Subtitle/Caption Generation

### OpenAI Whisper (Best Option)

**GitHub:** https://github.com/openai/whisper
**Output formats:** SRT, VTT, JSON, TSV, TXT

```bash
# Command line
whisper audio.wav --model turbo -f srt --output_dir ./subs/

# Python API
import whisper
model = whisper.load_model("turbo")
result = model.transcribe("audio.wav")
# result contains segments with timestamps
```

### faster-whisper (Recommended for Speed)
- CTranslate2-based reimplementation. 4-8x faster than original Whisper.
- Same model quality, dramatically less compute.
- **GitHub:** https://github.com/SYSTRAN/faster-whisper

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.wav")
# Write SRT from segments
```

### Workflow for This Project
1. Record/generate voiceover audio.
2. Run faster-whisper to generate SRT with timestamps.
3. Optionally hand-edit SRT for accuracy.
4. Burn subtitles into video with FFmpeg:
   ```bash
   ffmpeg -i video.mp4 -vf "subtitles=subs.srt:force_style='FontSize=24,FontName=Arial'" output.mp4
   ```
5. Or upload SRT separately to YouTube (preferred — allows viewer toggle).

### Other Tools
- **whisper_autosrt:** CLI tool wrapping faster-whisper + optional Google Translate for multilingual subs.
- **Whisper.cpp:** C++ port for even faster inference. Good for CI/CD pipelines.

---

## Summary: Recommended Pipeline Stack

| Component | Tool | Reason |
|-----------|------|--------|
| **Video composition** | FFmpeg (subprocess) | Full control, fast, free |
| **Prototyping/overlays** | MoviePy 2.x | Quick iteration on text/image overlays |
| **Architecture animations** | Manim Community | Perfect for system diagrams |
| **Audio ducking** | FFmpeg sidechaincompress | Automatic, sounds natural |
| **Final encode** | FFmpeg (H.264, CRF 18) | YouTube-optimized |
| **Subtitles** | faster-whisper + SRT | Fast, accurate, free |
| **AI avatar (if needed)** | HeyGen ($24/mo) | Best quality/price ratio |

### What to Skip
- **Remotion** — overkill React setup for one video.
- **Editly** — interesting but Node.js, single maintainer.
- **Shotstack** — paying for cloud rendering we don't need.
- **Synthesia/D-ID** — more expensive, less quality than HeyGen.

### Architecture

```
Script/Timeline (JSON/Python)
    |
    +---> Manim (architecture animations) ---> MP4 segments
    |
    +---> Screen recordings ---> MP4 segments
    |
    +---> Voiceover audio ---> faster-whisper ---> SRT
    |
    +---> All segments ---> FFmpeg composite
              |
              +---> sidechaincompress (duck music under voice)
              +---> Text overlays / subtitles
              +---> Final encode (H.264, CRF 18, 1080p60)
              |
              +---> YouTube upload (+ separate SRT file)
```

---

## Sources

- [MoviePy PyPI](https://pypi.org/project/moviepy/)
- [MoviePy GitHub](https://github.com/Zulko/moviepy)
- [MoviePy v2 Migration Guide](https://zulko.github.io/moviepy/getting_started/updating_to_v2.html)
- [MoviePy v1 vs v2 Performance Issue](https://github.com/Zulko/moviepy/issues/2395)
- [MovieLite (Numba-accelerated alternative)](https://github.com/francozanardi/movielite)
- [ffmpeg-python GitHub](https://github.com/kkroening/ffmpeg-python)
- [pyffmpeg PyPI](https://pypi.org/project/pyffmpeg/)
- [Remotion](https://www.remotion.dev/)
- [Remotion Python Integration](https://www.remotion.dev/docs/lambda/python)
- [Manim Community](https://www.manim.community/)
- [Manim GitHub](https://github.com/ManimCommunity/manim)
- [Generative Manim](https://generative-manim.vercel.app/)
- [Editly GitHub](https://github.com/mifi/editly)
- [Shotstack Pricing](https://shotstack.io/pricing/)
- [HeyGen API Pricing](https://www.heygen.com/api-pricing)
- [Synthesia Pricing](https://www.synthesia.io/pricing)
- [D-ID API Pricing](https://www.d-id.com/pricing/api/)
- [FFmpeg sidechaincompress docs](https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/sidechaincompress.html)
- [FFmpeg audio ducking mailing list](https://ffmpeg.org/pipermail/ffmpeg-user/2018-August/040933.html)
- [YouTube Recommended Upload Settings](https://support.google.com/youtube/answer/1722171?hl=en)
- [YouTube FFmpeg encoding gist](https://gist.github.com/mikoim/27e4e0dc64e384adbcb91ff10a2d3678)
- [Best YouTube Upload Settings 2026](https://www.zebgardner.com/photo-and-video-editing/2026-update-best-upload-settings-for-youtube)
- [OpenAI Whisper Subtitle Generation (DigitalOcean)](https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg)
- [faster-whisper SRT Generator](https://github.com/YounessMoustaouda/faster-whisper-generate-srt-subtitles)
- [HeyGen vs Synthesia 2026](https://wavespeed.ai/blog/posts/heygen-vs-synthesia-comparison-2026/)
