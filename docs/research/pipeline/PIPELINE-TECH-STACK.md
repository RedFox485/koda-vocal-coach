# Demo Video Pipeline — Finalized Tech Stack

Based on research across 5 domains. This is the stack we're building.

## Core Stack

| Layer | Tool | Why |
|-------|------|-----|
| **UI Capture** | Playwright + CDP `HeadlessExperimental.beginFrame` | Time virtualization = perfect frames, zero drops, any resolution |
| **Voiceover** | ElevenLabs API (cloned voice) | Best quality, Python SDK, instant cloning from short sample |
| **Voiceover prototyping** | Azure TTS (SSML) | Free 5M chars/month, best emotion/pacing controls via SSML |
| **Audio injection** | Custom WebSocket proxy script | Feeds WAV through WS as if live mic — backend can't tell the difference |
| **Composition** | FFmpeg via subprocess | Direct control, no wrapper overhead, native audio ducking |
| **Architecture animation** | Manim (optional) | Animated diagram > static PNG for Shot 7 |
| **Subtitles** | faster-whisper | 4-8x faster than Whisper, outputs SRT directly |
| **QA** | Claude vision on extracted keyframes | Verify gauge state, text readability, timing |

## Key Techniques

1. **Time virtualization** — Replace `Date.now`, `performance.now`, `requestAnimationFrame` with fakes. Advance clock manually between frame captures. Browser renders perfectly regardless of capture speed.

2. **4K trick** — Viewport 1920x1080 + `device_scale_factor=2` = 3840x2160 actual capture. YouTube allocates 2.5x more bitrate to 4K uploads even when viewed at 1080p.

3. **Audio ducking** — FFmpeg `sidechaincompress` filter: voiceover as sidechain, singing ducks to 40% automatically.

4. **In-browser text overlays** — Inject overlay elements via `page.evaluate()` during capture. Text renders with the page aesthetic, no post-processing needed.

5. **Separate coaching audio** — Generate Gemini coaching cues independently, overlay at exact timestamps. Guarantees perfect timing and content relevance.

## YouTube Upload Settings

```bash
ffmpeg -i composed.mp4 \
  -c:v libx264 -preset slow -crf 18 -profile:v high \
  -pix_fmt yuv420p -r 60 \
  -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart \
  final_youtube.mp4
```

## Dependencies to Install

```bash
pip install playwright elevenlabs ffmpeg-python soundfile numpy websockets
playwright install chromium

# Optional
pip install manim faster-whisper moviepy
```
