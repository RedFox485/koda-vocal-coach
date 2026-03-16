# Browser Video Capture Research

Research into programmatic video production using browser automation for creating production-grade demo videos.

---

## 1. Playwright Native Video Recording

**API**: `browser.new_context(record_video_dir="videos/", record_video_size={"width": 1920, "height": 1080})`

### What You Get
- **Codec**: VP8 in WebM container (Chromium). No codec choice exposed.
- **Bitrate**: Hardcoded at ~1 Mbit/s for Chromium. Not configurable.
- **Frame rate**: Not configurable via the API. Internally tied to browser rendering cadence.
- **Resolution**: Configurable via `record_video_size`. Defaults to viewport scaled down to fit 800x800, or 800x450 if no viewport set.
- **Quality controls**: None. There are open feature requests (issues #12056, #31424) but no timeline.

### Retrieving the Video
```python
context = browser.new_context(
    record_video_dir="videos/",
    record_video_size={"width": 1920, "height": 1080}
)
page = context.new_page()
# ... do things ...
context.close()  # Video is finalized on context close
video_path = page.video.path()
```

### Verdict
Adequate for test recordings. **Not suitable for production-grade demos** due to hardcoded 1 Mbit/s bitrate and no quality controls. Output looks noticeably compressed, especially with text-heavy UIs.

---

## 2. Screenshot Sequences + FFmpeg Composition

**Approach**: Take rapid `page.screenshot()` calls, pipe PNGs to ffmpeg.

### Performance
- Each `page.screenshot(type="png")` takes roughly 50-200ms depending on page complexity and viewport size.
- At best: ~5-20 fps from sequential screenshots. Not enough for smooth video of animations.
- PNG screenshots are lossless -- no compression artifacts on text.

### The Pipeline
```python
import subprocess, asyncio

async def capture_frames(page, duration_s=10, fps=30):
    """Capture screenshots and pipe to ffmpeg for video creation."""
    proc = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'image2pipe', '-framerate', str(fps),
        '-i', '-',  # stdin
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-preset', 'slow',
        'output.mp4'
    ], stdin=subprocess.PIPE)

    interval = 1.0 / fps
    for i in range(int(duration_s * fps)):
        screenshot = await page.screenshot(type="png")
        proc.stdin.write(screenshot)
        await asyncio.sleep(interval)

    proc.stdin.close()
    proc.wait()
```

### Verdict
Good for **static or slowly-changing content** (dashboards, form fills, text displays). Not viable for smooth animations at 30+ fps due to screenshot latency. However, combined with time virtualization (see section 3), this becomes the best approach.

---

## 3. CDP (Chrome DevTools Protocol) Capture

### Option A: Page.startScreencast

**API**: Sends frames as events from the browser's rendering pipeline.

```python
cdp = await page.context.new_cdp_session(page)

frames = []
def on_frame(params):
    frames.append(base64.b64decode(params['data']))
    cdp.send('Page.screencastFrameAck', {'sessionId': params['sessionId']})

cdp.on('Page.screencastFrame', on_frame)
await cdp.send('Page.startScreencast', {
    'format': 'png',       # or 'jpeg'
    'quality': 100,         # 0-100, jpeg only
    'maxWidth': 1920,
    'maxHeight': 1080,
    'everyNthFrame': 1      # capture every frame
})
```

**Characteristics**:
- Browser renders internally at 60fps but delivers frames at ~30fps max due to encoding overhead.
- Frame delivery is event-driven: you get every frame the rendering engine produces (could be 0fps on static pages, burst during animations).
- JPEG quality 80 = ~50-100KB per frame at 720p. PNG is lossless but larger.
- Must acknowledge each frame via `screencastFrameAck` or frames back up.

### Option B: HeadlessExperimental.beginFrame (Deterministic Rendering)

This is the gold standard for programmatic video. You control exactly when each frame renders.

```python
cdp = await page.context.new_cdp_session(page)

# Enable deterministic rendering
# Chrome must be launched with: --run-all-compositor-stages-before-draw
# and BeginFrameControl enabled

fps = 60
frame_interval_ms = 1000.0 / fps

for frame_num in range(total_frames):
    frame_time = frame_num * frame_interval_ms
    result = await cdp.send('HeadlessExperimental.beginFrame', {
        'frameTimeTicks': frame_time,
        'interval': frame_interval_ms,
        'noDisplayUpdates': False
    })
    # result contains the rendered frame
    # Capture via Page.captureScreenshot
    screenshot = await cdp.send('Page.captureScreenshot', {
        'format': 'png'
    })
```

**Key insight**: This decouples rendering from wall-clock time. The browser thinks 16.67ms passed between frames regardless of how long capture actually takes. Animations play at exactly the right speed. Every frame is perfect.

**Requirements**:
- Chromium only (not Firefox or WebKit)
- Must launch with `--run-all-compositor-stages-before-draw` flag
- Need to inject time virtualization shim (replace `Date.now`, `performance.now`, `requestAnimationFrame`, `setTimeout`, `setInterval`) so JavaScript also sees fake time

### Verdict
**CDP beginFrame is the best approach for production-grade video.** It produces deterministic, frame-perfect output at any resolution and frame rate, regardless of machine speed.

---

## 4. Existing Tools and Frameworks

### WebVideoCreator (Best for this use case)
- **Repo**: https://github.com/Vinlic/WebVideoCreator
- Node.js + Puppeteer + Chrome + FFmpeg
- Pioneered time virtualization + BeginFrame capture
- Supports: CSS3/SVG/Lottie/GIF animations, transition compositing, audio synthesis
- A 5-minute video renders in ~1 minute
- **Limitation**: Node.js only (not Python). Could be called as subprocess.

### Remotion
- **Site**: https://www.remotion.dev
- React-based programmatic video framework
- Each frame is a React component render
- Built-in rendering pipeline: React -> screenshots -> ffmpeg -> MP4
- Great for building video content from scratch, less ideal for recording existing web apps
- Has `@remotion/renderer` for server-side rendering

### puppeteer-screen-recorder
- **npm**: `puppeteer-screen-recorder`
- Uses CDP for frame-by-frame capture
- Configurable: fps (default 25), codec (libx264), bitrate (default 1000kbps)
- Supports headless and headful modes

### Playwright-Screen-Recorder (Port)
- **Repo**: https://github.com/raymelon/playwright-screen-recorder
- Port of puppeteer-screen-recorder for Playwright (Node.js)

### Replit's Video Rendering Engine
- Uses time virtualization + BeginFrame (inspired by WebVideoCreator)
- Patches setTimeout, setInterval, requestAnimationFrame, Date, performance.now()
- Planning to open-source their implementation

---

## 5. Resolution and Scaling

### Achieving 4K with Crisp Text

```python
context = browser.new_context(
    viewport={'width': 3840, 'height': 2160},  # 4K viewport
    device_scale_factor=2,                       # 2x DPI (retina)
    # Video will be captured at 3840x2160
    record_video_dir="videos/",
    record_video_size={'width': 3840, 'height': 2160}
)
```

**Key settings**:
- `viewport`: Sets the CSS pixel dimensions the page renders at
- `device_scale_factor`: Multiplies the pixel density. At 2x with a 1920x1080 viewport, the actual pixel buffer is 3840x2160.
- For screenshots: `page.screenshot()` automatically captures at full device pixel resolution
- For video: `record_video_size` should match the actual pixel dimensions

**Practical recommendation for YouTube**:
- Render at 1920x1080 viewport with `device_scale_factor=2` -> 3840x2160 actual pixels
- This gives you 4K output with retina-quality text
- Encode at 4K for YouTube even if content is designed for 1080p -- YouTube allocates 2.5x more bitrate to 4K uploads, so your 1080p-designed content will look sharper after YouTube re-encodes

**Browser compatibility**:
- Chromium: Full `device_scale_factor` support
- Firefox: `device_scale_factor` is IGNORED (known bug, issue #36628)
- Use Chromium for all video capture work

---

## 6. Audio Capture

### Current State
**Playwright cannot capture audio.** There is no API for it. Feature requests exist (issues #4870, #16526) but are not implemented.

### Workarounds

**Option A: Separate audio file (recommended)**
- Pre-record or synthesize audio tracks separately
- Compose with video in post-production via ffmpeg
- Gives you complete control over audio quality and timing

**Option B: Web Audio API extraction**
- Inject JavaScript to capture Web Audio API output via `AudioWorkletNode` or `ScriptProcessorNode`
- Write audio data to a buffer, extract via `page.evaluate()`
- Complex but possible for capturing synthesized audio from the page

**Option C: System audio loopback**
- Use virtual audio device (e.g., BlackHole on macOS, PulseAudio loopback on Linux)
- Record system audio in parallel with video capture
- Sync issues likely; not deterministic

**Option D: Browserless.io Screencast API**
- Third-party service that supports combined video + audio capture
- Outputs WebM with audio
- Not free, but handles the hard parts

**Recommendation**: Option A. For a demo video, compose audio separately. You get better quality, easier editing, and deterministic sync.

---

## 7. Timing Control

### Playwright Clock API (Fake Timers)
```python
# Pause time at a specific point
await page.clock.pause_at("2026-03-16T12:00:00")

# Manually advance time
await page.clock.fast_forward("00:00:30")  # Jump 30 seconds

# Tick precisely (fires all timers/animation frames in between)
await page.clock.run_for(1000)  # Advance 1000ms, firing everything
```

### Wait for Animations
```python
# Wait for network idle (all resources loaded)
await page.wait_for_load_state('networkidle')

# Wait for specific element to be stable (no bounding box changes for 2 animation frames)
await page.locator('.my-element').wait_for(state='visible')

# Wait for CSS animation/transition to complete
await page.wait_for_function('''
    () => {
        const el = document.querySelector('.animated-element');
        return getComputedStyle(el).animationPlayState === 'paused'
            || getComputedStyle(el).animationName === 'none';
    }
''')

# Simple sleep between actions for pacing
await page.wait_for_timeout(2000)  # 2 second pause
```

### Time Virtualization (for deterministic capture)
Inject a shim that replaces all time-dependent APIs:
```javascript
// Injected via page.add_init_script()
const virtualTime = { now: 0 };

window.Date.now = () => virtualTime.now;
window.performance.now = () => virtualTime.now;

const origRAF = window.requestAnimationFrame;
const rafCallbacks = [];
window.requestAnimationFrame = (cb) => { rafCallbacks.push(cb); return rafCallbacks.length; };

// Called from Python between frame captures
window.__advanceFrame = (deltaMs) => {
    virtualTime.now += deltaMs;
    const cbs = rafCallbacks.splice(0);
    cbs.forEach(cb => cb(virtualTime.now));
};
```

Then from Python:
```python
await page.add_init_script(path="time_shim.js")
# ... navigate to page ...

for frame in range(total_frames):
    await page.evaluate(f"window.__advanceFrame({frame_interval_ms})")
    screenshot = await page.screenshot(type="png")
    # write to ffmpeg pipe
```

This gives you perfect frame-by-frame control. Animations play at exactly the designed speed regardless of capture performance.

---

## 8. FFmpeg Composition Commands

### Overlay Audio at Specific Timestamp
```bash
# Add audio starting at 5 seconds into the video
ffmpeg -i video.mp4 -i narration.wav \
  -filter_complex "[1:a]adelay=5000|5000[delayed]; [0:a][delayed]amix=inputs=2:duration=first" \
  -c:v copy -c:a aac -b:a 192k output.mp4

# Multiple audio tracks at different timestamps
ffmpeg -i video.mp4 -i narration.wav -i sfx.wav \
  -filter_complex \
  "[1:a]adelay=2000|2000[narr]; \
   [2:a]adelay=8000|8000[sfx]; \
   [narr][sfx]amix=inputs=2[audio]" \
  -map 0:v -map "[audio]" -c:v copy -c:a aac output.mp4
```

### Text Overlays
```bash
# Centered text with fade in/out
ffmpeg -i input.mp4 -vf \
  "drawtext=text='Koda Vocal Coach':fontsize=64:fontcolor=white:\
  x=(w-text_w)/2:y=(h-text_h)/2:\
  alpha='if(lt(t,1),t,if(lt(t,4),1,if(lt(t,5),(5-t),0)))'" \
  -c:a copy output.mp4

# Text appearing at specific time (3s-7s)
ffmpeg -i input.mp4 -vf \
  "drawtext=text='Real-time Analysis':fontsize=48:fontcolor=white:\
  x=100:y=50:enable='between(t,3,7)':\
  alpha='if(lt(t-3,0.5),(t-3)*2,if(lt(7-t,0.5),(7-t)*2,1))'" \
  -c:a copy output.mp4
```

### Concatenating Clips
```bash
# Method 1: Concat demuxer (same codec, fast, no re-encode)
# Create file list: clips.txt
# file 'clip1.mp4'
# file 'clip2.mp4'
# file 'clip3.mp4'
ffmpeg -f concat -safe 0 -i clips.txt -c copy output.mp4

# Method 2: Concat filter (different codecs/resolutions, re-encodes)
ffmpeg -i clip1.mp4 -i clip2.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]" \
  -map "[outv]" -map "[outa]" output.mp4
```

### Crossfade Transitions
```bash
# 1-second crossfade between two clips
# offset = duration_of_first_clip - crossfade_duration
ffmpeg -i clip1.mp4 -i clip2.mp4 \
  -filter_complex \
  "xfade=transition=fade:duration=1:offset=9, \
   acrossfade=d=1" \
  output.mp4

# Available transitions: fade, wipeleft, wiperight, wipeup, wipedown,
# slideleft, slideright, slideup, slidedown, circlecrop, rectcrop,
# distance, fadeblack, fadewhite, radial, smoothleft, smoothright,
# smoothup, smoothdown, circleopen, circleclose, vertopen, vertclose,
# horzopen, horzclose, dissolve, pixelize, diagtl, diagtr, diagbl, diagbr
```

### YouTube-Optimized Encoding
```bash
# 1080p for YouTube (high quality)
ffmpeg -i input.mp4 \
  -c:v libx264 -preset slow -crf 18 \
  -b:v 10M -maxrate 15M -bufsize 20M \
  -vf "scale=1920:1080" \
  -c:a aac -b:a 192k -ar 48000 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  youtube_1080p.mp4

# 4K for YouTube (maximum quality)
ffmpeg -i input.mp4 \
  -c:v libx264 -preset slow -crf 16 \
  -b:v 40M -maxrate 60M -bufsize 80M \
  -vf "scale=3840:2160" \
  -c:a aac -b:a 384k -ar 48000 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  youtube_4k.mp4

# Key settings explained:
# -crf 18: Visually lossless quality (lower = better, 0 = lossless)
# -preset slow: Better compression (slower encode, smaller file, same quality)
# -movflags +faststart: Moves metadata to start of file for faster streaming
# -pix_fmt yuv420p: Required for compatibility
# -maxrate/-bufsize: Rate control to prevent bitrate spikes
```

---

## 9. MoviePy as Alternative to FFmpeg

### Overview
MoviePy is a Python library wrapping ffmpeg with a Pythonic API. Current version: 2.1.2 (Feb 2026).

### Pros
- Pure Python API -- no shell command construction
- Easy text overlays with `TextClip`
- Simple composition with `CompositeVideoClip`
- Good for complex timeline assembly
- Integrates naturally with a Python/Playwright pipeline

### Cons
- **Slower than raw ffmpeg** -- loads entire clips into memory as numpy arrays
- **Memory hungry** -- a 4K 60fps clip eats RAM fast
- **Maintenance concerns** -- limited maintainer bandwidth noted by the project
- **Font rendering**: Requires ImageMagick for text (extra dependency)

### Example Usage
```python
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

# Load clips
clip1 = VideoFileClip("scene1.mp4")
clip2 = VideoFileClip("scene2.mp4")

# Add text overlay
title = TextClip(
    text="Koda Vocal Coach",
    font_size=64, color='white', font='Arial-Bold',
    duration=3
).with_position('center').with_start(1)

# Compose
video = CompositeVideoClip([clip1, title])

# Concatenate with crossfade
final = concatenate_videoclips([
    clip1.with_effects([vfx.CrossFadeOut(1)]),
    clip2.with_effects([vfx.CrossFadeIn(1)])
], method="compose")

# Export for YouTube
final.write_videofile("output.mp4",
    codec='libx264',
    bitrate='10M',
    audio_codec='aac',
    audio_bitrate='192k',
    fps=60
)
```

### Verdict
**Use MoviePy for timeline assembly and text overlays** where the Python API matters. **Use raw ffmpeg for encoding, concatenation, and any operation on large files** where memory and speed matter. They complement each other.

---

## 10. Real Examples and Open Source Projects

### WebVideoCreator (Most relevant)
- https://github.com/Vinlic/WebVideoCreator
- Node.js framework that does exactly what we need: time virtualization + BeginFrame + ffmpeg
- Used by Replit for their video rendering engine
- Handles CSS animations, SVG, Lottie, APNG, WebP animations
- Supports transition compositing and audio synthesis

### Replit's Approach (Blog post)
- https://blog.replit.com/browsers-dont-want-to-be-cameras
- Details the full technique: fake clock injection, BeginFrame capture, video element workarounds
- Key quote: "We built a video rendering engine by lying to the browser about what time it is"
- Planning to open-source

### Remotion
- https://www.remotion.dev
- React-based: define video as React components, render frame-by-frame
- Not ideal for recording existing web apps, but excellent for building video content from scratch
- Could be used to build overlay/intro/outro sequences that get composited with captured footage

### headless-screen-recorder
- https://github.com/brianbaso/headless-screen-recorder
- Uses HeadlessExperimental.beginFrame for high-quality capture
- Based on puppeteer-screen-recorder

---

## Recommended Architecture for Koda Demo Video Pipeline

Based on this research, the optimal pipeline is:

```
[Playwright + CDP BeginFrame]  -->  [PNG frames]  -->  [ffmpeg]  -->  [Final MP4]
        |                                                   |
   Time virtualization                              Audio overlay
   Device scale factor 2x                           Text overlays
   1920x1080 viewport                               Transitions
   (3840x2160 actual pixels)                         YouTube encoding
```

### Capture Layer (Python + Playwright)
1. Launch Chromium with `--run-all-compositor-stages-before-draw`
2. Create context with `viewport=1920x1080`, `device_scale_factor=2`
3. Inject time virtualization shim via `page.add_init_script()`
4. Use CDP session for `HeadlessExperimental.beginFrame` at 60fps intervals
5. Capture each frame via `Page.captureScreenshot` as PNG
6. Pipe frames to ffmpeg subprocess

### Composition Layer (ffmpeg)
1. Encode captured frames: libx264, CRF 18, preset slow
2. Overlay narration audio at specific timestamps
3. Add text overlays for labels/titles via drawtext filter
4. Concatenate scenes with crossfade transitions
5. Final encode for YouTube: 4K, high bitrate, movflags +faststart

### Fallback (If BeginFrame is too complex)
Use Playwright's Clock API + sequential screenshots:
1. Pause page time with `page.clock.pause_at()`
2. Advance time in small increments with `page.clock.run_for()`
3. Take screenshot after each time advance
4. Compose with ffmpeg

This is simpler but slightly less deterministic for CSS animations that bypass JavaScript timers.

---

## Sources

- [Playwright Python Video Docs](https://playwright.dev/python/docs/videos)
- [Playwright Video Quality Issue #31424](https://github.com/microsoft/playwright/issues/31424)
- [Playwright Video Quality Issue #10855](https://github.com/microsoft/playwright/issues/10855)
- [Playwright Video Quality Issue #12056](https://github.com/microsoft/playwright/issues/12056)
- [Playwright Emulation Docs (device_scale_factor)](https://playwright.dev/python/docs/emulation)
- [Playwright Clock API](https://playwright.dev/python/docs/clock)
- [Playwright CDPSession Docs](https://playwright.dev/python/docs/api/class-cdpsession)
- [Playwright Audio Feature Request #4870](https://github.com/microsoft/playwright/issues/4870)
- [HeadlessExperimental.beginFrame - PyCDP Docs](https://py-cdp.readthedocs.io/en/latest/api/headless_experimental.html)
- [Chrome DevTools Protocol - Page Domain](https://chromedevtools.github.io/devtools-protocol/tot/Page/)
- [CDP FPS for startScreencast Issue #63](https://github.com/ChromeDevTools/devtools-protocol/issues/63)
- [WebVideoCreator (GitHub)](https://github.com/Vinlic/WebVideoCreator)
- [Replit: We Built a Video Rendering Engine by Lying to the Browser About What Time It Is](https://blog.replit.com/browsers-dont-want-to-be-cameras)
- [Remotion](https://www.remotion.dev/)
- [puppeteer-screen-recorder (npm)](https://www.npmjs.com/package/puppeteer-screen-recorder)
- [playwright-screen-recorder (GitHub)](https://github.com/raymelon/playwright-screen-recorder/)
- [headless-screen-recorder (GitHub)](https://github.com/brianbaso/headless-screen-recorder)
- [FFmpeg Filters Documentation](https://ffmpeg.org/ffmpeg-filters.html)
- [FFmpeg xfade Crossfade Guide](https://ottverse.com/crossfade-between-videos-ffmpeg-xfade-filter/)
- [FFmpeg drawtext Filter Guide](https://ottverse.com/ffmpeg-drawtext-filter-dynamic-overlays-timecode-scrolling-text-credits/)
- [FFmpeg drawtext Fade In/Out Example](https://ffmpegbyexample.com/examples/50gowmkq/fade_in_and_out_text_using_the_drawtext_filter/)
- [YouTube Recommended Encoding Settings](https://support.google.com/youtube/answer/1722171?hl=en)
- [FFmpeg YouTube Encoding Settings (Gist)](https://gist.github.com/mikoim/27e4e0dc64e384adbcb91ff10a2d3678)
- [MoviePy (GitHub)](https://github.com/Zulko/moviepy)
- [MoviePy Documentation](https://zulko.github.io/moviepy/getting_started/quick_presentation.html)
- [Record Video with FFmpeg + Playwright (Gist)](https://gist.github.com/jkosoy/3e448bd82a36181cac600e42bd59bfd5)
