# Smooth Browser Animation Recording Research

**Date:** 2026-03-16
**Purpose:** Determine the optimal technique for recording web app UIs as production-grade demo videos with perfectly smooth animation — zero stutter, zero dropped frames.

---

## Table of Contents

1. [Why Screen Recordings Stutter](#1-why-screen-recordings-stutter)
2. [Playwright Screenshot Burst Rate](#2-playwright-screenshot-burst-rate)
3. [CDP Page.screencastFrame](#3-cdp-pagescreencastframe)
4. [requestAnimationFrame Synchronization](#4-requestanimationframe-synchronization)
5. [Headless vs Headed Chrome](#5-headless-vs-headed-chrome)
6. [CSS Animation Recording](#6-css-animation-recording)
7. [Web Animations API](#7-web-animations-api)
8. [Device Pixel Ratio and Scaling](#8-device-pixel-ratio-and-scaling)
9. [Canvas Recording](#9-canvas-recording)
10. [Optimal Approach for Guaranteed Smooth Output](#10-optimal-approach-for-guaranteed-smooth-output)
11. [Frame Timing Math](#11-frame-timing-math)
12. [Comparison: Playwright vs Puppeteer vs Selenium](#12-comparison-playwright-vs-puppeteer-vs-selenium)
13. [Existing Tools and Prior Art](#13-existing-tools-and-prior-art)
14. [Recommended Architecture](#14-recommended-architecture)

---

## 1. Why Screen Recordings Stutter

### Root Cause Analysis

Screen recordings stutter because of a **fundamental timing mismatch** between the recorder and the browser's rendering pipeline:

- **Frame timing inconsistency:** The browser renders frames when it can, targeting 16.67ms intervals (60fps). A screen recorder samples the screen on its own schedule. When these clocks drift, you get duplicate frames (stutter) or missed frames (jank).

- **Dropped frames under load:** When the screenshot/capture operation itself takes time (50-200ms+), the browser continues animating in real time. By the time the next capture happens, multiple animation frames have been skipped. The video shows the animation "jumping" between states.

- **The core problem (from Replit's research):** "Browsers are real-time systems that render frames when they can, skip frames under load, and tie animations to wall-clock time. If a screenshot takes 200ms but an animation expects 16ms frames, you get a stuttery, unwatchable mess."

- **Compositor lag:** The GPU compositor may batch or delay frame presentation. V-sync introduces additional timing constraints. Screen recorders reading from the framebuffer may get partially-composed frames.

- **Encoding overhead:** Real-time encoders (H.264, VP9) compete for CPU/GPU resources with the browser's rendering, causing both to drop frames.

### What Happens When Frames Drop

When a screen recorder drops frames, the resulting video shows:
- **Temporal aliasing:** Smooth 60fps motion becomes 15-20fps jerky motion
- **Non-uniform frame spacing:** Some frames hold for 2-3x their expected duration, others are missing entirely
- **Animation state jumping:** CSS transitions appear to "teleport" between keyframes

---

## 2. Playwright Screenshot Burst Rate

### Realistic Performance Numbers

| Scenario | Latency per Screenshot | Effective FPS |
|----------|----------------------|---------------|
| Small viewport (800x600), JPEG, optimizeForSpeed | ~50-80ms | ~12-20 fps |
| Standard viewport (1280x720), PNG | ~100-200ms | ~5-10 fps |
| Large viewport (1920x1080), PNG | ~200-500ms | ~2-5 fps |
| 2x DPR (3840x2160 actual pixels), PNG | ~500ms+ | ~2 fps |
| WebGL/Canvas heavy page | Variable, often worse | Variable |

### Bottlenecks (in order of impact)

1. **Screenshot encoding:** PNG compression is the biggest bottleneck. The `optimizeForSpeed` CDP parameter helps significantly (~2x speedup) by using zlib q1 (RLE encoding) instead of full compression.
2. **GPU readback:** Reading the framebuffer from GPU memory has considerable latency. Unlike normal rendering where Chrome can produce frames incrementally, screenshots require reading the entire viewport at once.
3. **WebSocket transfer:** Screenshots are not saved directly to disk by Chrome -- they're base64-encoded and streamed over the WebSocket to Node.js, then decoded and saved. This adds overhead proportional to image size.
4. **Redundant CDP calls:** Each `Page.captureScreenshot` triggers up to 4 WebSocket calls before the actual capture: `Target.activateTarget`, `Page.getLayoutMetrics`, `Emulation.setDeviceMetricsOverride`, `Emulation.setDefaultBackgroundColorOverride`.

### Key Finding: 30fps is NOT achievable in real-time

At best, with JPEG format, small viewport, and `optimizeForSpeed`, you might sustain ~20fps. For production 1080p PNG captures, expect 2-10fps real-time. **This means real-time screenshot burst cannot produce smooth video. You MUST decouple capture time from video time.**

### Optimization: Burst Mode

Puppeteer implemented a burst mode (Issue #3502) that skips redundant WebSocket calls after the first screenshot, achieving ~2.5x speedup. However, even with burst mode, real-time 30fps is unrealistic for anything above a small viewport.

---

## 3. CDP Page.screencastFrame

### How It Works

`Page.startScreencast` streams frames as JPEG images via the `Page.screencastFrame` event. Each frame must be acknowledged before the next is sent.

### Parameters

- **format:** "jpeg" or "png" (jpeg recommended for speed)
- **quality:** 0-100 (80 is a good balance, ~50-100KB per frame at 720p)
- **maxWidth / maxHeight:** Maximum resolution constraints
- **everyNthFrame:** Skip frames (e.g., 2 = every other frame)

### Performance Characteristics

- Frame rate is limited by acknowledgment round-trip time and encoding speed
- JPEG quality 80 at 1280x720 produces ~50-100KB per frame
- Frames arrive asynchronously and can arrive out of order (known bug: Issue #117)
- Causes significant performance overhead on the page being captured (Chromium Issue #40934921: "Page screencasting causes huge performance regression")

### Verdict

Screencast is designed for live preview/debugging, not production video. The frame rate is inconsistent, frames can arrive out of order, and it degrades page performance. **Not suitable for production demo videos.**

---

## 4. requestAnimationFrame Synchronization

### The Problem

You cannot simply hook into rAF from outside the page to guarantee frame-perfect capture. The rAF callback runs inside the browser's rendering loop, and by the time you receive notification and take a screenshot, the browser may have already advanced to the next frame.

### What Works

**Injecting a time shim** that replaces `requestAnimationFrame` with a controlled version:

```javascript
// Conceptual shim (simplified from Replit's ~1,200 line version)
let virtualTime = 0;
const frameInterval = 1000 / 30; // 30fps

window.requestAnimationFrame = (callback) => {
  // Don't call immediately -- wait for external "advance" signal
  pendingRAFCallbacks.push(callback);
};

function advanceFrame() {
  virtualTime += frameInterval;
  const callbacks = pendingRAFCallbacks.splice(0);
  callbacks.forEach(cb => cb(virtualTime));
}
```

### Interaction with CSS vs JS Animations

- **JS animations using rAF:** Fully controllable via the shim -- they only advance when you call `advanceFrame()`
- **CSS animations:** NOT controlled by rAF shim. CSS animations use the browser's internal clock. You need either the Web Animations API (see section 7) or CDP virtual time to control these.
- **CSS transitions:** Same limitation as CSS animations -- they follow wall-clock time unless intercepted at the browser engine level.

---

## 5. Headless vs Headed Chrome

### Key Differences

| Aspect | Headless | Headed |
|--------|----------|--------|
| Renderer | SwiftRender (software) | GPU-accelerated |
| WebGL FPS | ~8fps without `--use-angle=gl` | 60fps with GPU |
| Compositor | Simplified, controllable via BeginFrame | Full desktop compositor |
| Font rendering | May differ from headed | System fonts |
| Canvas rendering | Software rasterization by default | GPU rasterization |
| Screenshot consistency | Deterministic (no OS compositor variance) | Varies by display, OS |

### Headless Advantages for Recording

- **No compositor interference:** Headless doesn't fight with the OS window manager
- **BeginFrameControl:** Only available in headless mode. Gives you deterministic control over Chrome's rendering pipeline via `HeadlessExperimental.beginFrame`
- **Resolution freedom:** No monitor resolution limits -- can render at any size
- **Reproducibility:** Same input always produces same output

### Headless Disadvantages

- **Software rendering by default:** Without `--use-angle=gl`, GPU-dependent content (WebGL, heavy Canvas) renders slowly
- **Visual differences:** Font rendering, anti-aliasing, and subpixel rendering differ from headed mode
- **Canvas/WebGL:** May need `--use-angle=gl` flag to enable hardware acceleration

### Recommendation

Use headless mode with `--use-angle=gl` for GPU acceleration. The deterministic rendering control far outweighs the visual differences.

---

## 6. CSS Animation Recording

### Do CSS Animations Run Correctly in Headless?

CSS animations in headless Chrome run at the **correct speed relative to wall-clock time** by default. They do NOT automatically synchronize with your capture pipeline. This means:

- A 2-second CSS transition takes 2 real seconds regardless of how fast/slow you capture
- If your capture is slower than real-time, you'll miss intermediate animation states
- CSS animations tied to `animation-delay`, `animation-duration`, and `transition-duration` all follow the internal clock

### Time Virtualization Options for CSS Animations

1. **CDP `Emulation.setVirtualTimePolicy`:** Makes Chrome's internal clock virtual. CSS animations advance only when virtual time advances. Policies:
   - `advance`: Virtual time base fast-forwards when idle
   - `pause`: Virtual time stops entirely
   - `pauseIfNetworkFetchesPending`: Pauses during network fetches

2. **`--virtual-time-budget` flag:** CLI flag that fakes the internal clock. Chrome behaves as if N seconds have passed without actually waiting.

3. **`HeadlessExperimental.beginFrame` with `--run-all-compositor-stages-before-draw`:** Replaces Chrome's rendering loop entirely. CSS animations advance by the `interval` parameter (default 16.666ms) per BeginFrame call.

4. **JavaScript override (limited):** Can override `Date.now()` and `performance.now()` but CSS animations don't use JS timing -- they use the compositor's internal clock. JS shims alone won't fix CSS animation timing.

### Critical Insight

**CSS animations require browser-level time control (CDP virtual time or BeginFrame), not just JavaScript shims.** This is a key limitation of tools like timecut that only override JS timing functions.

---

## 7. Web Animations API

### Frame-by-Frame Control

The Web Animations API provides programmatic control over all animations on a page:

```javascript
// Get all running animations (CSS + JS-driven)
const animations = document.getAnimations();

// Pause everything
animations.forEach(anim => anim.pause());

// Step to a specific time (in milliseconds)
animations.forEach(anim => {
  anim.currentTime = frameNumber * (1000 / 30); // 30fps stepping
});

// Resume
animations.forEach(anim => anim.play());
```

### What It Controls

- CSS animations declared with `@keyframes`
- CSS transitions
- Web Animations API animations created via `element.animate()`
- Does NOT control: Canvas animations, requestAnimationFrame-driven animations, or any non-CSS visual changes

### Limitations for Recording

- Only controls animations the browser knows about (CSS/WAAPI)
- Canvas-based animations (our strain gauge, pitch tuner) are invisible to this API
- Would need to be combined with rAF shim for JS-driven animations
- Setting `currentTime` may cause visual snapping rather than smooth interpolation if the animation uses easing

### Verdict

Useful as a supplement but not a complete solution. Good for ensuring CSS animations are at the right state when a screenshot is taken, but doesn't help with Canvas or JS animations.

---

## 8. Device Pixel Ratio and Scaling

### How 2x DPR Works in Chrome

When you set `deviceScaleFactor: 2` (via CDP or Playwright), Chrome:
1. Reports `window.devicePixelRatio` as 2 to the page
2. Renders the page at 2x the CSS pixel dimensions in actual pixels
3. All vector content (text, SVG, CSS) renders natively at 2x -- **genuinely sharper, not upscaled**
4. Screenshots capture the full 2x resolution

### Impact on Different Content Types

| Content Type | 2x DPR Effect | Result |
|-------------|---------------|--------|
| Text | Renders at 2x resolution | Genuinely sharper |
| SVG | Renders at 2x resolution | Genuinely sharper |
| CSS borders/gradients | Renders at 2x resolution | Genuinely sharper |
| Raster images | Upscaled unless @2x srcset provided | Blurry |
| **Canvas** | **Depends on implementation** | **See below** |

### Canvas and DPR -- Critical for Our App

Canvas elements do NOT automatically benefit from 2x DPR. The canvas must explicitly handle it:

```javascript
// Correct way to handle DPR in Canvas
const dpr = window.devicePixelRatio || 1;
canvas.width = cssWidth * dpr;
canvas.height = cssHeight * dpr;
canvas.style.width = cssWidth + 'px';
canvas.style.height = cssHeight + 'px';
ctx.scale(dpr, dpr);
```

If our strain gauge and pitch tuner Canvas elements don't do this, they'll render at 1x and get upscaled to 2x (blurry). **We need to verify our Canvas code handles DPR correctly.**

### Recommendation for Recording

- Use `deviceScaleFactor: 2` on a 1920x1080 viewport = 3840x2160 actual pixels
- This gives genuinely sharper text and UI elements
- Ensure Canvas elements are DPR-aware
- Downscale to 1080p in FFmpeg for the final video if needed (supersampling = free anti-aliasing)

---

## 9. Canvas Recording

### Canvas Animation in Headless Playwright

Canvas animations that use `requestAnimationFrame` behave the same as any other rAF-driven animation in headless Chrome:
- They run at whatever frame rate Chrome's software renderer can achieve
- Without GPU acceleration, complex Canvas rendering may drop below 60fps
- With `--use-angle=gl`, Canvas gets GPU acceleration even in headless mode

### Gotchas

1. **requestAnimationFrame timing:** In headless mode, rAF fires based on Chrome's internal vsync simulation. If the page is backgrounded or Chrome is under load, rAF timing becomes irregular.

2. **Canvas capture mode (timecut):** Some tools can directly copy Canvas data via `canvas.toDataURL()` or `canvas.toBlob()`, which is often faster than full-page screenshots for Canvas-heavy pages.

3. **WebGL in headless:** Without `--use-angle=gl`, WebGL contexts may fail or fall back to software rendering at ~8fps. Always enable this flag.

4. **OffscreenCanvas:** If using OffscreenCanvas with Web Workers, the worker's rAF operates independently. Time shims injected into the main thread won't affect worker timing.

5. **Canvas state is not in the DOM:** Unlike CSS animations, Canvas state is opaque to the browser. You can't inspect or control Canvas animations through the Web Animations API. The only way to control Canvas animation timing is through the rAF/time shim approach.

### Recommendation for Our App

Since our strain gauge and pitch tuner are Canvas-based:
- They MUST be controlled via rAF/time shim (not Web Animations API)
- Ensure they use delta-time patterns (calculating movement from elapsed time) rather than fixed-step patterns
- If they read `Date.now()` or `performance.now()` directly, those must be shimmed too
- Test with `--use-angle=gl` to ensure GPU acceleration is available

---

## 10. Optimal Approach for Guaranteed Smooth Output

### The Answer: Time Virtualization + Deterministic Frame Capture

After analyzing all approaches, the industry-proven technique is:

**Decouple capture time from video time by virtualizing the browser's clock, then capture one frame at a time, at whatever speed Chrome can render.**

This is the approach used by:
- **Remotion** (React video framework, production-grade)
- **WebVideoCreator** (pioneered time virtualization + BeginFrame)
- **Replit's video renderer** (~1,200 line time shim)
- **puppeteer-capture** (BeginFrame + deterministic mode)
- **timecut** (JS time override + screenshot capture)

### Why This Works

A 60fps animation that takes 500ms per frame to actually render will still produce a **butter-smooth 60fps video** because:
1. The page thinks only 16.67ms passed between frames (1000/60)
2. The actual wall-clock time per frame is irrelevant
3. Every frame is captured at exactly the right animation state
4. No frames are ever dropped because time doesn't advance until you say so

### Two Sub-Approaches

#### Approach A: JavaScript Time Shim + Screenshot Capture

**How it works:**
- Inject ~1,200 lines of JavaScript that replaces `setTimeout`, `setInterval`, `requestAnimationFrame`, `Date`, `Date.now()`, and `performance.now()` with a fake clock
- For each frame: advance virtual time by `1000/fps` ms, trigger all pending callbacks, take a screenshot
- Pipe screenshots to FFmpeg

**Pros:**
- Works with Playwright (no Puppeteer dependency)
- Controls JS-driven animations perfectly
- Controls Canvas animations (if they use rAF/Date/performance.now)

**Cons:**
- Does NOT control CSS animations (they use browser internal clock, not JS APIs)
- Requires the shim to be injected before any page JavaScript runs
- Complex to get right (Replit's shim is 1,200 lines for a reason)

#### Approach B: CDP BeginFrame + Deterministic Mode (RECOMMENDED)

**How it works:**
- Launch Chrome with `--deterministic-mode` and `--enable-begin-frame-control`
- Use `HeadlessExperimental.beginFrame` to manually trigger each render cycle
- Chrome advances ALL timing (JS, CSS, compositor) by the specified interval
- Each BeginFrame can optionally return a screenshot

**Pros:**
- Controls EVERYTHING: JS timing, CSS animations, CSS transitions, compositor animations
- No JavaScript injection needed (works at browser engine level)
- Frame-perfect: one BeginFrame = one complete layout-paint-composite cycle
- Can return screenshot directly from BeginFrame (no separate capture call)

**Cons:**
- Only works in headless Chrome
- Requires specific Chrome flags
- HeadlessExperimental domain may change between Chrome versions
- Puppeteer has better CDP integration than Playwright for this

### Recommended Hybrid Approach

For maximum reliability, combine both:

1. **Use BeginFrame for compositor control** (handles CSS animations, compositor timing)
2. **Inject a minimal time shim** as insurance (catches any edge cases where JS reads time directly)
3. **Capture each frame via BeginFrame's built-in screenshot** or `Page.captureScreenshot` with `optimizeForSpeed: true`
4. **Pipe frames to FFmpeg** for final video encoding

---

## 11. Frame Timing Math

### The Problem

If we want 30fps output and each screenshot takes ~50ms, real-time capture gives us only 20fps. But this is the **wrong way to think about it.**

### The Solution: Virtual Time Decoupling

With time virtualization, the math becomes:

```
Target: 30fps video
Virtual time per frame: 1000ms / 30fps = 33.33ms

For each frame:
  1. Advance virtual time by 33.33ms
  2. Wait for all animations/callbacks to settle
  3. Take screenshot (takes 50-200ms real time -- doesn't matter)
  4. Move to next frame

Real time per frame: ~50-200ms (screenshot) + ~10-50ms (rendering)
Real time for 30 seconds of video at 30fps:
  900 frames x ~100ms average = ~90 seconds

Result: 30 seconds of perfectly smooth 30fps video
         rendered in ~90 seconds of wall-clock time
```

### Key Insight

**The video doesn't know or care how long each frame took to capture.** Each frame is stamped at exactly 33.33ms intervals in the output, regardless of actual capture time. The video player plays them back at 30fps and they look perfectly smooth.

### FFmpeg Command for Frame Sequence

```bash
ffmpeg -framerate 30 -i frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  output.mp4
```

Parameters:
- `-framerate 30`: Input frame rate (must match your capture rate)
- `-crf 18`: High quality (lower = better, 0 = lossless, 18 = visually lossless)
- `-pix_fmt yuv420p`: Wide player compatibility
- `-movflags +faststart`: Enables streaming playback

### Alternative: Direct Pipe to FFmpeg

Instead of saving individual frames to disk:

```bash
# Pipe raw frames directly to FFmpeg (faster, no disk I/O)
ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 30 -i pipe:0 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p output.mp4
```

Or with PNG pipe:

```bash
ffmpeg -framerate 30 -f image2pipe -i pipe:0 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p output.mp4
```

---

## 12. Comparison: Playwright vs Puppeteer vs Selenium

### For Smooth Video Capture Specifically

| Feature | Playwright | Puppeteer | Selenium |
|---------|-----------|-----------|----------|
| CDP access | Via CDPSession (indirect) | Native (direct) | Via CDP bridge (limited) |
| BeginFrame support | Possible via CDPSession | Native | Not practical |
| Built-in video | Yes (real-time only, VP8) | No | No |
| Screenshot speed | Same as Puppeteer (both use CDP) | Same as Playwright | Slower |
| Deterministic mode | Must configure manually via CDP | Better documented | Not supported |
| Multi-browser | Chromium, Firefox, WebKit | Chromium only | All browsers |
| Time shim injection | `page.addInitScript()` | `page.evaluateOnNewDocument()` | Possible but harder |

### Verdict: Puppeteer Wins for This Use Case

**Puppeteer is the better choice for deterministic video capture** because:

1. **Direct CDP integration:** Puppeteer talks CDP natively. Playwright wraps CDP in its own protocol layer, adding overhead and abstraction.
2. **BeginFrame support:** `puppeteer-capture` already implements the full BeginFrame pipeline. No equivalent exists for Playwright.
3. **Deterministic mode:** Better documented and tested with Puppeteer.
4. **Existing ecosystem:** `puppeteer-capture`, `timecut`, `WebVideoCreator` all use Puppeteer.

**However, Playwright can still work** via CDPSession if you prefer its API or already have a Playwright-based pipeline. The CDP commands are the same underneath.

---

## 13. Existing Tools and Prior Art

### puppeteer-capture (RECOMMENDED)

- **URL:** https://github.com/alexey-pelykh/puppeteer-capture
- **Approach:** CDP BeginFrame + deterministic mode
- **Key feature:** Full control over rendering pipeline. Frames captured on demand, not in real time.
- **Usage:** `npm install puppeteer-capture`
- **Chrome flags:** `--deterministic-mode --enable-begin-frame-control`

### WebVideoCreator (WVC)

- **URL:** https://github.com/Vinlic/WebVideoCreator
- **Approach:** Time virtualization + BeginFrame (pioneered the technique)
- **Key feature:** Handles CSS animations, SVG, Lottie, GIF, APNG, Canvas rAF animations
- **Built on:** Node.js + Puppeteer + Chrome + FFmpeg

### Remotion

- **URL:** https://remotion.dev
- **Approach:** React components rendered frame-by-frame via `useCurrentFrame()`
- **Key feature:** Parallel rendering across multiple browser tabs. Production-grade.
- **Limitation:** Requires building your content as React components (not arbitrary web pages)

### Replit Video Renderer

- **URL:** https://blog.replit.com/browsers-dont-want-to-be-cameras
- **Approach:** ~1,200 line JS time shim + CDP virtual time
- **Key feature:** Works on arbitrary web pages (not just controlled React components)
- **Insight:** Combined JS shim + browser-level virtual time for full coverage

### timecut

- **URL:** https://github.com/tungs/timecut
- **Approach:** JS time override + Puppeteer screenshots + FFmpeg
- **Limitation:** Only overrides JS functions. **CSS animations are NOT controlled.** Pages where changes occur via CSS rules alone will not render as intended.
- **Canvas mode:** Experimental direct canvas capture via `toDataURL()`, often faster than screenshots

### timesnap

- **URL:** https://github.com/tungs/timesnap
- **Approach:** Same as timecut but outputs image sequences instead of video
- **Useful for:** Custom FFmpeg encoding pipelines

---

## 14. Recommended Architecture

### For Our Koda Demo Video Pipeline

Given that our app uses:
- CSS animations (UI transitions, layout effects)
- Canvas animations (strain gauge, pitch tuner)
- Likely requestAnimationFrame-driven rendering

The recommended architecture is:

```
                    +-------------------+
                    |   Orchestrator    |
                    | (Node.js script)  |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   Puppeteer       |
                    | --deterministic   |
                    | --begin-frame-ctl |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    | Time Shim (JS)    |         | CDP BeginFrame    |
    | - rAF override    |         | - Compositor ctrl |
    | - Date/perf.now   |         | - CSS animation   |
    | - setTimeout/Int  |         | - Frame capture   |
    +-------------------+         +---------+---------+
                                            |
                                  +---------v---------+
                                  | Frame Buffer      |
                                  | (PNG/JPEG stream) |
                                  +---------+---------+
                                            |
                                  +---------v---------+
                                  |   FFmpeg           |
                                  | -framerate 30      |
                                  | -crf 18            |
                                  | -pix_fmt yuv420p   |
                                  +---------+---------+
                                            |
                                  +---------v---------+
                                  |   output.mp4      |
                                  +-------------------+
```

### Implementation Steps

1. **Launch Chrome headless** with flags:
   ```
   --deterministic-mode
   --enable-begin-frame-control
   --run-all-compositor-stages-before-draw
   --use-angle=gl
   --disable-gpu-vsync
   ```

2. **Inject time shim** via `page.evaluateOnNewDocument()` before page load:
   - Override `requestAnimationFrame`, `setTimeout`, `setInterval`
   - Override `Date.now()`, `performance.now()`, `new Date()`
   - Virtual time advances only on external signal

3. **For each frame:**
   ```javascript
   // Advance virtual time by 1000/fps ms
   advanceVirtualTime(1000 / targetFPS);

   // Trigger BeginFrame with screenshot
   const { screenshotData } = await cdp.send('HeadlessExperimental.beginFrame', {
     interval: 1000 / targetFPS,
     screenshot: { format: 'png' }
   });

   // Pipe to FFmpeg or save to disk
   writeFrame(screenshotData);
   ```

4. **Encode with FFmpeg:**
   ```bash
   ffmpeg -framerate 30 -f image2pipe -i pipe:0 \
     -c:v libx264 -preset slow -crf 18 \
     -pix_fmt yuv420p -movflags +faststart \
     demo.mp4
   ```

### Quick Start: Use puppeteer-capture

For fastest time-to-working, use `puppeteer-capture` directly:

```javascript
import { capture, launch } from 'puppeteer-capture';

const browser = await launch({ headless: true });
const page = await browser.newPage();
const cap = await capture(page);

await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 2 });

await cap.start('demo.mp4');

// Run your demo scenario here
// cap.waitForTimeout() uses virtual time, not wall-clock time
await cap.waitForTimeout(30000); // 30 seconds of virtual time

await cap.stop();
await browser.close();
```

### Performance Expectations

| Setting | Rendering Speed | Output Quality |
|---------|----------------|----------------|
| 1080p @ 30fps, PNG capture | ~2-3x slower than real-time | Excellent |
| 1080p @ 30fps, JPEG capture | ~1.5-2x slower than real-time | Good |
| 4K @ 30fps (2x DPR), PNG | ~4-6x slower than real-time | Outstanding |
| 1080p @ 60fps, PNG | ~4-6x slower than real-time | Excellent |

A 30-second demo at 1080p/30fps should render in roughly 60-90 seconds. Perfectly acceptable for a competition submission pipeline.

---

## Sources

- [Replit: We Built a Video Rendering Engine by Lying to the Browser About What Time It Is](https://blog.replit.com/browsers-dont-want-to-be-cameras)
- [Why I Built puppeteer-capture - Alexey Pelykh](https://alexey-pelykh.com/blog/why-i-built-puppeteer-capture/)
- [puppeteer-capture on npm](https://www.npmjs.com/package/puppeteer-capture)
- [WebVideoCreator - GitHub](https://github.com/Vinlic/WebVideoCreator)
- [Remotion: renderFrames() docs](https://www.remotion.dev/docs/renderer/render-frames)
- [Why Remotion chose frame-by-frame over recording](https://github.com/orgs/remotion-dev/discussions/4351)
- [timecut - GitHub](https://github.com/tungs/timecut)
- [Chrome DevTools Protocol - HeadlessExperimental domain](https://chromedevtools.github.io/devtools-protocol/tot/HeadlessExperimental/)
- [Chrome DevTools Protocol - Page domain](https://chromedevtools.github.io/devtools-protocol/tot/Page/)
- [Chrome DevTools Protocol - Emulation domain](https://chromedevtools.github.io/devtools-protocol/tot/Emulation/)
- [Capture CSS Animations into video with frame precision - Chromium headless-dev](https://groups.google.com/a/chromium.org/g/headless-dev/c/QBQEm5Yd3_E)
- [Page.screencastFrame causes huge performance regression](https://issues.chromium.org/issues/40934921)
- [screencastFrame frames out-of-order - Issue #117](https://github.com/ChromeDevTools/devtools-protocol/issues/117)
- [Screenshot performance - Issue #28](https://github.com/ChromeDevTools/devtools-protocol/issues/28)
- [Puppeteer slow screenshots - Issue #476](https://github.com/puppeteer/puppeteer/issues/476)
- [Puppeteer burst mode - Issue #3502](https://github.com/puppeteer/puppeteer/issues/3502)
- [Optimize for speed with CDP screenshots](https://screenshotone.com/blog/optimize-for-speed-when-rendering-screenshots-in-puppeteer-and-chrome-devtools-protocol/)
- [Headless Chrome testing WebGL with Playwright](https://www.createit.com/blog/headless-chrome-testing-webgl-using-playwright/)
- [Headfull browsers beat headless - Pierce Freeman](https://freeman.vc/notes/headfull-browsers-beat-headless)
- [High DPI rendering on HTML5 canvas](https://cmdcolin.github.io/posts/2014-05-22/)
- [Canvas retina fix - GitHub Gist](https://gist.github.com/callumlocke/cc258a193839691f60dd)
- [FFmpeg image sequence to video](https://shotstack.io/learn/use-ffmpeg-to-convert-images-to-video/)
- [Web Animation Performance Fundamentals - freeCodeCamp](https://www.freecodecamp.org/news/web-animation-performance-fundamentals/)
- [Jank busting for better rendering performance - web.dev](https://web.dev/articles/speed-rendering)
- [Playwright video recording docs](https://playwright.dev/docs/videos)
- [Playwright CDPSession docs](https://playwright.dev/docs/api/class-cdpsession)
- [Remotion fundamentals](https://www.remotion.dev/docs/the-fundamentals)
- [Document.getAnimations() - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Document/getAnimations)
- [Animation.pause() - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Animation/pause)
- [Window.devicePixelRatio - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Window/devicePixelRatio)
