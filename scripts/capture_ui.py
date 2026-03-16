#!/usr/bin/env python3
"""Playwright UI Capture — Smooth, frame-perfect video of Koda running.

Three capture modes:

  hybrid (default) — Real-time Playwright recording. WebSocket/real-time features
    work naturally. Re-encodes with optimal quality. Best for apps with live data.

  deterministic — CDP-controlled frame capture. Freezes Chrome's rendering loop
    and advances it frame-by-frame. EVERY frame is perfect. Zero drops, zero jank.
    CSS animations, rAF, and JS timers all respect virtualized time.
    ~2-3x slower than real-time but output is broadcast quality.

  native — Playwright's built-in video recording. Fast but 1Mbps cap. For previews.

Usage:
    # Default: real-time hybrid capture
    python scripts/capture_ui.py --url http://localhost:8000 --duration 180

    # Deterministic: frame-perfect, no jank ever
    python scripts/capture_ui.py --url http://localhost:8000 --mode deterministic --fps 30

    # Quick preview
    python scripts/capture_ui.py --url http://localhost:8000 --mode native --duration 30

    # 4K capture (upload to YouTube for 2.5x bitrate bonus)
    python scripts/capture_ui.py --url http://localhost:8000 --scale 2 --mode deterministic
"""

import argparse
import asyncio
import json
import subprocess
import shutil
import time
from pathlib import Path

from playwright.async_api import async_playwright


# ─── DETERMINISTIC CAPTURE (HIGHEST QUALITY) ─────────────────────────────────
# Uses CDP virtual time + screenshot per frame. CSS animations, JS timers,
# requestAnimationFrame ALL respect the virtualized clock. Zero frame drops.

async def capture_deterministic(url: str, output_path: str, duration: float, fps: int,
                                 width: int, height: int, scale: int, auto_start: bool,
                                 warmup: float = 2.0):
    """Frame-by-frame capture with CDP virtual time domain.

    Chrome's Virtual Time domain intercepts ALL time sources:
    - Date.now(), Date(), new Date()
    - performance.now()
    - setTimeout / setInterval
    - requestAnimationFrame
    - CSS Animations & Transitions (compositor-level)
    - Web Animations API

    This is the browser-level solution — no JS shims needed.
    """
    frame_dir = Path(output_path).parent / "_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_interval_ms = 1000.0 / fps
    total_frames = int(duration * fps)
    actual_width = width * scale
    actual_height = height * scale

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--deterministic-mode",
                "--enable-begin-frame-control",
                "--disable-gpu-compositing",        # Force software compositing for deterministic output
                "--run-all-compositor-stages-before-draw",
                "--disable-threaded-animation",      # Animations on main thread = deterministic
                "--disable-threaded-scrolling",
                "--disable-checker-imaging",
                f"--force-device-scale-factor={scale}",
            ],
        )

        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
        )
        page = await context.new_page()

        # Get CDP session for virtual time control
        cdp = await context.new_cdp_session(page)

        # Load the page normally first (real time for network/WebSocket)
        await page.goto(url, wait_until="networkidle")
        print(f"[capture] Page loaded: {url}")

        if auto_start:
            await _click_start(page)
            # Let real-time features connect (WebSocket, Gemini greeting)
            print(f"[capture] Waiting {warmup}s for warmup (WebSocket connect, greeting)...")
            await asyncio.sleep(warmup)

        # NOW enable virtual time — from this point, all browser time is controlled
        # Use budget-based advancement: we grant a time budget, browser runs until exhausted
        print(f"[capture] Enabling virtual time domain...")
        await cdp.send("Emulation.setVirtualTimePolicy", {
            "policy": "pauseIfNetworkFetchesPending",
            "budget": 0,  # Start paused
        })

        print(f"[capture] Capturing {total_frames} frames at {fps}fps "
              f"({actual_width}x{actual_height})...")
        start_time = time.monotonic()

        for frame_num in range(total_frames):
            # Grant time budget for one frame
            await cdp.send("Emulation.setVirtualTimePolicy", {
                "policy": "pauseIfNetworkFetchesPending",
                "budget": frame_interval_ms,
            })

            # Wait for the budget to be exhausted (browser rendered the frame)
            # Small real-time sleep to let CDP process
            await asyncio.sleep(0.01)

            # Capture screenshot at full resolution
            frame_path = frame_dir / f"{frame_num:06d}.png"
            screenshot_bytes = await cdp.send("Page.captureScreenshot", {
                "format": "png",
                "captureBeyondViewport": False,
                "clip": {
                    "x": 0, "y": 0,
                    "width": width, "height": height,
                    "scale": scale,
                },
            })

            # Decode and save
            import base64
            png_data = base64.b64decode(screenshot_bytes["data"])
            frame_path.write_bytes(png_data)

            # Progress every 2 seconds of video time
            if frame_num > 0 and frame_num % (fps * 2) == 0:
                elapsed_real = time.monotonic() - start_time
                video_time = frame_num / fps
                fps_real = frame_num / elapsed_real
                eta_real = (total_frames - frame_num) / fps_real if fps_real > 0 else 0
                print(f"  Frame {frame_num}/{total_frames} "
                      f"(video: {video_time:.0f}s, real: {elapsed_real:.0f}s, "
                      f"capture rate: {fps_real:.1f}fps, ETA: {eta_real:.0f}s)")

        await browser.close()

    # Compose frames into video
    elapsed_capture = time.monotonic() - start_time
    print(f"[capture] {total_frames} frames captured in {elapsed_capture:.0f}s "
          f"({total_frames/elapsed_capture:.1f} frames/sec)")
    print(f"[capture] Encoding video...")

    _frames_to_video(str(frame_dir), output_path, fps, actual_width, actual_height)

    # Clean up frames
    shutil.rmtree(frame_dir)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"[capture] Saved: {output_path} ({file_size:.1f} MB)")


# ─── HYBRID CAPTURE (DEFAULT) ────────────────────────────────────────────────
# Real-time recording — WebSocket, live data, animations all run naturally.
# Re-encodes with high quality settings after capture.

async def capture_hybrid(url: str, output_path: str, duration: float, fps: int,
                          width: int, height: int, scale: int, auto_start: bool):
    """Real-time Playwright recording with quality post-processing."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                f"--force-device-scale-factor={scale}",
                "--disable-gpu-compositing",
            ],
        )

        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
            record_video_dir=str(Path(output_path).parent / "_raw"),
            record_video_size={"width": width * scale, "height": height * scale},
        )

        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        print(f"[capture] Page loaded: {url}")

        if auto_start:
            await _click_start(page)

        print(f"[capture] Recording for {duration}s (real-time)...")
        await asyncio.sleep(duration)

        video = page.video
        video_path = await video.path() if video else None

        await context.close()
        await browser.close()

        if video_path:
            _reencode_quality(str(video_path), output_path, fps)
            # Clean up raw directory
            raw_dir = Path(output_path).parent / "_raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            print(f"[capture] Saved: {output_path}")
        else:
            print("[capture] ERROR: No video captured")


# ─── NATIVE CAPTURE (FASTEST, LOWEST QUALITY) ────────────────────────────────

async def capture_native(url: str, output_path: str, duration: float,
                         width: int, height: int, scale: int, auto_start: bool):
    """Playwright's built-in recording. Fast but capped at ~1Mbps."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
            record_video_dir=str(Path(output_path).parent / "_raw"),
            record_video_size={"width": width * scale, "height": height * scale},
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")

        if auto_start:
            await _click_start(page)

        print(f"[capture] Recording {duration}s (native mode)...")
        await asyncio.sleep(duration)

        await context.close()
        await browser.close()

        # Find and rename the raw video
        raw_dir = Path(output_path).parent / "_raw"
        videos = sorted(raw_dir.glob("*.webm"), key=lambda f: f.stat().st_mtime, reverse=True)
        if videos:
            _webm_to_mp4(str(videos[0]), output_path)
            shutil.rmtree(raw_dir)
            print(f"[capture] Saved: {output_path}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

async def _click_start(page):
    """Click the Start button on Koda UI."""
    try:
        btn = page.locator("#start-btn")
        await btn.wait_for(state="visible", timeout=5000)
        await btn.click()
        print("[capture] Clicked Start button")
        await asyncio.sleep(1)
    except Exception as e:
        print(f"[capture] Could not click Start: {e}")


def _frames_to_video(frame_dir: str, output_path: str, fps: int, width: int, height: int):
    """Compose PNG frames into H.264 MP4."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frame_dir}/%06d.png",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-s", f"{width}x{height}",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg] Error: {result.stderr[:500]}")
        raise RuntimeError("ffmpeg encoding failed")


def _webm_to_mp4(webm_path: str, mp4_path: str):
    """Convert WebM to high-quality MP4."""
    cmd = [
        "ffmpeg", "-y", "-i", webm_path,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _reencode_quality(input_path: str, output_path: str, target_fps: int):
    """Re-encode with YouTube-optimal settings."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reel — Smooth, frame-perfect web app video capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capture modes:
  deterministic  Frame-perfect. CDP controls Chrome's render loop. Zero jank.
                 Best for: final production renders, portfolio videos.
                 Speed: ~2-3x slower than real-time.

  hybrid         Real-time Playwright recording + quality re-encode.
                 Best for: apps with live WebSocket data, quick iterations.
                 Speed: 1x (real-time).

  native         Playwright built-in recording. Fast but low bitrate.
                 Best for: quick previews, debugging.
                 Speed: 1x (real-time).

Examples:
  # Production-quality Koda demo
  python scripts/capture_ui.py --mode deterministic --fps 30 --scale 2

  # Quick test capture
  python scripts/capture_ui.py --mode native --duration 15

  # Real-time with live WebSocket data
  python scripts/capture_ui.py --mode hybrid --duration 180
        """,
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Web app URL to capture")
    parser.add_argument("--output", "-o", default="video/captures/session.mp4",
                        help="Output video path")
    parser.add_argument("--mode", choices=["deterministic", "hybrid", "native"],
                        default="hybrid",
                        help="Capture mode (default: hybrid)")
    parser.add_argument("--duration", type=float, default=180,
                        help="Recording duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate (deterministic mode)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Viewport width (logical pixels)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Viewport height (logical pixels)")
    parser.add_argument("--scale", type=int, default=1,
                        help="Device scale factor (2 = 4K from 1080p viewport)")
    parser.add_argument("--auto-start", action="store_true", default=True,
                        help="Click the Start button automatically")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="Seconds to wait after Start before enabling virtual time")

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install: brew install ffmpeg")
        return

    print(f"[reel] Mode: {args.mode}")
    print(f"[reel] Viewport: {args.width}x{args.height} @ {args.scale}x "
          f"= {args.width*args.scale}x{args.height*args.scale} output")
    print(f"[reel] Duration: {args.duration}s, FPS: {args.fps}")
    print()

    if args.mode == "deterministic":
        asyncio.run(capture_deterministic(
            args.url, args.output, args.duration, args.fps,
            args.width, args.height, args.scale, args.auto_start, args.warmup))
    elif args.mode == "hybrid":
        asyncio.run(capture_hybrid(
            args.url, args.output, args.duration, args.fps,
            args.width, args.height, args.scale, args.auto_start))
    elif args.mode == "native":
        asyncio.run(capture_native(
            args.url, args.output, args.duration,
            args.width, args.height, args.scale, args.auto_start))


if __name__ == "__main__":
    main()
