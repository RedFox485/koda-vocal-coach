#!/usr/bin/env python3
"""Capture Koda demo video — inject audio + record browser simultaneously.

Usage:
    python scripts/capture_demo.py
    python scripts/capture_demo.py --speed 1.0 --duration 180

This script:
1. Opens Playwright browser at the Koda URL
2. Clicks Start to begin the session
3. Injects the singing audio via WebSocket (with pauses for coaching cues)
4. Records the browser UI responding in real-time
5. Saves the recording to video/captures/demo_session.mp4
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root and reel to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "reel"))

from playwright.async_api import async_playwright

# Import inject from reel
from core.inject import inject


# ─── CONFIG ───────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"
WS_INJECT_URL = "ws://localhost:8000/inject/ws"
AUDIO_FILE = str(ROOT / "audio/singing/better_days_courtney_odom.wav")
OUTPUT_DIR = ROOT / "video" / "captures"
OUTPUT_FILE = OUTPUT_DIR / "demo_session.webm"
FINAL_FILE = OUTPUT_DIR / "demo_session.mp4"

# Pauses for coaching cue moments — the inject script sends silence here,
# giving Gemini time to speak a coaching cue
PAUSES = [
    {"at": 45.0, "duration": 8.0, "label": "coaching_cue_1"},
    {"at": 75.0, "duration": 8.0, "label": "coaching_cue_2"},
]

VIEWPORT = {"width": 1920, "height": 1080}
SPEED = 1.0  # Real-time


async def capture_demo():
    """Run the full capture pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser with video recording
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUTPUT_DIR),
            record_video_size=VIEWPORT,
            device_scale_factor=1,
        )
        page = await context.new_page()

        print(f"[capture] Navigating to {BACKEND_URL}...")
        await page.goto(BACKEND_URL, wait_until="networkidle")
        await asyncio.sleep(2)  # Let UI settle

        # Take a screenshot of the start screen
        await page.screenshot(path=str(OUTPUT_DIR / "shot2_start.png"))
        print("[capture] Start screen captured")

        # Click Watch Only — headless browser has no mic, so we use watch mode
        # + inject audio via /inject/ws endpoint
        print("[capture] Clicking Watch Only (headless has no mic)...")
        watch_btn = page.locator("#watch-btn")
        if await watch_btn.is_visible():
            await watch_btn.click()
            print("[capture] Watch mode activated — dashboard visible")
            await asyncio.sleep(3)  # Let UI settle and WebSocket connect

            # Take screenshot of dashboard
            await page.screenshot(path=str(OUTPUT_DIR / "shot3_greeting.png"))
            print("[capture] Dashboard state captured")
        else:
            print("[capture] WARNING: Watch button not found, trying Start...")
            start_btn = page.locator("#start-btn")
            if await start_btn.is_visible():
                await start_btn.click()
                await asyncio.sleep(5)
                await page.screenshot(path=str(OUTPUT_DIR / "shot3_greeting.png"))

        # Now inject audio — this runs in real-time while the browser records
        print(f"[capture] Starting audio injection: {Path(AUDIO_FILE).name}")
        print(f"[capture] Pauses at: {[p['at'] for p in PAUSES]}s")

        event_log = []
        def on_event(evt):
            evt["_wall_time"] = time.time()
            event_log.append(evt)
            etype = evt.get("type", "")
            if etype == "gemini_coaching":
                print(f"[capture] ** COACHING CUE received **")
            elif etype == "ears_frame":
                zone = evt.get("zone", "")
                strain = evt.get("strain_score", 0)
                if event_log and len([e for e in event_log if e.get("type") == "ears_frame"]) % 50 == 0:
                    print(f"[capture] Frame: strain={strain:.3f} zone={zone}")

        try:
            await inject(
                ws_url=WS_INJECT_URL,
                audio_path=AUDIO_FILE,
                pauses=PAUSES,
                speed=SPEED,
                on_event=on_event,
            )
        except Exception as e:
            print(f"[capture] Injection error: {e}")

        # Wait a moment for final UI updates
        await asyncio.sleep(3)

        # Take final screenshots
        await page.screenshot(path=str(OUTPUT_DIR / "shot8_close.png"))
        print("[capture] Final state captured")

        # Save event log
        event_log_path = OUTPUT_DIR / "event_log.jsonl"
        with open(event_log_path, "w") as f:
            for evt in event_log:
                f.write(json.dumps(evt) + "\n")
        print(f"[capture] Saved {len(event_log)} events to {event_log_path}")

        # Close browser — this finalizes the video recording
        await context.close()
        await browser.close()

    # Find the recorded video (Playwright saves it with a random name)
    webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda f: f.stat().st_mtime, reverse=True)
    if webm_files:
        recorded = webm_files[0]
        print(f"[capture] Raw recording: {recorded} ({recorded.stat().st_size / 1024 / 1024:.1f} MB)")

        # Re-encode to MP4 with good quality
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", str(recorded),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",  # No audio in capture (audio comes from CutRoom mix)
            str(FINAL_FILE),
        ]
        print(f"[capture] Re-encoding to MP4...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            size = FINAL_FILE.stat().st_size / 1024 / 1024
            print(f"[capture] Final: {FINAL_FILE} ({size:.1f} MB)")
        else:
            print(f"[capture] FFmpeg error: {result.stderr[-300:]}")
    else:
        print("[capture] WARNING: No webm recording found!")

    # Summary
    ears_frames = len([e for e in event_log if e.get("type") == "ears_frame"])
    coaching = len([e for e in event_log if e.get("type") == "gemini_coaching"])
    print(f"\n[capture] DONE")
    print(f"  Frames: {ears_frames}")
    print(f"  Coaching cues: {coaching}")
    print(f"  Screenshots: {len(list(OUTPUT_DIR.glob('*.png')))}")
    print(f"  Video: {FINAL_FILE}")


if __name__ == "__main__":
    asyncio.run(capture_demo())
