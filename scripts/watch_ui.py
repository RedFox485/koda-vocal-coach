#!/usr/bin/env python3
"""
Watch the test UI while a recording streams — takes screenshots at intervals.
Outputs PNGs to /tmp/koda_watch/ so Claude can view them with Read tool.

Usage:
  python scripts/watch_ui.py                        # stream first file, 4x speed
  python scripts/watch_ui.py --file "Chris Young"   # match by name substring
  python scripts/watch_ui.py --speed 1 --interval 2 # 1x speed, shot every 2s
  python scripts/watch_ui.py --url http://localhost:8080/test
"""
import argparse
import os
import time
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUTDIR = Path("/tmp/koda_watch")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/test")
    parser.add_argument("--file", default=None, help="Substring to match recording name")
    parser.add_argument("--speed", default="4", help="Speed multiplier select option value")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between screenshots")
    parser.add_argument("--duration", type=float, default=30.0, help="Total watch duration in seconds")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    # Clean old shots
    for f in OUTDIR.glob("shot_*.png"):
        f.unlink()

    print(f"Opening: {args.url}")
    print(f"Speed: {args.speed}x | Screenshot every {args.interval}s for {args.duration}s")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        # Capture console for debugging
        page.on("console", lambda msg: print(f"  [browser] {msg.type}: {msg.text[:120]}"))

        page.goto(args.url, wait_until="networkidle")
        print("Page loaded.")

        # Set speed
        page.select_option("#speed-select", args.speed)

        # Select file if specified
        if args.file:
            btns = page.query_selector_all(".rec-btn")
            matched = False
            for btn in btns:
                if args.file.lower() in btn.inner_text().lower():
                    btn.click()
                    print(f"Selected: {btn.inner_text()}")
                    matched = True
                    break
            if not matched:
                print(f"[WARN] No file matching '{args.file}' — using auto-selected first")

        # Take initial shot
        shot0 = OUTDIR / "shot_00_before.png"
        page.screenshot(path=str(shot0), full_page=True)
        print(f"  Shot: {shot0.name}")

        # Click Stream
        page.click("#start-btn")
        print("Clicked Stream — recording started")
        time.sleep(0.5)

        # Take shots at intervals
        shot_num = 1
        t0 = time.time()
        while time.time() - t0 < args.duration:
            elapsed = time.time() - t0
            shot_path = OUTDIR / f"shot_{shot_num:02d}_{elapsed:.0f}s.png"
            page.screenshot(path=str(shot_path), full_page=True)
            print(f"  Shot: {shot_path.name}")
            shot_num += 1
            time.sleep(args.interval)

        # Final shot
        shot_final = OUTDIR / f"shot_{shot_num:02d}_final.png"
        page.screenshot(path=str(shot_final), full_page=True)
        print(f"  Shot: {shot_final.name} (final)")

        browser.close()

    shots = sorted(OUTDIR.glob("shot_*.png"))
    print(f"\nDone. {len(shots)} screenshots in {OUTDIR}")
    print("View with: Read tool on each file path")
    for s in shots:
        print(f"  {s}")


if __name__ == "__main__":
    main()
