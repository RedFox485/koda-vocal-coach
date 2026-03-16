#!/usr/bin/env python3
"""Capture the start screen → finger tap → dashboard transition.

Records a short clip showing:
1. Koda start screen (2s idle)
2. Animated finger-tap on "Enable Microphone" button
3. Dashboard loads with UI elements appearing
4. Hold dashboard view (3s)

Output: video/captures/tap_transition.mp4
"""

import asyncio
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "video" / "captures"
BACKEND_URL = "http://localhost:8000"
VIEWPORT = {"width": 1920, "height": 1080}

# CSS for finger-tap ripple animation (Material Design style)
TAP_ANIMATION_CSS = """
@keyframes tap-ripple {
  0% { transform: translate(-50%, -50%) scale(0.3); opacity: 0.9; }
  40% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
  100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
}
@keyframes tap-finger-in {
  0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
  30% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
  60% { transform: translate(-50%, -50%) scale(0.92); opacity: 1; }
  100% { transform: translate(-50%, -50%) scale(0.92); opacity: 1; }
}
@keyframes tap-finger-out {
  0% { transform: translate(-50%, -50%) scale(0.92); opacity: 1; }
  100% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
}
#tap-cursor {
  position: fixed;
  width: 56px;
  height: 56px;
  pointer-events: none;
  z-index: 99999;
  display: none;
}
#tap-cursor .finger {
  position: absolute;
  top: 50%; left: 50%;
  width: 56px; height: 56px;
  transform: translate(-50%, -50%) scale(0.8);
  opacity: 0;
}
#tap-cursor .finger svg {
  width: 100%; height: 100%;
  filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
}
#tap-cursor .ripple {
  position: absolute;
  top: 50%; left: 50%;
  width: 80px; height: 80px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 70%);
  transform: translate(-50%, -50%) scale(0);
  opacity: 0;
}
"""

# SVG finger icon (simple touch indicator)
TAP_CURSOR_HTML = """
<div id="tap-cursor">
  <div class="ripple"></div>
  <div class="finger">
    <svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2C10.34 2 9 3.34 9 5v5.5c-.6-.31-1.28-.5-2-.5-2.21 0-4 1.79-4 4 0 .05 0 .09.01.14C3.01 14.26 3 14.46 3 14.67V17c0 3.31 2.69 6 6 6h4c2.97 0 5.43-2.17 5.9-5.01L20 12.5c0-1.38-1.12-2.5-2.5-2.5-.37 0-.72.08-1.04.22-.51-.88-1.46-1.47-2.54-1.47-.36 0-.7.07-1.01.21-.15-.24-.34-.46-.55-.64V5c0-1.66-1.34-3-3-3zm0 2c.55 0 1 .45 1 1v6h1.5c.55 0 1 .45 1 1s-.45 1-1 1h-.5v-1c0-.55.45-1 1-1s1 .45 1 1v1h.5c.28 0 .5.22.5.5l-1.05 5.25C15.67 19.56 14.42 21 13 21H9c-2.21 0-4-1.79-4-4v-2.33c0-.13.01-.25.02-.38C5.03 14.1 5 13.9 5 13.7 5 12.21 6.21 11 7.5 11c.56 0 1.08.18 1.5.5V5c0-.55.45-1 1-1z" opacity="0.9"/>
      <circle cx="12" cy="5" r="3" fill="rgba(255,255,255,0.15)"/>
    </svg>
  </div>
</div>
"""


async def animate_tap(page, selector):
    """Show finger-tap animation on an element, then click it."""
    # Get button position
    box = await page.locator(selector).bounding_box()
    if not box:
        print(f"[tap] WARNING: Could not find {selector}")
        return

    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2

    # Position the cursor
    await page.evaluate(f"""() => {{
        const cursor = document.getElementById('tap-cursor');
        cursor.style.left = '{cx}px';
        cursor.style.top = '{cy}px';
        cursor.style.display = 'block';
    }}""")

    # Animate finger appearing (press down)
    await page.evaluate("""() => {
        const finger = document.querySelector('#tap-cursor .finger');
        finger.style.animation = 'tap-finger-in 0.4s ease-out forwards';
    }""")
    await asyncio.sleep(0.4)

    # Ripple effect (the "press")
    await page.evaluate("""() => {
        const ripple = document.querySelector('#tap-cursor .ripple');
        ripple.style.animation = 'tap-ripple 0.6s ease-out forwards';
    }""")
    await asyncio.sleep(0.15)

    # Also trigger button active state
    await page.evaluate(f"""() => {{
        const btn = document.querySelector('{selector}');
        btn.style.transform = 'scale(0.98)';
    }}""")
    await asyncio.sleep(0.2)

    # Release
    await page.evaluate(f"""() => {{
        const btn = document.querySelector('{selector}');
        btn.style.transform = '';
    }}""")

    # Finger lifts away
    await page.evaluate("""() => {
        const finger = document.querySelector('#tap-cursor .finger');
        finger.style.animation = 'tap-finger-out 0.3s ease-in forwards';
    }""")
    await asyncio.sleep(0.15)

    # Actually click
    await page.click(selector)

    await asyncio.sleep(0.3)

    # Hide cursor
    await page.evaluate("""() => {
        document.getElementById('tap-cursor').style.display = 'none';
    }""")


async def capture_tap():
    """Capture the tap transition sequence."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUTPUT_DIR),
            record_video_size=VIEWPORT,
            device_scale_factor=1,
        )
        page = await context.new_page()

        print("[tap] Navigating to Koda...")
        await page.goto(BACKEND_URL, wait_until="networkidle")

        # Inject tap animation CSS and HTML
        await page.add_style_tag(content=TAP_ANIMATION_CSS)
        await page.evaluate(f"""() => {{
            document.body.insertAdjacentHTML('beforeend', `{TAP_CURSOR_HTML}`);
        }}""")

        # Let start screen render and settle (this is the "hero" moment)
        print("[tap] Recording start screen (2s)...")
        await asyncio.sleep(2.0)

        # Animate the tap on "Enable Microphone" button
        # Note: We actually click watch-btn but visually target start-btn
        # because headless has no mic. We'll make watch-btn invisible
        # and position the tap on start-btn, then secretly click watch-btn
        print("[tap] Animating finger tap on Enable Microphone...")

        # Hide watch button, make start button the visual target
        await page.evaluate("""() => {
            document.getElementById('watch-btn').style.display = 'none';
        }""")

        # Animate tap on start button
        await animate_tap(page, "#start-btn")

        # The start-btn click will try to get mic (will fail silently in headless)
        # We need to actually dismiss the overlay ourselves
        await asyncio.sleep(0.3)
        await page.evaluate("""() => {
            // Dismiss overlay like watch-btn does
            document.getElementById('start-overlay').style.display = 'none';
            // Connect as watch-only
            const wsUrl = `ws://${window.location.host}/ws`;
            const ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                ws.send(JSON.stringify({type: 'register', role: 'watch'}));
            };
            ws.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    if (window.handleEvent) window.handleEvent(data);
                } catch(e) {}
            };
        }""")

        # Dashboard loads
        print("[tap] Dashboard loading...")
        await asyncio.sleep(4.0)  # Let dashboard settle

        print("[tap] Recording dashboard state (3s)...")
        await asyncio.sleep(3.0)

        # Close - this finalizes video
        await context.close()
        await browser.close()

    # Find and convert the recording
    webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda f: f.stat().st_mtime, reverse=True)
    output_mp4 = OUTPUT_DIR / "tap_transition.mp4"

    if webm_files:
        recorded = webm_files[0]
        print(f"[tap] Raw recording: {recorded} ({recorded.stat().st_size / 1024:.0f} KB)")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(recorded),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            size = output_mp4.stat().st_size / 1024 / 1024
            print(f"[tap] Final: {output_mp4} ({size:.1f} MB)")
        else:
            print(f"[tap] FFmpeg error: {result.stderr[-300:]}")

        # Clean up webm
        recorded.unlink(missing_ok=True)
    else:
        print("[tap] WARNING: No webm recording found!")

    return output_mp4


if __name__ == "__main__":
    result = asyncio.run(capture_tap())
    print(f"\nDone! Clip: {result}")
