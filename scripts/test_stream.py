#!/usr/bin/env python3
"""
Stream an audio file through the vocal health WebSocket backend.
Captures live analysis results exactly as the browser would see them.

Usage:
  python scripts/test_stream.py <audio_file>
  python scripts/test_stream.py <audio_file> --url wss://koda-vocal-coach.fly.dev/ws
  python scripts/test_stream.py <audio_file> --url ws://localhost:8080/ws --speed 4

Options:
  --url    WebSocket endpoint (default: wss://koda-vocal-coach.fly.dev/ws)
  --speed  Playback speed multiplier (default: 1.0 = real-time, 4 = 4x faster)
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import websockets

CHUNK_SAMPLES = 4096    # matches browser ScriptProcessor(4096, 1, 1)
SAMPLE_RATE = 44100
DEFAULT_URL = "wss://koda-vocal-coach.fly.dev/ws"

ZONE_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴", "idle": "⬜"}
ZONE_LABEL = {"green": "GREEN", "yellow": "YELLOW", "red": "RED",   "idle": "idle "}

BAR_WIDTH = 30

def strain_bar(score, zone):
    filled = int(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    colors = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "idle": "\033[90m"}
    c = colors.get(zone, "")
    reset = "\033[0m"
    return f"{c}{bar}{reset}"


async def stream_file(audio_path, url, speed=1.0):
    print(f"\n{'='*65}")
    print(f"  FILE : {os.path.basename(audio_path)}")
    print(f"  URL  : {url}")
    print(f"  SPEED: {speed}x")
    print(f"{'='*65}")

    # Convert to WAV
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', str(SAMPLE_RATE), tmp.name],
        capture_output=True, check=True
    )
    import librosa
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    total_s = len(audio) / SAMPLE_RATE
    print(f"  Duration: {total_s:.1f}s  |  {len(audio)//CHUNK_SAMPLES} chunks\n")

    results = []
    chunk_time_s = CHUNK_SAMPLES / SAMPLE_RATE   # ~92.9ms

    async with websockets.connect(url, ping_interval=20) as ws:
        print("  Connected. Streaming...\n")
        print(f"  {'t':>5}  {'Strain':>6}  {'Zone':>6}  {'Pitch':>6}  {'Tonos':>6}  Bar")
        print(f"  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*BAR_WIDTH}")

        # Send audio chunks + receive results concurrently
        async def send_chunks():
            for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
                chunk = audio[i:i + CHUNK_SAMPLES].astype(np.float32)
                await ws.send(chunk.tobytes())
                if speed > 0:
                    await asyncio.sleep(chunk_time_s / speed)
            # Small pause then close
            await asyncio.sleep(2.0)

        async def recv_results():
            t_start = time.time()
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        break
                    event = json.loads(msg)
                    if event.get("type") == "ears_frame":
                        t = round(time.time() - t_start, 1)
                        strain = event.get("strain_score", 0.0)
                        zone = event.get("zone", "idle")
                        pitch = event.get("pitch_note", "—")
                        tonos = event.get("tonos", 0.0)
                        emoji = ZONE_EMOJI.get(zone, "?")
                        zlabel = ZONE_LABEL.get(zone, zone)
                        bar = strain_bar(strain, zone)
                        print(f"  {t:>5.1f}  {strain:>6.3f}  {emoji} {zlabel}  {pitch:>6}  {tonos:>6.3f}  {bar}")
                        results.append(event)
            except websockets.exceptions.ConnectionClosed:
                pass

        sender = asyncio.create_task(send_chunks())
        await recv_results()
        sender.cancel()

    # Summary
    if results:
        strains = [r["strain_score"] for r in results if r.get("zone") != "idle"]
        zones = [r["zone"] for r in results if r.get("zone") != "idle"]
        if strains:
            avg = np.mean(strains)
            peak = np.max(strains)
            green_pct  = zones.count("green") / len(zones) * 100
            yellow_pct = zones.count("yellow") / len(zones) * 100
            red_pct    = zones.count("red") / len(zones) * 100
            print(f"\n  {'─'*55}")
            print(f"  SUMMARY  avg={avg:.3f}  peak={peak:.3f}")
            print(f"  🟢 {green_pct:.0f}%  🟡 {yellow_pct:.0f}%  🔴 {red_pct:.0f}%")
            print(f"  {'─'*55}\n")
        else:
            print("\n  (no voiced frames detected)\n")
    else:
        print("\n  (no results received — check server is running)\n")

    return results


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    audio_file = args[0]
    url = DEFAULT_URL
    speed = 1.0

    i = 1
    while i < len(args):
        if args[i] == "--url" and i + 1 < len(args):
            url = args[i + 1]; i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = float(args[i + 1]); i += 2
        else:
            i += 1

    asyncio.run(stream_file(audio_file, url, speed))


if __name__ == "__main__":
    main()
