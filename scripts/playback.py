#!/usr/bin/env python3
"""
Stream an audio file through the backend so the browser UI shows live analysis
and Gemini coaching fires — no microphone needed.

Usage:
    .venv/bin/python3 scripts/playback.py              # interactive file picker
    .venv/bin/python3 scripts/playback.py --list       # list available files
    .venv/bin/python3 scripts/playback.py 1            # play file #1 directly
    .venv/bin/python3 scripts/playback.py --speed 2    # play at 2x (faster test)

The browser must be open at http://localhost:8765 — it will show all zones,
pitch, and play Gemini coaching audio in real-time.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
import websockets

SR         = 44100
CHUNK_S    = 0.1
CHUNK_SAMP = int(SR * CHUNK_S)
WS_URL     = "ws://localhost:8765/ws"

# Audio file search paths
SEARCH_DIRS = [
    Path("Vocal test recording sessions"),
    Path("Vocal test recording sessions/Anchors"),
    Path("data"),
    Path("."),
]
AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}


def find_audio_files() -> list[Path]:
    files = []
    for d in SEARCH_DIRS:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS and f not in files:
                    files.append(f)
    return files


def pick_file(arg: str | None) -> Path | None:
    files = find_audio_files()
    if not files:
        print("No audio files found. Place .m4a/.wav files in 'Vocal test recording sessions/'")
        return None

    if arg and arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(files):
            return files[idx]
        print(f"Invalid number — pick 1–{len(files)}")
        return None

    print("\nAvailable audio files:")
    for i, f in enumerate(files, 1):
        print(f"  {i:>2}.  {f}")
    print()
    try:
        choice = input("Enter number: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
        print(f"Invalid — pick 1–{len(files)}")
        return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


async def stream(path: Path, speed: float):
    print(f"\nLoading: {path}")
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    frames = [y[i * CHUNK_SAMP:(i + 1) * CHUNK_SAMP] for i in range(len(y) // CHUNK_SAMP)]
    duration = len(y) / SR
    print(f"Duration: {duration:.1f}s  →  {len(frames)} frames  speed={speed}x")
    print(f"\nOpen http://localhost:8765 in your browser to see live analysis.")
    print(f"Streaming... (press q + Enter to stop early)\n")

    delay = CHUNK_S / speed
    stop_event = asyncio.Event()

    # Listen for 'q' keypress in background
    async def _watch_quit():
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if line.strip().lower() == 'q':
            stop_event.set()

    # Start speaker playback (full file, always 1x)
    print("Playing audio through speakers...")
    sd.play(y, samplerate=SR)

    try:
        # ping_interval=None disables auto-pings that close the connection on long files
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            # Drain the config message the server sends on connect
            try:
                import json
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                cfg = json.loads(raw)
                print(f"Backend: green<{cfg['strain_thresholds']['green']}  "
                      f"yellow<{cfg['strain_thresholds']['yellow']}\n")
            except Exception:
                pass

            quit_task = asyncio.create_task(_watch_quit())

            for i, chunk in enumerate(frames):
                if stop_event.is_set():
                    break
                await ws.send(chunk.astype(np.float32).tobytes())
                await asyncio.sleep(delay)
                # Progress every 10s
                if i > 0 and (i * CHUNK_S) % 10 < CHUNK_S:
                    elapsed = i * CHUNK_S
                    print(f"  {elapsed:.0f}s / {duration:.0f}s  (q + Enter to stop)")

            quit_task.cancel()

            if not stop_event.is_set():
                # Wait for audio to finish playing + Gemini response
                remaining = duration - (len(frames) * CHUNK_S)
                wait_s = max(0, remaining) + 6.0
                print(f"\nDone streaming — waiting {wait_s:.0f}s for audio + Gemini...")
                await asyncio.sleep(wait_s)
                print("Finished.")

    except websockets.exceptions.ConnectionRefusedError:
        print(f"ERROR: Could not connect to {WS_URL}")
        print("Is the backend running? Try: .venv/bin/python3 src/vocal_health_backend.py")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sd.stop()


def main():
    parser = argparse.ArgumentParser(description="Stream audio file to Koda backend")
    parser.add_argument("file", nargs="?", help="File number or path")
    parser.add_argument("--list", action="store_true", help="List available files and exit")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default 1.0)")
    args = parser.parse_args()

    if args.list:
        files = find_audio_files()
        print("\nAvailable audio files:")
        for i, f in enumerate(files, 1):
            print(f"  {i:>2}.  {f}")
        return

    # Change to project root so relative paths work
    import os
    os.chdir(Path(__file__).parent.parent)

    path = pick_file(args.file)
    if path is None:
        return

    asyncio.run(stream(path, args.speed))


if __name__ == "__main__":
    main()
