#!/usr/bin/env python3
"""Harvest Gemini audio clips by injecting singing and capturing all responses.

Connects to /debug/ws to capture events while inject script feeds audio via /inject/ws.
Saves each Gemini audio response (greeting, coaching cues, song summary) as a WAV file.

Usage:
    python scripts/harvest_gemini_audio.py
    python scripts/harvest_gemini_audio.py --runs 3  # Multiple runs for best coaching cues
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent / "reel"))

from core.inject import inject

# Config
BACKEND = "localhost:8000"
AUDIO_FILE = str(ROOT / "audio/singing/better_days_courtney_odom.wav")
OUTPUT_DIR = ROOT / "audio" / "gemini_clips"
PAUSES = [
    {"at": 45.0, "duration": 8.0, "label": "coaching_cue_1"},
    {"at": 75.0, "duration": 8.0, "label": "coaching_cue_2"},
]


def save_audio_clip(audio_b64: str, filename: str, sample_rate: int = 24000):
    """Decode base64 audio from Gemini and save as WAV."""
    audio_bytes = base64.b64decode(audio_b64)
    # Gemini native audio is raw PCM Int16 at 24kHz
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(filename, audio, sample_rate)
    duration = len(audio) / sample_rate
    print(f"  Saved: {filename} ({duration:.1f}s, {len(audio_bytes)/1024:.0f}KB)")
    return duration


async def harvest_run(run_id: int):
    """Single harvest run: inject audio + capture all Gemini responses."""
    run_dir = OUTPUT_DIR / f"run_{run_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    events_captured = []

    # First, connect a "singer" via /ws to trigger Gemini greeting and register for events
    # Then listen on that connection for Gemini audio
    singer_ws_url = f"ws://{BACKEND}/ws"
    inject_ws_url = f"ws://{BACKEND}/inject/ws"

    print(f"\n{'='*60}")
    print(f"Run {run_id}: Connecting singer + injecting audio")
    print(f"{'='*60}")

    async with websockets.connect(singer_ws_url, max_size=2**22) as ws:
        # Receive config message
        config_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        config = json.loads(config_msg)
        print(f"  Backend config: SR={config.get('sample_rate')}, Praat={config.get('praat_available')}")

        # Start listening for events in background
        gemini_clips = []

        async def listen():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        evt = json.loads(msg)
                        evt_type = evt.get("type", "")
                        events_captured.append(evt)

                        if evt_type in ("gemini_coaching", "gemini_greeting", "song_praise"):
                            transcript = evt.get("transcript", "")
                            audio_b64 = evt.get("audio_b64")
                            zone = evt.get("zone", "")

                            print(f"\n  ** {evt_type.upper()} ** zone={zone}")
                            print(f"     \"{transcript[:100]}\"")

                            if audio_b64:
                                clip_idx = len(gemini_clips)
                                filename = str(run_dir / f"{evt_type}_{clip_idx:02d}.wav")
                                duration = save_audio_clip(audio_b64, filename)
                                gemini_clips.append({
                                    "type": evt_type,
                                    "transcript": transcript,
                                    "zone": zone,
                                    "file": filename,
                                    "duration": duration,
                                    "timestamp": time.time(),
                                })
                            else:
                                print(f"     (no audio data)")

                        elif evt_type == "ears_frame":
                            # Count but don't print every frame
                            pass
            except websockets.ConnectionClosed:
                pass

        listener = asyncio.create_task(listen())

        # Wait a moment for Gemini greeting to arrive
        print("  Waiting for Gemini greeting (5s)...")
        await asyncio.sleep(5)

        # Now inject the singing audio
        print(f"  Injecting: {Path(AUDIO_FILE).name}")
        try:
            await inject(
                ws_url=inject_ws_url,
                audio_path=AUDIO_FILE,
                pauses=PAUSES,
                speed=1.0,
            )
        except Exception as e:
            print(f"  Injection error: {e}")

        # Wait for song-end summary
        print("  Waiting for song-end summary (10s)...")
        await asyncio.sleep(10)

        listener.cancel()

    # Save manifest
    manifest = {
        "run_id": run_id,
        "audio_file": AUDIO_FILE,
        "clips": gemini_clips,
        "total_events": len(events_captured),
        "ears_frames": len([e for e in events_captured if e.get("type") == "ears_frame"]),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n  Run {run_id} complete:")
    print(f"    Gemini clips: {len(gemini_clips)}")
    print(f"    Events: {len(events_captured)}")
    ears = len([e for e in events_captured if e.get("type") == "ears_frame"])
    print(f"    EARS frames: {ears}")
    for clip in gemini_clips:
        print(f"    - {clip['type']}: \"{clip['transcript'][:60]}\" ({clip['duration']:.1f}s)")

    return gemini_clips


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="Number of harvest runs")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_clips = []

    for i in range(1, args.runs + 1):
        clips = await harvest_run(i)
        all_clips.extend(clips)
        if i < args.runs:
            print("\n  Waiting 5s between runs (backend cooldown)...")
            await asyncio.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print(f"HARVEST COMPLETE — {len(all_clips)} Gemini clips across {args.runs} runs")
    print(f"{'='*60}")
    for clip in all_clips:
        print(f"  [{clip['type']}] \"{clip['transcript'][:70]}\" → {clip['file']}")

    # Save combined manifest
    combined = OUTPUT_DIR / "all_clips.json"
    combined.write_text(json.dumps(all_clips, indent=2))
    print(f"\nManifest: {combined}")


if __name__ == "__main__":
    asyncio.run(main())
