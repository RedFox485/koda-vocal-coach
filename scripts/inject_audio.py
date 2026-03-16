#!/usr/bin/env python3
"""WebSocket Audio Injection — Feeds a WAV file to Koda as if from a live mic.

Usage:
    python scripts/inject_audio.py audio/singing.wav
    python scripts/inject_audio.py audio/singing.wav --ws ws://localhost:8000/ws
    python scripts/inject_audio.py audio/singing.wav --pauses pauses.json
    python scripts/inject_audio.py audio/singing.wav --speed 1.0

The backend processes the audio identically to a live browser session — strain gauge,
pitch tuner, Gemini coaching, everything responds naturally.

Pause file format (pauses.json):
    [
        {"at": 45.0, "duration": 8.0, "label": "coaching_cue_1"},
        {"at": 62.0, "duration": 8.0, "label": "coaching_cue_2"}
    ]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


CHUNK_SAMPLES = 4096   # Match browser's ScriptProcessor buffer size
TARGET_SR = 44100      # Koda expects 44100 Hz mono Float32


async def inject(ws_url: str, audio_path: str, pauses: list | None = None,
                 speed: float = 1.0, on_event=None):
    """Stream audio file to Koda backend via WebSocket.

    Args:
        ws_url: WebSocket URL (e.g. ws://localhost:8000/ws)
        audio_path: Path to WAV/FLAC/OGG audio file
        pauses: List of {"at": seconds, "duration": seconds, "label": str}
        speed: Playback speed multiplier (1.0 = real-time)
        on_event: Callback for backend events: fn(event_dict) -> None
    """
    # Load and prepare audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Stereo → mono

    if sr != TARGET_SR:
        # Simple resample via linear interpolation (good enough for this purpose)
        ratio = TARGET_SR / sr
        n_out = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_out)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = TARGET_SR

    total_duration = len(audio) / sr
    print(f"[inject] Loaded {audio_path}: {total_duration:.1f}s, {sr}Hz, {len(audio)} samples")

    # Sort pauses by timestamp
    pauses = sorted(pauses or [], key=lambda p: p["at"])
    pause_idx = 0

    chunk_duration = CHUNK_SAMPLES / sr  # ~0.093s per chunk

    async with websockets.connect(ws_url, max_size=2**20) as ws:
        print(f"[inject] Connected to {ws_url}")

        # Wait for config message from backend (only /ws sends one, /inject/ws does not)
        if "/inject/" not in ws_url:
            config_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            config = json.loads(config_msg)
            print(f"[inject] Backend config: SR={config.get('sample_rate')}, "
                  f"Praat={config.get('praat_available')}")
        else:
            print(f"[inject] Injection endpoint — no config handshake")

        # Start listening for backend events in background
        event_count = {"ears_frame": 0, "gemini_coaching": 0, "gemini_greeting": 0}

        async def listen_events():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        evt = json.loads(msg)
                        evt_type = evt.get("type", "unknown")
                        event_count[evt_type] = event_count.get(evt_type, 0) + 1
                        if on_event:
                            on_event(evt)
                        # Log important events
                        if evt_type == "gemini_coaching":
                            print(f"[inject] COACHING CUE received at {audio_pos:.1f}s")
                        elif evt_type == "gemini_greeting":
                            print(f"[inject] Gemini greeting received")
                        elif evt_type == "gemini_song_end":
                            print(f"[inject] Song-end summary received")
            except websockets.ConnectionClosed:
                pass

        listener = asyncio.create_task(listen_events())

        # Stream audio chunks
        audio_pos = 0.0  # Current position in seconds
        chunk_idx = 0
        start_time = time.monotonic()

        print(f"[inject] Streaming at {speed}x speed...")

        while chunk_idx * CHUNK_SAMPLES < len(audio):
            # Check for pause
            if pause_idx < len(pauses) and audio_pos >= pauses[pause_idx]["at"]:
                p = pauses[pause_idx]
                print(f"[inject] PAUSE: '{p.get('label', 'pause')}' — "
                      f"pausing {p['duration']}s at {audio_pos:.1f}s")

                # Send silence during pause (keeps WS alive, backend detects silence)
                silence_chunks = int(p["duration"] / chunk_duration)
                silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                for _ in range(silence_chunks):
                    await ws.send(silence.tobytes())
                    await asyncio.sleep(chunk_duration / speed)

                pause_idx += 1
                print(f"[inject] Resuming audio")
                continue

            # Extract chunk
            start_sample = chunk_idx * CHUNK_SAMPLES
            end_sample = min(start_sample + CHUNK_SAMPLES, len(audio))
            chunk = audio[start_sample:end_sample]

            # Pad last chunk if needed
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            # Send binary Float32 PCM (exactly as browser does)
            await ws.send(chunk.tobytes())

            audio_pos = start_sample / sr
            chunk_idx += 1

            # Real-time pacing (adjusted for speed multiplier)
            expected_time = (chunk_idx * chunk_duration) / speed
            elapsed = time.monotonic() - start_time
            sleep_time = expected_time - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Progress indicator every 5 seconds
            if chunk_idx % int(5 / chunk_duration) == 0:
                zone = event_count.get("ears_frame", 0)
                print(f"[inject] {audio_pos:.1f}s / {total_duration:.1f}s "
                      f"({audio_pos/total_duration*100:.0f}%) — "
                      f"{event_count.get('ears_frame', 0)} frames processed")

        # Send trailing silence to trigger song-end summary
        print(f"[inject] Audio complete. Sending 6s silence for song-end trigger...")
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        silence_chunks = int(6.0 / chunk_duration)
        for i in range(silence_chunks):
            await ws.send(silence.tobytes())
            await asyncio.sleep(chunk_duration / speed)

        # Wait for song-end summary
        print(f"[inject] Waiting for Gemini song-end summary (up to 15s)...")
        await asyncio.sleep(15)

        # Summary
        listener.cancel()
        print(f"\n[inject] Done!")
        print(f"  Total audio: {total_duration:.1f}s")
        print(f"  Frames processed: {event_count.get('ears_frame', 0)}")
        print(f"  Coaching cues: {event_count.get('gemini_coaching', 0)}")
        print(f"  Events: {dict(event_count)}")


async def inject_with_capture(ws_url: str, audio_path: str, pauses: list | None = None,
                               speed: float = 1.0, event_log_path: str | None = None):
    """Inject audio and capture all backend events to a JSONL file.

    Useful for building the timeline — run once, capture all events with timestamps,
    then use the event log to place audio/overlays in the composition phase.
    """
    events = []

    def log_event(evt):
        evt["_capture_time"] = time.time()
        events.append(evt)

    await inject(ws_url, audio_path, pauses, speed, on_event=log_event)

    if event_log_path:
        with open(event_log_path, "w") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")
        print(f"[inject] Saved {len(events)} events to {event_log_path}")

    return events


def main():
    parser = argparse.ArgumentParser(description="Inject audio into Koda via WebSocket")
    parser.add_argument("audio", help="Path to audio file (WAV/FLAC/OGG)")
    parser.add_argument("--ws", default="ws://localhost:8000/inject/ws",
                        help="WebSocket URL (default: ws://localhost:8000/inject/ws)")
    parser.add_argument("--pauses", help="Path to pauses JSON file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--capture", help="Path to save event log (JSONL)")

    args = parser.parse_args()

    pauses = None
    if args.pauses:
        pauses = json.loads(Path(args.pauses).read_text())

    if args.capture:
        asyncio.run(inject_with_capture(args.ws, args.audio, pauses, args.speed, args.capture))
    else:
        asyncio.run(inject(args.ws, args.audio, pauses, args.speed))


if __name__ == "__main__":
    main()
