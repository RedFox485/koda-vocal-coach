#!/usr/bin/env python3
"""
Live mic capture → backend WebSocket → frame log.

Records from mic in real-time, streams to the vocal health backend,
logs every frame to JSONL + saves raw audio to WAV.

After the session ends, prints the same analysis table as stream_test.py
so Claude can read /tmp/koda_live_frames.jsonl and diagnose issues.

Usage:
    .venv/bin/python3 scripts/live_capture.py
    .venv/bin/python3 scripts/live_capture.py --device "iPhone"  # use iPhone mic
    .venv/bin/python3 scripts/live_capture.py --duration 30      # auto-stop after 30s
    .venv/bin/python3 scripts/live_capture.py --replay           # replay last session

Press ENTER to stop recording.
"""

import argparse
import asyncio
import json
import math
import queue
import sys
import time
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

SR         = 44100
CHUNK_S    = 0.1
CHUNK_SAMP = int(SR * CHUNK_S)
WS_URL     = "ws://localhost:8765/ws"

FRAMES_LOG  = Path("/tmp/koda_live_frames.jsonl")
AUDIO_OUT   = Path("/tmp/koda_live_audio.wav")
SUMMARY_OUT = Path("/tmp/koda_live_summary.json")

STRAIN_GREEN  = 0.40
STRAIN_YELLOW = 0.60
ZONE_ORDER    = {"green": 0, "yellow": 1, "red": 2}

RESET = "\033[0m"
COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "idle":   "\033[90m",
}

def zone_of(score: float) -> str:
    if score < STRAIN_GREEN:  return "green"
    if score < STRAIN_YELLOW: return "yellow"
    return "red"

def cz(z, txt=None):
    return f"{COLORS.get(z,'')}{txt or z}{RESET}"


# ─── Mic capture ─────────────────────────────────────────────────────────────

audio_queue: queue.Queue = queue.Queue()
all_audio: list = []

def mic_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block."""
    if status:
        print(f"[mic] {status}", file=sys.stderr)
    chunk = indata[:, 0].copy()  # mono
    audio_queue.put(chunk)
    all_audio.append(chunk)


def find_device(name_fragment: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and name_fragment.lower() in d['name'].lower():
            return i
    return None


# ─── WebSocket receiver ───────────────────────────────────────────────────────

async def receiver(ws, done_event: asyncio.Event, log_fh):
    """Receive frames, print table, write JSONL log."""
    stats = {"total": 0, "active": 0, "idle": 0, "yellow_red": 0}
    response_count = 0

    print(f"\n{'Time':>6}  {'RMS':>7}  {'Score':>6}  {'Zone':<8}  "
          f"{'shim':>6}  {'cpp':>5}  {'v8':>5}  {'e11':>5}  {'ph':>5}  {'flags'}")
    print("─" * 80)

    try:
        while not done_event.is_set() or True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                if done_event.is_set():
                    break
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t    = msg.get("type", "")
            sess = msg.get("session_t", 0.0)

            if t == "phrase_start":
                print(f"\033[2m  ── PHRASE START at {sess:.1f}s ──\033[0m")
                continue

            if t == "phrase_end":
                avg  = msg.get("average_strain", 0)
                pk   = msg.get("peak_strain", 0)
                dur  = msg.get("duration_s", 0)
                zone = msg.get("zone", "?")
                print(f"\033[2m  ── PHRASE END  at {sess:.1f}s  "
                      f"avg={avg:.3f}  peak={pk:.3f}  dur={dur:.1f}s  zone={zone} ──\033[0m")
                continue

            if t == "gemini_coaching":
                txt = msg.get("text", "")
                print(f"\033[95m  ── GEMINI: {txt[:80]} ──\033[0m")
                continue

            if t != "ears_frame":
                continue

            stats["total"] += 1
            active      = msg.get("active", False)
            rms         = msg.get("rms", 0.0)
            score       = msg.get("strain_score", 0.0)
            shimmer     = msg.get("shimmer_pct", float('nan'))
            cpp         = msg.get("cpp_db", float('nan'))
            v8          = msg.get("v8_strain", score)
            ears11      = msg.get("ears_v11", 0.0)
            phonation   = msg.get("phonation_score", 0.0)
            onset_gated = msg.get("onset_gated", False)
            low_energy  = msg.get("low_energy", False)

            # Estimate song time from response count (ring buffer = 10 frames)
            song_t = (response_count + 10) * 0.1
            response_count += 1

            # Enrich and log
            frame_data = {
                **msg,
                "_song_t": round(song_t, 2),
                "_response_idx": response_count - 1,
            }
            log_fh.write(json.dumps(frame_data) + "\n")
            log_fh.flush()

            if not active:
                stats["idle"] += 1
                print(f"\033[2m{song_t:>5.1f}s  {rms:>7.4f}  {'—':>6}  idle\033[0m")
                continue

            stats["active"] += 1
            pred_zone = zone_of(score)
            if pred_zone in ("yellow", "red"):
                stats["yellow_red"] += 1

            shim_str = f"{shimmer:>6.1f}" if not math.isnan(shimmer) else f"   nan"
            cpp_str  = f"{cpp:>5.3f}"     if not math.isnan(cpp)     else f"  nan"
            flags    = ("O" if onset_gated else "") + ("L" if low_energy else "")

            print(f"{song_t:>5.1f}s  {rms:>7.4f}  {score:>6.3f}  "
                  f"{cz(pred_zone, f'{pred_zone:<8}')}  "
                  f"{shim_str}  {cpp_str}  {v8:>5.3f}  {ears11:>5.3f}  "
                  f"{phonation:>5.3f}  {flags}")

    except websockets.exceptions.ConnectionClosed:
        pass

    # Print summary
    print("\n" + "═" * 80)
    t = stats["active"]
    yr_pct = int(100 * stats["yellow_red"] / t) if t else 0
    print(f"  Active frames: {t}  |  Strain triggered (≥yellow): {stats['yellow_red']} ({yr_pct}%)")
    print(f"  Idle frames:   {stats['idle']}")
    print(f"  Log:  {FRAMES_LOG}")
    print(f"  Audio: {AUDIO_OUT}")
    print("═" * 80 + "\n")

    # Save summary
    SUMMARY_OUT.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "total_frames": response_count,
    }, indent=2))

    return stats


# ─── Sender: drain audio_queue and send to WebSocket ─────────────────────────

async def sender(ws, stop_event: asyncio.Event):
    """Pull chunks from audio_queue and send to backend."""
    loop = asyncio.get_event_loop()
    while not stop_event.is_set():
        try:
            # Non-blocking get with asyncio yield
            chunk = await loop.run_in_executor(None, _get_chunk)
            if chunk is not None:
                payload = chunk.astype(np.float32).tobytes()
                await ws.send(payload)
        except Exception as e:
            if not stop_event.is_set():
                print(f"[sender] {e}", file=sys.stderr)
            break

def _get_chunk():
    import queue as q
    try:
        return audio_queue.get(timeout=0.1)
    except q.Empty:
        return None


# ─── ENTER key watcher ────────────────────────────────────────────────────────

async def wait_for_enter(stop_event: asyncio.Event, duration: float | None):
    """Set stop_event when user presses ENTER (or duration elapses)."""
    loop = asyncio.get_event_loop()
    if duration:
        print(f"  Auto-stopping in {duration:.0f}s — or press ENTER to stop early\n")
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=duration
            )
        except asyncio.TimeoutError:
            print("\n[auto-stop reached]")
    else:
        print("  Press ENTER to stop recording...\n")
        await loop.run_in_executor(None, sys.stdin.readline)
    stop_event.set()


# ─── Replay last session ──────────────────────────────────────────────────────

async def replay_session():
    """Re-stream saved audio through backend and show analysis."""
    if not AUDIO_OUT.exists():
        print(f"No saved audio found at {AUDIO_OUT}")
        return

    import librosa as _librosa
    print(f"Replaying: {AUDIO_OUT}")
    y, _ = _librosa.load(str(AUDIO_OUT), sr=SR, mono=True)
    frames = [y[i * CHUNK_SAMP:(i+1) * CHUNK_SAMP] for i in range(len(y) // CHUNK_SAMP)]
    print(f"Duration: {len(y)/SR:.1f}s → {len(frames)} frames\n")

    done = asyncio.Event()
    with open(FRAMES_LOG, "w") as log_fh:
        async with websockets.connect(WS_URL) as ws:
            try:
                cfg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                print(f"Server config: green<{cfg['strain_thresholds']['green']} "
                      f"yellow<{cfg['strain_thresholds']['yellow']}\n")
            except Exception:
                pass

            recv_task = asyncio.create_task(receiver(ws, done, log_fh))
            for chunk in frames:
                await ws.send(chunk.astype(np.float32).tobytes())
                await asyncio.sleep(CHUNK_S)
            await asyncio.sleep(1.5)
            done.set()
            await recv_task


# ─── Main ─────────────────────────────────────────────────────────────────────

async def run(device_name: str | None, duration: float | None):
    # Find device
    device_idx = None
    if device_name:
        device_idx = find_device(device_name)
        if device_idx is None:
            print(f"Device '{device_name}' not found. Available input devices:")
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    print(f"  [{i}] {d['name']}")
            return

    dev_name = sd.query_devices(device_idx or sd.default.device[0])['name']
    print(f"\nDevice: {dev_name}")
    print(f"Backend: {WS_URL}")
    print(f"Frame log: {FRAMES_LOG}")
    print(f"Audio out: {AUDIO_OUT}\n")

    stop_event   = asyncio.Event()
    done_event   = asyncio.Event()

    # Open log file
    log_fh = open(FRAMES_LOG, "w")

    try:
        async with websockets.connect(WS_URL) as ws:
            # Drain config
            try:
                cfg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                print(f"Server: green<{cfg['strain_thresholds']['green']}  "
                      f"yellow<{cfg['strain_thresholds']['yellow']}\n")
            except Exception:
                pass

            # Start mic stream
            stream = sd.InputStream(
                samplerate=SR,
                channels=1,
                dtype='float32',
                blocksize=CHUNK_SAMP,
                device=device_idx,
                callback=mic_callback,
            )
            stream.start()
            print("Recording started.")

            # Run all tasks concurrently
            recv_task   = asyncio.create_task(receiver(ws, done_event, log_fh))
            send_task   = asyncio.create_task(sender(ws, stop_event))
            enter_task  = asyncio.create_task(wait_for_enter(stop_event, duration))

            # Wait for stop signal
            await stop_event.wait()
            stream.stop()
            stream.close()

            # Drain remaining queue
            await asyncio.sleep(1.5)
            done_event.set()

            send_task.cancel()
            enter_task.cancel()
            await recv_task

    finally:
        log_fh.close()

    # Save audio
    if all_audio:
        audio_arr = np.concatenate(all_audio).astype(np.float32)
        sf.write(str(AUDIO_OUT), audio_arr, SR)
        print(f"Saved {len(audio_arr)/SR:.1f}s of audio to {AUDIO_OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live mic capture → backend analysis")
    p.add_argument("--device",   type=str,   default=None,  help="Mic name fragment (e.g. 'iPhone')")
    p.add_argument("--duration", type=float, default=None,  help="Auto-stop after N seconds")
    p.add_argument("--replay",   action="store_true",       help="Replay last saved session")
    args = p.parse_args()

    if args.replay:
        asyncio.run(replay_session())
    else:
        asyncio.run(run(args.device, args.duration))
