#!/usr/bin/env python3
"""
Stream a known audio file through the live backend WebSocket and print
what the system actually scores — with ground truth labels side by side.

Architecture note: the backend analysis loop fires asynchronously once the
1-second ring buffer is full (~10 chunks in). We use concurrent send/recv
tasks so we never deadlock waiting for a response that hasn't fired yet.

Usage:
    .venv/bin/python3 scripts/stream_test.py [--speed 1.0]
    --speed 2.0 = play at 2x (faster test, same analysis results)
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import librosa
import websockets

SR         = 44100
CHUNK_S    = 0.1          # 100ms per frame — matches backend
CHUNK_SAMP = int(SR * CHUNK_S)
WS_URL     = "ws://localhost:8765/ws"

GT_PATH    = Path("data/ground_truth/lizajane_labels.json")
AUDIO_PATH = Path("Vocal test recording sessions/Danny - Liza Jane R1 (longer).m4a")

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


async def sender(ws, frames: list, speed: float):
    """Send audio frames at real-ish time pace."""
    delay = CHUNK_S / speed
    for chunk in frames:
        payload = chunk.astype(np.float32).tobytes()
        await ws.send(payload)
        await asyncio.sleep(delay)
    # Signal receiver that we're done
    await asyncio.sleep(1.0)  # wait for final responses to arrive


async def receiver(ws, gt_chunks: dict, done_event: asyncio.Event):
    """Receive and print all events from the server."""
    stats = {"total": 0, "exact": 0, "over": 0, "under": 0, "idle_frames": 0}
    per_chunk_scores = {}   # chunk_idx → list of frame scores
    response_count = 0      # local counter — response N = song frame N = song_t N*0.1s

    print(f"\n{'Time':>6}  {'RMS':>7}  {'Score':>6}  {'Zone':<8}  {'GT':<8}  "
          f"{'Match':>5}  {'shim':>6}  {'cpp':>5}  {'v8':>5}  {'e11':>5}  {'ph':>5}  {'flags'}")
    print("─" * 90)

    try:
        while not done_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
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
                avg = msg.get("average_strain", 0)
                pk  = msg.get("peak_strain", 0)
                dur = msg.get("duration_s", 0)
                zone = msg.get("zone", "?")
                print(f"\033[2m  ── PHRASE END  at {sess:.1f}s  "
                      f"avg={avg:.3f}  peak={pk:.3f}  dur={dur:.1f}s  zone={zone} ──\033[0m")
                continue

            if t != "ears_frame":
                continue

            active      = msg.get("active", False)
            rms         = msg.get("rms", 0.0)
            score       = msg.get("strain_score", 0.0)
            shimmer     = msg.get("shimmer_pct", float('nan'))
            cpp         = msg.get("cpp_db", float('nan'))   # server key is cpp_db, not cpp_val
            v8          = msg.get("v8_strain", score)
            ears11      = msg.get("ears_v11", 0.0)
            phonation   = msg.get("phonation_score", 0.0)
            onset_gated = msg.get("onset_gated", False)
            low_energy  = msg.get("low_energy", False)

            # Map to song time using local response count — count ALL responses including idle.
            # Server sends one ears_frame per input chunk once ring buffer is full.
            # Ring buffer fills after 10 frames → first response = song position 1.0s (10 × 0.1s).
            song_t = (response_count + 10) * 0.1
            response_count += 1
            ci = int(song_t // 2)
            gt_label = gt_chunks.get(ci, "—")

            if not active:
                stats["idle_frames"] += 1
                print(f"\033[2m{song_t:>5.1f}s  {rms:>7.4f}  {'—':>6}  "
                      f"{'idle':<8}  {gt_label:<8}\033[0m")
                continue

            pred_zone = zone_of(score)

            # Per-chunk accumulation (keyed by song chunk index)
            if ci not in per_chunk_scores:
                per_chunk_scores[ci] = []
            per_chunk_scores[ci].append(score)
            stats["total"] += 0  # frame total tracked separately above

            # Frame-level match
            match_str = ""
            if gt_label in ZONE_ORDER and pred_zone in ZONE_ORDER:
                diff = ZONE_ORDER[pred_zone] - ZONE_ORDER[gt_label]
                stats["total"] += 1
                if diff == 0:
                    stats["exact"] += 1
                    match_str = f"\033[92m ✓\033[0m"
                elif diff > 0:
                    stats["over"] += 1
                    match_str = f"\033[91m↑{diff}\033[0m"
                else:
                    stats["under"] += 1
                    match_str = f"\033[94m↓{abs(diff)}\033[0m"

            flags = ("O" if onset_gated else "") + ("L" if low_energy else "")
            shim_str = f"{shimmer:>6.1f}" if not math.isnan(shimmer) else f"  nan"
            cpp_str  = f"{cpp:>5.3f}"     if not math.isnan(cpp)     else f"  nan"
            gt_col   = cz(gt_label, f"{gt_label:<8}") if gt_label in ZONE_ORDER else f"{gt_label:<8}"

            print(f"{song_t:>5.1f}s  {rms:>7.4f}  {score:>6.3f}  "
                  f"{cz(pred_zone, f'{pred_zone:<8}')}  {gt_col}  "
                  f"{match_str:>5}  {shim_str}  {cpp_str}  {v8:>5.3f}  {ears11:>5.3f}  {phonation:>5.3f}  {flags}")

    except websockets.exceptions.ConnectionClosed:
        pass

    # Summary
    print("\n" + "═" * 80)
    t = stats["total"]
    pct = 100 * stats['exact'] // t if t else 0
    print(f"  Frame-level  exact={stats['exact']}/{t} ({pct}%)  "
          f"over={stats['over']}  under={stats['under']}  idle={stats['idle_frames']}")

    # Chunk-level P80 summary (matches validate_songs logic)
    if per_chunk_scores and gt_chunks:
        chunk_exact = chunk_over = chunk_under = 0
        print(f"\n  {'Chunk':>5}  {'Time':>6}  {'GT':<8}  {'P80':>6}  {'Pred':<8}  {'Match':>5}")
        print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*5}")
        for ci in sorted(set(list(per_chunk_scores.keys()) + list(gt_chunks.keys()))):
            gt_label = gt_chunks.get(ci, "—")
            if gt_label == "skip" or gt_label not in ZONE_ORDER:
                continue
            scores = per_chunk_scores.get(ci, [])
            if not scores:
                print(f"  {ci:>5}  {ci*2:>3}-{(ci+1)*2:<3}s  {cz(gt_label,f'{gt_label:<8}')}  "
                      f"{'no data':>6}")
                continue
            p80 = float(np.percentile(scores, 80))
            pred = zone_of(p80)
            diff = ZONE_ORDER[pred] - ZONE_ORDER[gt_label]
            if diff == 0:
                chunk_exact += 1; m = "\033[92m✓\033[0m"
            elif diff > 0:
                chunk_over  += 1; m = f"\033[91m↑\033[0m"
            else:
                chunk_under += 1; m = f"\033[94m↓\033[0m"
            print(f"  {ci:>5}  {ci*2:>3}-{(ci+1)*2:<3}s  "
                  f"{cz(gt_label,f'{gt_label:<8}')}  {p80:>6.3f}  "
                  f"{cz(pred,f'{pred:<8}')}  {m}")
        ct = chunk_exact + chunk_over + chunk_under
        print(f"\n  Chunk P80    exact={chunk_exact}/{ct} ({100*chunk_exact//ct if ct else 0}%)  "
              f"over={chunk_over}  under={chunk_under}")
    print("═" * 80 + "\n")

    return stats


async def run(speed: float):
    gt = json.loads(GT_PATH.read_text())
    gt_chunks = {int(k): v for k, v in gt["chunks"].items()}

    print(f"Loading audio: {AUDIO_PATH}")
    y, _ = librosa.load(str(AUDIO_PATH), sr=SR, mono=True)
    frames = [y[i * CHUNK_SAMP:(i + 1) * CHUNK_SAMP] for i in range(len(y) // CHUNK_SAMP)]
    print(f"Duration: {len(y)/SR:.1f}s  →  {len(frames)} frames  speed={speed}x")
    print(f"Ring buffer fills after 10 frames (~1s) — first responses arrive ~1.1s in\n")

    done = asyncio.Event()

    async with websockets.connect(WS_URL) as ws:
        # Drain config message
        try:
            cfg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
            print(f"Server config: thresholds green<{cfg['strain_thresholds']['green']} "
                  f"yellow<{cfg['strain_thresholds']['yellow']}\n")
        except Exception:
            pass

        recv_task = asyncio.create_task(receiver(ws, gt_chunks, done))
        await sender(ws, frames, speed)
        done.set()
        await recv_task


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = p.parse_args()
    asyncio.run(run(args.speed))
