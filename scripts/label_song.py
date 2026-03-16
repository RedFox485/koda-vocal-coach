#!/usr/bin/env python3
"""
Manual song labeler — plays 2s chunks, you label each one.
Saves labels to a JSON ground-truth file for validation.

Controls:
  g  = GREEN  (healthy, easy, no strain)
  y  = YELLOW (light strain, some push)
  r  = RED    (clear strain, pushed, rough)
  s  = SKIP   (transition, breath, unclear — excluded from validation)
  b  = BACK   (redo previous chunk)
  q  = QUIT   (saves progress, resume later)

Usage:
    .venv/bin/python3 scripts/label_song.py --song lizajane
    .venv/bin/python3 scripts/label_song.py --song chrisyoung
    .venv/bin/python3 scripts/label_song.py --song runnin
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 44100
CHUNK_S = 2.0          # seconds per chunk
CHUNK_SAMPLES = int(SR * CHUNK_S)
SILENCE_RMS = 0.008

SONGS = {
    "lizajane": "Vocal test recording sessions/Danny - Liza Jane R1 (longer).m4a",
    "chrisyoung": "Vocal test recording sessions/Danny - Chris Young R1.m4a",
    "runnin": "Vocal test recording sessions/Danny - Runnin down a dream R1.m4a",
}

LABELS_DIR = Path("data/ground_truth")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

ZONE_COLOR = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "skip":   "\033[90m",
    "reset":  "\033[0m",
}


def play_wav(wav_path: str):
    """Play a wav file using macOS afplay (non-blocking start, we wait manually)."""
    proc = subprocess.Popen(["afplay", wav_path])
    return proc


def chunk_label_to_zone(label):
    return {"g": "green", "y": "yellow", "r": "red", "s": "skip"}.get(label)


def load_labels(labels_path: Path):
    if labels_path.exists():
        return json.loads(labels_path.read_text())
    return {"song": "", "chunks": {}}


def save_labels(labels_path: Path, data: dict):
    labels_path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True, choices=list(SONGS.keys()))
    ap.add_argument("--chunk-s", type=float, default=CHUNK_S,
                    help="Chunk size in seconds (default 2)")
    ap.add_argument("--replay", action="store_true",
                    help="Force re-label all chunks even if already labeled")
    args = ap.parse_args()

    song_path = Path(SONGS[args.song])
    if not song_path.exists():
        print(f"ERROR: {song_path} not found")
        sys.exit(1)

    labels_path = LABELS_DIR / f"{args.song}_labels.json"
    data = load_labels(labels_path)
    data["song"] = args.song
    data["path"] = str(song_path)
    if "chunks" not in data:
        data["chunks"] = {}

    chunk_s = args.chunk_s
    chunk_samples = int(SR * chunk_s)

    print(f"\nLoading {song_path.name}...")
    y, _ = librosa.load(str(song_path), sr=SR, mono=True)
    duration = len(y) / SR
    n_chunks = int(duration / chunk_s)

    print(f"Duration: {duration:.1f}s  →  {n_chunks} chunks of {chunk_s:.0f}s each")
    print(f"\nControls: [g]reen  [y]ellow  [r]ed  [s]kip  [b]ack  [q]uit")
    print(f"Labels save to: {labels_path}\n")

    # Show existing progress
    done = [k for k in data["chunks"] if data["chunks"][k] != "skip"]
    skipped = [k for k in data["chunks"] if data["chunks"][k] == "skip"]
    if data["chunks"]:
        print(f"Existing labels: {len(done)} labeled, {len(skipped)} skipped")
        dist = {}
        for v in data["chunks"].values():
            dist[v] = dist.get(v, 0) + 1
        for zone, count in sorted(dist.items()):
            c = ZONE_COLOR.get(zone, "")
            print(f"  {c}{zone:<8}{ZONE_COLOR['reset']}: {count}")
        print()

    i = 0
    play_proc = None

    while i < n_chunks:
        chunk_key = str(i)
        t_start = i * chunk_s
        t_end   = min((i + 1) * chunk_s, duration)

        # Skip already labeled (unless --replay)
        if not args.replay and chunk_key in data["chunks"]:
            i += 1
            continue

        chunk = y[i * chunk_samples : (i + 1) * chunk_samples]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        # Show context
        prev_labels = []
        for j in range(max(0, i - 3), i):
            prev_zone = data["chunks"].get(str(j), "?")
            c = ZONE_COLOR.get(prev_zone, "")
            prev_labels.append(f"{c}{prev_zone[0].upper()}{ZONE_COLOR['reset']}")
        context = " ".join(prev_labels) if prev_labels else "—"

        silence_note = "  [SILENCE]" if rms < SILENCE_RMS else ""
        print(f"\033[1m[{t_start:.0f}s–{t_end:.0f}s]  chunk {i+1}/{n_chunks}\033[0m  "
              f"prev: {context}{silence_note}")

        # Write chunk to temp wav and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, chunk, SR)

        if play_proc:
            play_proc.wait()   # wait for previous to finish

        play_proc = play_wav(tmp_path)
        play_start = time.time()

        # Get input while audio plays
        while True:
            try:
                raw = input(f"  label [g/y/r/s/b/q]: ").strip().lower()
            except EOFError:
                raw = "q"

            if raw == "q":
                if play_proc:
                    play_proc.terminate()
                save_labels(labels_path, data)
                print(f"\nSaved. Resume with: .venv/bin/python3 scripts/label_song.py --song {args.song}")
                sys.exit(0)

            elif raw == "b":
                if i > 0:
                    i -= 1
                    # Remove the previous label so it gets re-labeled
                    prev_key = str(i)
                    if prev_key in data["chunks"]:
                        del data["chunks"][prev_key]
                    if play_proc:
                        play_proc.terminate()
                    print(f"  → Going back to {i * chunk_s:.0f}s–{(i+1)*chunk_s:.0f}s")
                    break
                else:
                    print("  Already at start.")
                    continue

            elif raw in ("g", "y", "r", "s"):
                zone = chunk_label_to_zone(raw)
                data["chunks"][chunk_key] = zone
                c = ZONE_COLOR.get(zone, "")
                print(f"  → {c}{zone.upper()}{ZONE_COLOR['reset']}")

                # Wait for audio to finish if still playing
                elapsed = time.time() - play_start
                remaining = chunk_s - elapsed
                if remaining > 0.1 and play_proc:
                    play_proc.wait()

                i += 1
                break

            elif raw == "p":
                # Replay current chunk
                if play_proc:
                    play_proc.terminate()
                play_proc = play_wav(tmp_path)
                play_start = time.time()
                print("  ↺ Replaying...")

            else:
                print("  ? Use: g y r s b q  (or p to replay)")

    if play_proc:
        try:
            play_proc.wait(timeout=3)
        except Exception:
            play_proc.terminate()

    save_labels(labels_path, data)

    # Summary
    print(f"\n{'═'*50}")
    print(f"  LABELING COMPLETE: {args.song}")
    print(f"{'═'*50}")
    dist = {}
    for v in data["chunks"].values():
        dist[v] = dist.get(v, 0) + 1
    for zone in ("green", "yellow", "red", "skip"):
        count = dist.get(zone, 0)
        c = ZONE_COLOR.get(zone, "")
        bar = "█" * count
        print(f"  {c}{zone:<8}{ZONE_COLOR['reset']}  {count:>3}  {bar}")
    print(f"\n  Saved to: {labels_path}")
    print(f"\nNow run validation against your labels:")
    print(f"  .venv/bin/python3 scripts/validate_songs.py --song {args.song} --ground-truth")


if __name__ == "__main__":
    main()
