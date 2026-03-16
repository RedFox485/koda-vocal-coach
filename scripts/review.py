#!/usr/bin/env python3
"""
Segment-by-segment vocal strain annotation tool.

Plays short audio segments and waits for your G/Y/R rating after each one.
No time pressure — replay any segment, skip uncertain ones.

Controls:
  G / 1  →  GREEN  (relaxed, clean)
  Y / 2  →  YELLOW (moderate strain)
  R / 3  →  RED    (pushing hard)
  SPACE  →  replay segment (or press any time during playback to rate mid-clip)
  S      →  skip / uncertain (excluded from calibration)
  Q      →  quit and save progress

Output: docs/annotations/<filename>_reviewed.json
  Compatible with compare.py: python scripts/compare.py "<name>_reviewed"

Usage:
  python scripts/review.py                              # pick from list
  python scripts/review.py "Danny - Chris Young R1"     # by name
  python scripts/review.py --segment 1.5                # 1.5s segments (default: 2.0)
  python scripts/review.py --segment 1.0                # 1s = max precision
"""
import os
import sys
import tty
import termios
import select
import json
import time
import argparse
import subprocess
import tempfile
import threading

import numpy as np
import sounddevice as sd

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR  = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "docs", "annotations")
SAMPLE_RATE     = 44100

ZONE_COLORS = {
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "skip":   "\033[90m",
}
RESET      = "\033[0m"
BOLD       = "\033[1m"
CLEAR_LINE = "\033[2K\r"

ZONE_LABELS = {
    "green":  "🟢 GREEN ",
    "yellow": "🟡 YELLOW",
    "red":    "🔴 RED   ",
    "skip":   "⬜ SKIP  ",
}


def load_audio(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
        capture_output=True, check=True
    )
    import librosa
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    return audio


def pick_file():
    files = sorted([
        f for f in os.listdir(RECORDINGS_DIR)
        if f.endswith((".m4a", ".wav", ".mp3", ".aac"))
    ])
    if not files:
        print("No recordings found in:", RECORDINGS_DIR)
        sys.exit(1)
    print("\nPick a recording:\n")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    print()
    while True:
        try:
            n = int(input("Enter number: ").strip())
            if 1 <= n <= len(files):
                return os.path.join(RECORDINGS_DIR, files[n - 1])
        except (ValueError, EOFError):
            pass
        print("  Invalid — enter a number 1 to", len(files))


def get_char(fd):
    """Block until a keypress, return char."""
    select.select([fd], [], [], None)
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        if select.select([fd], [], [], 0.02)[0]:
            os.read(fd, 2)  # consume escape sequence bytes
        return "ESC"
    return ch.decode("utf-8", errors="ignore")


def get_char_nonblocking(fd):
    """Return a char if one is buffered, else None."""
    if select.select([fd], [], [], 0)[0]:
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            if select.select([fd], [], [], 0.02)[0]:
                os.read(fd, 2)
            return "ESC"
        return ch.decode("utf-8", errors="ignore")
    return None


def play_segment_interruptible(chunk, fd):
    """Play chunk; return any keypress that interrupted it (or None if played fully)."""
    play_done = threading.Event()

    def _play():
        sd.play(chunk, samplerate=SAMPLE_RATE)
        sd.wait()
        play_done.set()

    t = threading.Thread(target=_play, daemon=True)
    t.start()

    key_during = None
    while not play_done.is_set():
        ch = get_char_nonblocking(fd)
        if ch:
            key_during = ch
            sd.stop()
            play_done.set()
            break
        time.sleep(0.01)

    t.join(timeout=0.2)
    return key_during


def mini_waveform(chunk, width=36):
    """ASCII amplitude visualization of the chunk."""
    n = len(chunk)
    chars = " ▁▂▃▄▅▆▇█"
    bar = []
    for i in range(width):
        s = int(i / width * n)
        e = int((i + 1) / width * n)
        rms = float(np.sqrt(np.mean(chunk[s:e] ** 2))) if e > s else 0.0
        bar.append(chars[min(int(rms * 220), 8)])
    return "".join(bar)


def review(audio_path, segment_s=2.0):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    name = os.path.basename(audio_path)
    stem = os.path.splitext(name)[0]
    out_path = os.path.join(ANNOTATIONS_DIR, stem + "_reviewed.json")

    print(f"\nLoading {name}...")
    audio    = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s  |  Segment size: {segment_s}s\n")

    seg_samples = int(segment_s * SAMPLE_RATE)
    n_segs      = int(np.ceil(len(audio) / seg_samples))
    print(f"Total segments: {n_segs}  (~{n_segs * segment_s / 60:.1f} min of listening)")
    print()
    print(f"{BOLD}Controls:{RESET}")
    print("  G / 1  →  GREEN  (relaxed)")
    print("  Y / 2  →  YELLOW (moderate strain)")
    print("  R / 3  →  RED    (pushing hard)")
    print("  SPACE  →  replay segment")
    print("  S      →  skip / uncertain")
    print("  Q      →  quit and save")
    print()
    input("Press ENTER to start...")

    ratings = []  # {seg, t_start, t_end, zone, replays}

    fd           = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        print("\033[?25l", end="", flush=True)  # hide cursor

        for seg_idx in range(n_segs):
            t_start = seg_idx * segment_s
            t_end   = min(t_start + segment_s, duration)
            s0      = seg_idx * seg_samples
            chunk   = audio[s0: s0 + seg_samples]

            if len(chunk) == 0:
                break

            # Auto-skip silent segments
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.004:
                ratings.append({
                    "seg": seg_idx, "t_start": round(t_start, 2),
                    "t_end": round(t_end, 2), "zone": None, "skip_reason": "silent"
                })
                sys.stdout.write(CLEAR_LINE)
                sys.stdout.write(
                    f"  [{seg_idx+1:3d}/{n_segs}]  {t_start:5.1f}–{t_end:.1f}s  "
                    f"\033[90m(silent — skipped)\033[0m"
                )
                sys.stdout.flush()
                continue

            wave    = mini_waveform(chunk)
            replays = 0
            zone    = None

            while True:
                # Play (interruptible — key during playback registers immediately)
                rated   = sum(1 for r in ratings if r["zone"] is not None)
                skipped = sum(1 for r in ratings if r["zone"] is None and r.get("skip_reason") != "silent")
                replay_tag = f" (replay #{replays})" if replays > 0 else ""

                sys.stdout.write(CLEAR_LINE)
                sys.stdout.write(
                    f"  [{seg_idx+1:3d}/{n_segs}]  {t_start:5.1f}–{t_end:.1f}s  "
                    f"\033[90m▶{replay_tag}\033[0m  {wave}  "
                    f"\033[90m({rated} rated, {skipped} skipped)\033[0m"
                )
                sys.stdout.flush()

                ch = play_segment_interruptible(chunk, fd)

                # If no key pressed during playback, show prompt and wait
                if ch is None:
                    sys.stdout.write(CLEAR_LINE)
                    sys.stdout.write(
                        f"  [{seg_idx+1:3d}/{n_segs}]  {t_start:5.1f}–{t_end:.1f}s  "
                        f"\033[93mG / Y / R / SPACE(replay) / S(skip) / Q\033[0m"
                    )
                    sys.stdout.flush()
                    ch = get_char(fd)

                ch_lower = ch.lower()

                if ch_lower in ("g", "1"):
                    zone = "green"
                elif ch_lower in ("y", "2"):
                    zone = "yellow"
                elif ch_lower in ("r", "3"):
                    zone = "red"
                elif ch == " ":
                    replays += 1
                    continue  # replay
                elif ch_lower == "s":
                    zone = None  # intentional skip
                elif ch_lower == "q" or ch == "\x03":
                    raise KeyboardInterrupt
                else:
                    continue  # unrecognized, wait again

                # Show confirmation and move on
                if zone:
                    c     = ZONE_COLORS[zone]
                    label = ZONE_LABELS[zone]
                    replay_note = f"  \033[90m({replays} replays)\033[0m" if replays else ""
                    sys.stdout.write(CLEAR_LINE)
                    sys.stdout.write(
                        f"  [{seg_idx+1:3d}/{n_segs}]  {t_start:5.1f}–{t_end:.1f}s  "
                        f"{c}{BOLD}{label}{RESET}{replay_note}\n"
                    )
                else:
                    sys.stdout.write(CLEAR_LINE)
                    sys.stdout.write(
                        f"  [{seg_idx+1:3d}/{n_segs}]  {t_start:5.1f}–{t_end:.1f}s  "
                        f"\033[90m⬜ skipped\033[0m\n"
                    )
                sys.stdout.flush()

                ratings.append({
                    "seg":     seg_idx,
                    "t_start": round(t_start, 2),
                    "t_end":   round(t_end, 2),
                    "zone":    zone,
                    "replays": replays,
                })
                break

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\033[?25h", end="", flush=True)  # show cursor

    print()

    # ── Convert to compare.py-compatible annotation format ──────────────────────
    # Emit zone-transition events + dense 100ms "auto" events for rated segments.
    SAMPLE_INTERVAL = 0.1
    annotations = []
    prev_zone   = None

    for r in ratings:
        if r["zone"] is None:
            prev_zone = None  # gap resets zone
            continue
        z        = r["zone"]
        t0, t1   = r["t_start"], r["t_end"]
        ev_type  = "start" if prev_zone is None else "change"
        if ev_type == "start" or z != prev_zone:
            annotations.append({"t": round(t0, 2), "zone": z, "type": ev_type})
        # Dense auto events within rated segment
        t = t0
        while t < t1 - 0.01:
            annotations.append({"t": round(t, 2), "zone": z, "type": "auto"})
            t = round(t + SAMPLE_INTERVAL, 3)
        prev_zone = z

    result = {
        "file":            name,
        "duration":        round(duration, 2),
        "annotations":     sorted(annotations, key=lambda x: x["t"]),
        "created":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "method":          f"segment_review_{segment_s}s",
        "segment_ratings": ratings,   # raw segment data preserved for reference
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────────
    rated    = [r for r in ratings if r["zone"] is not None]
    skipped  = [r for r in ratings if r["zone"] is None and r.get("skip_reason") != "silent"]
    silent   = [r for r in ratings if r.get("skip_reason") == "silent"]

    g_s = sum(r["t_end"] - r["t_start"] for r in rated if r["zone"] == "green")
    y_s = sum(r["t_end"] - r["t_start"] for r in rated if r["zone"] == "yellow")
    r_s = sum(r["t_end"] - r["t_start"] for r in rated if r["zone"] == "red")
    labeled_s = g_s + y_s + r_s

    print(f"  {BOLD}REVIEW SUMMARY{RESET}")
    print(f"  {len(ratings)} segments  ({len(rated)} rated, {len(skipped)} skipped by you, {len(silent)} silent auto-skipped)")
    print(f"  Labeled: {labeled_s:.1f}s of {duration:.1f}s ({labeled_s/duration*100:.0f}%)")
    if rated:
        avg_replays = sum(r["replays"] for r in rated) / len(rated)
        print(f"  🟢 {g_s/duration*100:.0f}%  ({g_s:.1f}s)")
        print(f"  🟡 {y_s/duration*100:.0f}%  ({y_s:.1f}s)")
        print(f"  🔴 {r_s/duration*100:.0f}%  ({r_s:.1f}s)")
        print(f"  Avg replays per segment: {avg_replays:.1f}")

    print(f"\n  Saved: {out_path}")
    print(f"  Compare: python scripts/compare.py \"{stem}_reviewed\"\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Segment-by-segment vocal strain annotation")
    parser.add_argument("file", nargs="?", help="Recording name or path (partial match ok)")
    parser.add_argument("--segment", type=float, default=2.0,
                        help="Segment length in seconds (default: 2.0, use 1.0 for max precision)")
    args = parser.parse_args()

    if args.file:
        path = args.file
        if not os.path.exists(path):
            # Try recordings dir
            candidate = os.path.join(RECORDINGS_DIR, path)
            if os.path.exists(candidate):
                path = candidate
            else:
                # Partial name match
                for f in os.listdir(RECORDINGS_DIR):
                    if args.file.lower() in f.lower():
                        path = os.path.join(RECORDINGS_DIR, f)
                        break
                # Try adding extension
                if not os.path.exists(path):
                    for ext in (".m4a", ".wav", ".mp3"):
                        if os.path.exists(path + ext):
                            path = path + ext
                            break
    else:
        path = pick_file()

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    review(path, segment_s=args.segment)


if __name__ == "__main__":
    main()
