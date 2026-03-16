#!/usr/bin/env python3
"""
Real-time vocal strain annotation tool.
Play a recording and label strain level as you listen.

Controls:
  G  or  1  →  GREEN  (relaxed, clean)
  Y  or  2  →  YELLOW  (moderate strain)
  R  or  3  →  RED    (heavy strain / pushing)
  SPACE      →  drop a marker (logged with current zone)
  Q / Ctrl-C →  quit and save

Output: docs/annotations/<filename>.json
Compare against strain_chart.py output with: python scripts/compare.py <file>

Usage:
  python scripts/annotate.py                              # pick from list
  python scripts/annotate.py "Danny - Chris Young R1.m4a"
  python scripts/annotate.py path/to/file.m4a
"""
import os
import sys
import tty
import termios
import select
import json
import time
import threading
import subprocess
import tempfile

import numpy as np
import sounddevice as sd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "docs", "annotations")
SAMPLE_RATE = 44100

ZONE_COLORS = {
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "idle":   "\033[90m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
CLEAR_LINE = "\033[2K\r"

ZONE_LABELS = {"green": "🟢 GREEN ", "yellow": "🟡 YELLOW", "red": "🔴 RED   ", "idle": "⬜ idle  "}


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


def progress_bar(pos, total, width=40):
    filled = int(pos / total * width) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return bar


def get_char_nonblocking(fd):
    """Return a character if one is available, else None. Handles arrow keys."""
    if select.select([fd], [], [], 0)[0]:
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # Possible arrow key: read more
            if select.select([fd], [], [], 0.02)[0]:
                seq = os.read(fd, 2)
                if seq == b"[A":
                    return "UP"
                elif seq == b"[B":
                    return "DOWN"
                elif seq == b"[C":
                    return "RIGHT"
                elif seq == b"[D":
                    return "LEFT"
            return "ESC"
        return ch.decode("utf-8", errors="ignore")
    return None


def annotate(audio_path):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    name = os.path.basename(audio_path)
    out_path = os.path.join(ANNOTATIONS_DIR, os.path.splitext(name)[0] + ".json")

    print(f"\nLoading {name}...")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s\n")

    print(f"{BOLD}Controls:{RESET}")
    print("  G / 1  →  set GREEN  (relaxed — stays green until you change it)")
    print("  Y / 2  →  set YELLOW (moderate strain)")
    print("  R / 3  →  set RED    (pushing hard)")
    print("  SPACE  →  pause / resume sampling (gap in data)")
    print("  Q      →  quit and save")
    print()
    print("  Tip: press G when it starts, then tap Y/R when you feel strain shift.")
    print()
    input("Press ENTER to start playback...")

    # State — set-and-persist model
    current_zone = "idle"   # idle = not yet started
    paused = False          # SPACE toggles sampling off (intentional gap)
    annotations = []
    last_sample_t = -1.0
    SAMPLE_INTERVAL = 0.1   # sample every 100ms while active

    # Play audio in background
    play_start = time.time()
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    stream.start()

    play_thread_done = threading.Event()
    pos_lock = threading.Lock()
    play_position = [0]  # mutable via list

    def audio_thread():
        chunk = 4096
        i = 0
        while i < len(audio):
            data = audio[i:i+chunk].reshape(-1, 1)
            stream.write(data)
            with pos_lock:
                play_position[0] = i
            i += chunk
        with pos_lock:
            play_position[0] = len(audio)
        play_thread_done.set()

    t = threading.Thread(target=audio_thread, daemon=True)
    t.start()

    # Terminal raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        print("\033[?25l", end="", flush=True)  # hide cursor

        while not play_thread_done.is_set():
            with pos_lock:
                pos = play_position[0]
            elapsed = pos / SAMPLE_RATE

            # Continuous sampling while active and not paused
            sampling = current_zone != "idle" and not paused
            if sampling and elapsed - last_sample_t >= SAMPLE_INTERVAL:
                annotations.append({
                    "t": round(elapsed, 2),
                    "zone": current_zone,
                    "type": "auto"
                })
                last_sample_t = elapsed

            # Draw display
            bar = progress_bar(elapsed, duration)
            zcolor = ZONE_COLORS.get(current_zone, "")
            zlabel = ZONE_LABELS.get(current_zone, current_zone)
            n_auto = len([a for a in annotations if a["type"] == "auto"])
            pct_labeled = elapsed / duration * 100 if duration > 0 else 0

            if paused:
                status = "\033[90m⏸ PAUSED\033[0m"
            elif current_zone == "idle":
                status = "\033[90mpress G to start\033[0m"
            else:
                status = f"{zcolor}{BOLD}{zlabel}{RESET} \033[32m●\033[0m"

            sys.stdout.write(CLEAR_LINE)
            sys.stdout.write(
                f"  {elapsed:5.1f}s / {duration:.1f}s  [{bar}]  "
                f"{status}  ({n_auto} pts)"
            )
            sys.stdout.flush()

            # Check key
            ch = get_char_nonblocking(fd)
            if ch:
                ch_lower = ch.lower()
                new_zone = None
                if ch_lower in ("g", "1"):
                    new_zone = "green"
                elif ch_lower in ("y", "2"):
                    new_zone = "yellow"
                elif ch_lower in ("r", "3"):
                    new_zone = "red"
                elif ch == "UP":
                    zone_list = ["green", "yellow", "red"]
                    idx = zone_list.index(current_zone) if current_zone in zone_list else 0
                    new_zone = zone_list[min(idx + 1, 2)]
                elif ch == "DOWN":
                    zone_list = ["green", "yellow", "red"]
                    idx = zone_list.index(current_zone) if current_zone in zone_list else 2
                    new_zone = zone_list[max(idx - 1, 0)]
                elif ch == " ":
                    if current_zone != "idle":
                        paused = not paused
                        annotations.append({
                            "t": round(elapsed, 2),
                            "zone": current_zone,
                            "type": "pause" if paused else "resume"
                        })
                        last_sample_t = elapsed
                elif ch_lower == "q" or ch == "\x03":
                    break

                if new_zone:
                    paused = False  # any zone key resumes if paused
                    if new_zone != current_zone:
                        ev_type = "start" if current_zone == "idle" else "change"
                        annotations.append({"t": round(elapsed, 2), "zone": new_zone, "type": ev_type})
                        current_zone = new_zone
                        last_sample_t = elapsed

            time.sleep(0.05)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\033[?25h", end="", flush=True)  # show cursor
        stream.stop()
        stream.close()

    print()

    # Save
    result = {
        "file": name,
        "duration": round(duration, 2),
        "annotations": sorted(annotations, key=lambda x: x["t"]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary — % of TOTAL song duration (gaps count as unlabeled)
    auto = [a for a in annotations if a["type"] == "auto"]
    print(f"\n  {BOLD}ANNOTATION SUMMARY{RESET}")
    if auto:
        total_pts = len(auto)
        g_pts = sum(1 for a in auto if a["zone"] == "green")
        y_pts = sum(1 for a in auto if a["zone"] == "yellow")
        r_pts = sum(1 for a in auto if a["zone"] == "red")

        # % of total song (each sample = SAMPLE_INTERVAL seconds)
        labeled_s = total_pts * SAMPLE_INTERVAL
        g_s = g_pts * SAMPLE_INTERVAL
        y_s = y_pts * SAMPLE_INTERVAL
        r_s = r_pts * SAMPLE_INTERVAL
        gap_s = max(0.0, duration - labeled_s)

        g_pct = g_s / duration * 100
        y_pct = y_s / duration * 100
        r_pct = r_s / duration * 100
        gap_pct = gap_s / duration * 100

        print(f"  Duration: {duration:.1f}s  |  Labeled: {labeled_s:.1f}s ({100-gap_pct:.0f}%)")
        print(f"  🟢 {g_pct:.0f}%  ({g_s:.1f}s)")
        print(f"  🟡 {y_pct:.0f}%  ({y_s:.1f}s)")
        print(f"  🔴 {r_pct:.0f}%  ({r_s:.1f}s)")
        if gap_pct > 5:
            print(f"  ⬜ {gap_pct:.0f}%  ({gap_s:.1f}s) unlabeled gaps")
    else:
        print("  (no zones recorded — press G/Y/R to set zone at the start)")
    print(f"\n  Saved: {out_path}\n")

    return out_path


def main():
    args = sys.argv[1:]
    if args:
        path = args[0]
        if not os.path.isabs(path):
            # Try recordings dir
            candidate = os.path.join(RECORDINGS_DIR, path)
            if os.path.exists(candidate):
                path = candidate
    else:
        path = pick_file()

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    annotate(path)


if __name__ == "__main__":
    main()
