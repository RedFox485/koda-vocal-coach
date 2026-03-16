#!/usr/bin/env python3
"""
CVT Dataset Inspector — run this BEFORE training to validate labels.

Three checks:
  1. Structure: what's actually in the zip — directory names, file counts, naming patterns
  2. Acoustic sanity: do the CVT modes score differently on our existing v8 signals?
     Overdrive/edge (pressed) should have: high CPP, low shimmer — OPPOSITE of rough phonation.
     Neutral/curbing (modal) should have: mid CPP, low shimmer.
  3. Sample playback: print paths to a few samples per class so you can listen manually.

Usage:
    .venv/bin/python3 scripts/inspect_cvt_dataset.py --data data/cvt_dataset/
    .venv/bin/python3 scripts/inspect_cvt_dataset.py --data data/cvt_dataset/ --samples 3
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SR          = 44100
CHUNK       = 4410    # 100ms — same as backend
SILENCE_RMS = 0.008

CVT_KEYWORDS = {
    "overdrive": "overdrive",
    "over_drive": "overdrive",
    "edge": "edge",
    "neutral": "neutral",
    "curbing": "curbing",
    "curb": "curbing",
}

ZONE_COLOR = {"overdrive": "\033[91m", "edge": "\033[93m",
              "neutral": "\033[92m", "curbing": "\033[94m", "reset": "\033[0m"}


def find_label(path: Path):
    for part in reversed([p.lower() for p in path.parts]):
        for kw, lbl in CVT_KEYWORDS.items():
            if kw in part:
                return lbl
    name = path.stem.lower()
    for kw, lbl in CVT_KEYWORDS.items():
        if kw in name:
            return lbl
    return None


def compute_cpp(chunk, sr=SR):
    try:
        N = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N // 2]
        q_min = int(sr / 600)
        q_max = int(sr / 75)
        if q_max >= len(cepstrum):
            return float('nan')
        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max + 1]))
        q_axis = np.arange(len(cepstrum)) / float(sr)
        coeffs = np.polyfit(q_axis[q_min:q_max + 1], cepstrum[q_min:q_max + 1], 1)
        return float(cepstrum[peak_idx] - np.polyval(coeffs, q_axis[peak_idx]))
    except Exception:
        return float('nan')


def analyze_sample(path: Path):
    """Extract v8 acoustic features from a CVT sample. Returns dict of means."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        return None

    # Use middle third (cleanest phonation)
    seg = y[len(y)//3 : 2*len(y)//3]
    n_chunks = len(seg) // CHUNK

    shimmer_vals, cpp_vals, hnr_vals = [], [], []

    for i in range(n_chunks):
        chunk = seg[i*CHUNK:(i+1)*CHUNK]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < SILENCE_RMS:
            continue
        try:
            snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SR))
            harm = snd.to_harmonicity()
            vals = harm.values[0]
            valid = vals[vals > -200]
            hnr = float(np.mean(valid)) if len(valid) > 0 else float('nan')
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_pct = (shim or 0.0) * 100.0
        except Exception:
            hnr = float('nan')
            shimmer_pct = float('nan')

        cpp = compute_cpp(chunk)

        if not math.isnan(shimmer_pct):
            shimmer_vals.append(shimmer_pct)
        if not math.isnan(cpp):
            cpp_vals.append(cpp)
        if not math.isnan(hnr):
            hnr_vals.append(hnr)

    if not shimmer_vals and not cpp_vals:
        return None

    return {
        "shimmer": float(np.mean(shimmer_vals)) if shimmer_vals else float('nan'),
        "cpp":     float(np.mean(cpp_vals))     if cpp_vals     else float('nan'),
        "hnr":     float(np.mean(hnr_vals))     if hnr_vals     else float('nan'),
        "n_chunks": len(shimmer_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data/cvt_dataset/")
    ap.add_argument("--samples", type=int, default=5,
                    help="Audio samples per class to analyze acoustically (slow)")
    ap.add_argument("--structure-only", action="store_true",
                    help="Only show file structure, skip acoustic analysis")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # ── 1. STRUCTURE ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  DATASET STRUCTURE: {data_dir}")
    print(f"{'═'*65}")

    # Show top-level directories
    top_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nTop-level directories ({len(top_dirs)}):")
    for d in top_dirs[:20]:
        n_audio = sum(1 for _ in d.rglob("*.wav")) + sum(1 for _ in d.rglob("*.mp3")) + \
                  sum(1 for _ in d.rglob("*.flac"))
        print(f"  {d.name:<40} ({n_audio} audio files)")
    if len(top_dirs) > 20:
        print(f"  ... and {len(top_dirs)-20} more")

    # Find all audio files and attempt label detection
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(data_dir.rglob(f"*{ext}"))

    print(f"\nTotal audio files found: {len(audio_files)}")

    # Label distribution
    by_label = defaultdict(list)
    unlabeled = []
    for f in audio_files:
        lbl = find_label(f)
        if lbl:
            by_label[lbl].append(f)
        else:
            unlabeled.append(f)

    print(f"\nLabel distribution (by directory/filename matching):")
    for lbl, files in sorted(by_label.items()):
        c = ZONE_COLOR.get(lbl, "")
        print(f"  {c}{lbl:<12}{ZONE_COLOR['reset']}: {len(files)} files")
    if unlabeled:
        print(f"  {'unlabeled':<12}: {len(unlabeled)} files")
        print(f"\n  Example unlabeled paths:")
        for f in unlabeled[:5]:
            print(f"    {f.relative_to(data_dir)}")

    # Show example paths per label so you can verify naming
    print(f"\nExample paths per label (verify these look right):")
    for lbl, files in sorted(by_label.items()):
        c = ZONE_COLOR.get(lbl, "")
        print(f"\n  {c}{lbl.upper()}{ZONE_COLOR['reset']}:")
        for f in files[:3]:
            print(f"    {f.relative_to(data_dir)}")

    # Check for annotation/metadata files
    print(f"\nAnnotation/metadata files found:")
    meta_files = list(data_dir.rglob("*.json")) + list(data_dir.rglob("*.csv")) + \
                 list(data_dir.rglob("*.txt")) + list(data_dir.rglob("*.xlsx"))
    for f in meta_files[:10]:
        print(f"  {f.relative_to(data_dir)}  ({f.stat().st_size//1024}KB)")

    if args.structure_only:
        return

    if not by_label:
        print("\nERROR: No labels detected. Check CVT_KEYWORDS or dataset structure.")
        print("Showing first 10 file paths for manual inspection:")
        for f in audio_files[:10]:
            print(f"  {f.relative_to(data_dir)}")
        return

    # ── 2. ACOUSTIC SANITY CHECK ──────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  ACOUSTIC SANITY CHECK (v8 features per CVT mode)")
    print(f"  Expected for PRESSED (overdrive/edge): high CPP, LOW shimmer, high HNR")
    print(f"  Expected for ROUGH phonation: low CPP, HIGH shimmer, low HNR")
    print(f"  Expected for MODAL (neutral/curbing): mid CPP, low shimmer")
    print(f"{'═'*65}")

    results_by_label = {}
    for lbl, files in sorted(by_label.items()):
        # Sample evenly across the class
        step = max(1, len(files) // args.samples)
        sample_files = files[::step][:args.samples]

        print(f"\n  Analyzing {lbl} ({len(sample_files)} samples)...")
        all_stats = []
        for f in sample_files:
            stats = analyze_sample(f)
            if stats:
                all_stats.append(stats)
                c = ZONE_COLOR.get(lbl, "")
                print(f"    {c}{f.name[:45]:<45}{ZONE_COLOR['reset']}  "
                      f"shim={stats['shimmer']:>5.2f}%  cpp={stats['cpp']:>6.3f}  "
                      f"hnr={stats['hnr']:>5.1f}dB")

        if all_stats:
            results_by_label[lbl] = {
                "shimmer": np.mean([s["shimmer"] for s in all_stats if not math.isnan(s["shimmer"])]),
                "cpp":     np.mean([s["cpp"]     for s in all_stats if not math.isnan(s["cpp"])]),
                "hnr":     np.mean([s["hnr"]     for s in all_stats if not math.isnan(s["hnr"])]),
            }

    # Summary table
    print(f"\n{'═'*65}")
    print(f"  MEAN ACOUSTIC FEATURES BY CVT MODE")
    print(f"{'─'*65}")
    print(f"  {'Mode':<12}  {'Shimmer%':>9}  {'CPP':>8}  {'HNR dB':>8}  Interpretation")
    print(f"{'─'*65}")
    for lbl, stats in sorted(results_by_label.items()):
        c = ZONE_COLOR.get(lbl, "")
        # Interpret based on expected patterns
        shim = stats["shimmer"]
        cpp  = stats["cpp"]
        hnr  = stats["hnr"]
        if lbl in ("overdrive", "edge"):
            interp = "pressed (high CPP, low shim = correct)" if cpp > 0.2 and shim < 8 else "CHECK THIS"
        else:
            interp = "modal (mid CPP, low shim = correct)"   if cpp > 0.1 and shim < 8 else "CHECK THIS"
        print(f"  {c}{lbl:<12}{ZONE_COLOR['reset']}  {shim:>9.2f}  {cpp:>8.3f}  {hnr:>8.1f}  {interp}")

    print(f"{'─'*65}")
    print(f"\n  Key check: overdrive/edge CPP should be HIGHER than neutral/curbing.")
    print(f"  If not — CVT modes don't map the way we think. Re-examine before training.\n")

    # ── 3. LISTEN PATHS ───────────────────────────────────────────────────────
    print(f"{'═'*65}")
    print(f"  SAMPLE PATHS TO LISTEN TO (verify labels manually)")
    print(f"{'═'*65}")
    for lbl, files in sorted(by_label.items()):
        c = ZONE_COLOR.get(lbl, "")
        print(f"\n  {c}{lbl.upper()}{ZONE_COLOR['reset']} — listen and confirm this sounds like {lbl}:")
        for f in files[:2]:
            print(f"    open '{f}'")


if __name__ == "__main__":
    main()
