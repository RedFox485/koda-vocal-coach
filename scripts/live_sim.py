#!/usr/bin/env python3
"""
Simulates the backend analysis pipeline on audio files.
Feeds audio in 100ms chunks (exactly like the backend) through the same
HNR/shimmer/formula logic so we can see frame-by-frame scoring.

Usage:
    python scripts/live_sim.py                   # all anchor clips
    python scripts/live_sim.py --file "Easy 2.m4a"
    python scripts/live_sim.py --songs           # full song recordings
    python scripts/live_sim.py --all             # anchors + songs

Output: frame timeline + summary per clip
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

# EARS imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mel_extractor import MelExtractor
from frequency_explorer import analyze_mel

_mel = MelExtractor(sample_rate=44100)


ROOT = Path(__file__).parent.parent
ANCHOR_DIR = ROOT / "Vocal test recording sessions" / "Anchors"
SONG_DIR   = ROOT / "Vocal test recording sessions"

ANCHORS = [
    ("Easy 2.m4a",    "green"),
    ("Medium 1.m4a",  "green"),
    ("hard push.m4a", "yellow"),
    ("Rough 1.m4a",   "red"),
]

# Seeds calibrated per-signal:
# shim seed: Easy 2 anchor (universal — shimmer doesn't shift much with context)
# cpp  seed: Easy 2 CPP frames (voiced, mid-phrase), typical value
#            Easy 2 mean=0.198. Seed at 0.22 = conservative (slightly above Easy 2 average)
#            so adaptation can go down slightly if singer's voice is naturally lower.
DANIEL_SHIM_SEED = 5.26   # % shimmer — Easy 2 anchor
DANIEL_CPP_SEED  = 0.22   # CPP (neper units) — typical easy singing

SONGS = [
    ("Danny - Chris Young R1.m4a",           "mixed"),
    ("Danny - Liza Jane R1 (longer).m4a",    "mixed"),
    ("Danny - Runnin down a dream R1.m4a",   "mixed"),
    ("calibration_AB.wav",                    "mixed"),
]

SR        = 44100
CHUNK     = 4410    # 100ms — same as backend
PRAAT_WIN = 8820    # 200ms parselmouth window — matches backend PRAAT_WINDOW_SAMPLES
SILENCE_RMS = 0.003 # matches backend

STRAIN_GREEN  = 0.25  # matches backend
STRAIN_YELLOW = 0.40  # matches backend

BASELINE_EMA_ALPHA = 0.05   # per clean frame: ~10 frames → 40% dialed in, ~50 → 92%
BASELINE_MAX_SCORE = 0.35   # gate: only frames this relaxed contribute to adaptation (matches backend)
BASELINE_WARM_N    = 20     # clean frames before baseline is "warmed up" (display only)
ONSET_GATE_FRAMES  = 3      # match backend onset gate (300ms)
EMA_ALPHA          = 0.40   # output EMA smoothing — matches backend

# CPP 3-frame EMA — matches backend CPP_EMA_ALPHA
CPP_EMA_ALPHA = 0.33

ZONE_COLOR = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m", "reset": "\033[0m"}
ZONE_BAR   = {"green": "█", "yellow": "▓", "red": "░"}


def praat_hnr_shimmer(audio, sr=SR, win=PRAAT_WIN):
    """Extract HNR and shimmer from the last PRAAT_WIN samples."""
    chunk = audio[-win:] if len(audio) >= win else audio
    snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(sr))
    harm = snd.to_harmonicity()
    vals = harm.values[0]
    valid = vals[vals > -200]
    hnr_db = float(np.mean(valid)) if len(valid) > 0 else 20.0
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_pct = (shim or 0.0) * 100.0
    except Exception:
        shimmer_pct = float('nan')
    return hnr_db, shimmer_pct


def compute_cpp(chunk, sr=SR, win=PRAAT_WIN):
    """Cepstral Peak Prominence (CPP) — loudness-robust strain indicator.
    Loud+healthy → HIGH CPP (dB).  Loud+strained → LOWER CPP.
    Detects both rough phonation and pressed phonation since both reduce
    glottal periodicity regardless of vocal intensity.

    Computed via power cepstrum: log|FFT|^2 → IFFT → find peak in pitch quefrency range.
    CPP = peak height - regression line value at that quefrency (dB).
    """
    try:
        N = len(chunk)
        # Pre-emphasize (de-emphasize low-frequency dominance)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        # Windowed FFT → log power spectrum → cepstrum
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N//2]

        # Quefrency axis (seconds per sample = 1/sr)
        q_axis = np.arange(len(cepstrum)) / float(sr)

        # Quefrency range for fundamental 75–600 Hz = 1.67ms–13.3ms
        q_min = int(sr / 600)   # ~74 samples
        q_max = int(sr / 75)    # ~588 samples

        if q_max >= len(cepstrum):
            return float('nan')

        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max+1]))
        peak_val = cepstrum[peak_idx]

        # Fit regression line to cepstrum in the quefrency range [q_min, q_max]
        qs = q_axis[q_min:q_max+1]
        cs = cepstrum[q_min:q_max+1]
        coeffs = np.polyfit(qs, cs, 1)
        regression_at_peak = np.polyval(coeffs, q_axis[peak_idx])

        cpp_db = float(peak_val - regression_at_peak)
        return cpp_db
    except Exception:
        return float('nan')


def strain_v8(shimmer_pct, cpp, base_shim, base_cpp):
    """Formula v8: max(shim_dev, cpp_dev) — both onset-gated (voiced_run >= 3)
    shim_dev = rough phonation (shimmer spike ABOVE baseline)
    cpp_dev  = phonatory irregularity / tightness (CPP DROP BELOW baseline)
               CPP drops for BOTH rough and pressed phonation — detects tightness/strain
               regardless of vocal intensity (loudness-robust)
    Both suppressed on voiced onset to eliminate phrase boundary artifacts.
    Normalization: shimmer /10 (0-10% range), CPP relative drop /0.5
    """
    shim_dev = max(0.0, shimmer_pct - base_shim) / 7.0 if not math.isnan(shimmer_pct) else 0.0
    cpp_dev  = max(0.0, base_cpp - cpp) / 0.35 if not math.isnan(cpp) else 0.0  # drop of 0.35 = fully strained
    score    = min(1.0, max(shim_dev, cpp_dev))
    return score, shim_dev, cpp_dev


def zone_of(score):
    return "green" if score < STRAIN_GREEN else "yellow" if score < STRAIN_YELLOW else "red"


def analyze_clip(path: Path, true_label: str, show_frames=True,
                 seed_shim=DANIEL_SHIM_SEED, seed_cpp=DANIEL_CPP_SEED):
    y, _ = librosa.load(str(path), sr=SR, mono=True)

    # Baseline: seeded from known easy-singing values.
    base_shim    = seed_shim
    base_cpp     = seed_cpp
    clean_frames = 0
    cpp_ema      = seed_cpp   # CPP 3-frame EMA (matches backend CPP_EMA_ALPHA)
    ema_strain   = 0.0        # output EMA smoothing (matches backend EMA_ALPHA)

    frames = []
    n_chunks = len(y) // CHUNK
    voiced_run = 0

    # Build ring buffer for 200ms Praat windows (sliding window over audio)
    ring_size = max(PRAAT_WIN, CHUNK)

    print(f"\n{'─'*80}")
    c = ZONE_COLOR
    label_str = f"{c[true_label]}{true_label.upper()}{c['reset']}" if true_label in c else true_label
    print(f"  {path.name}   (true label: {label_str})")
    print(f"  {n_chunks} frames × 100ms = {n_chunks*0.1:.1f}s")
    print(f"{'─'*80}")

    if show_frames:
        print(f"  {'t':>5}  {'shim':>6}  {'cpp':>7}  {'cppE':>7}  {'shdev':>6}  {'cdev':>6}  {'raw':>6}  {'ema':>6}  {'zone':>7}  {'base_cpp':>8}  bar")

    for i in range(n_chunks):
        chunk = y[i*CHUNK:(i+1)*CHUNK]
        # Build audio window for Praat (200ms, like backend uses _ring[-PRAAT_WIN:])
        start_sample = max(0, (i+1)*CHUNK - PRAAT_WIN)
        audio_window = y[start_sample:(i+1)*CHUNK]
        rms = float(np.sqrt(np.mean(chunk**2)))
        t = i * 0.1
        low_energy = rms < 0.007  # matches backend LOW_ENERGY_RMS

        if rms < SILENCE_RMS:
            voiced_run = 0
            # Reset CPP EMA on silence (matches backend behavior)
            cpp_ema = base_cpp
            ema_strain = max(0.0, ema_strain - 0.02)  # decay like backend
            frames.append({"t": t, "zone": "silent", "score": 0.0, "ema_score": ema_strain, "rms": rms})
            if show_frames:
                print(f"  {t:>5.1f}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>6}  {'—':>6}  {'—':>6}  {ema_strain:>6.3f}  {'silent':>7}  {base_cpp:>8.3f}")
            continue

        voiced_run += 1
        onset_gated = voiced_run < ONSET_GATE_FRAMES

        # Gate onset — shimmer and CPP unreliable until folds settle
        if onset_gated or low_energy:
            shimmer_pct = float('nan')
            cpp = float('nan')
        else:
            _, shimmer_pct = praat_hnr_shimmer(audio_window)
            cpp = compute_cpp(audio_window[-PRAAT_WIN:] if len(audio_window) >= PRAAT_WIN else audio_window)

        # CPP EMA smoothing (3-frame, matches backend)
        if not math.isnan(cpp):
            cpp_ema = CPP_EMA_ALPHA * cpp + (1 - CPP_EMA_ALPHA) * cpp_ema
        cpp_for_score = cpp_ema

        # EMA baseline — only clean frames contribute
        shim_for_gate = shimmer_pct if not math.isnan(shimmer_pct) else base_shim
        t_shdev = max(0.0, shim_for_gate - base_shim) / 10.0
        t_cdev  = max(0.0, base_cpp - cpp_for_score) / 0.5
        t_score = min(1.0, max(t_shdev, t_cdev))
        is_clean = t_score < BASELINE_MAX_SCORE and not onset_gated
        if is_clean:
            a = BASELINE_EMA_ALPHA
            # Shimmer: asymmetric (UP=0.05, DOWN=0.01) — matches backend
            if not math.isnan(shimmer_pct):
                if shimmer_pct >= base_shim:
                    base_shim = (1 - a) * base_shim + a * shimmer_pct
                else:
                    base_shim = (1 - 0.01) * base_shim + 0.01 * shimmer_pct
            # CPP: symmetric (alpha=0.03) — prevents baseline ratcheting from loud clean frames
            if not math.isnan(cpp):
                base_cpp = (1 - 0.03) * base_cpp + 0.03 * cpp
            clean_frames += 1
            if clean_frames == BASELINE_WARM_N and show_frames:
                print(f"  >>> BASELINE WARM: shimmer={base_shim:.2f}%  cpp={base_cpp:.3f}")

        # Strain score using CPP EMA (not raw CPP)
        score, shdev, cdev = strain_v8(shimmer_pct, cpp_for_score, base_shim, base_cpp)
        # Output EMA smoothing (matches backend)
        ema_strain = EMA_ALPHA * score + (1 - EMA_ALPHA) * ema_strain
        ema_zone = zone_of(ema_strain)

        shim_str = f"{shimmer_pct:>6.2f}" if not math.isnan(shimmer_pct) else f"  {'nan':>4}"
        cpp_str  = f"{cpp:>7.3f}"          if not math.isnan(cpp)         else f"  {'nan':>5}"
        cppE_str = f"{cpp_for_score:>7.3f}"
        frames.append({"t": t, "zone": ema_zone, "score": score, "ema_score": ema_strain,
                       "shim": shimmer_pct, "cpp": cpp, "cpp_ema": cpp_for_score,
                       "shdev": shdev, "cdev": cdev, "rms": rms, "base_cpp": base_cpp})

        if show_frames:
            zc = ZONE_COLOR.get(ema_zone, "")
            bar_ch = ZONE_BAR.get(ema_zone, "?")
            bar_w = min(30, max(1, int(ema_strain * 30)))
            bar = bar_ch * bar_w
            cal = "" if clean_frames >= BASELINE_WARM_N else f" [{clean_frames}cf]"
            print(f"  {t:>5.1f}  {shim_str}  {cpp_str}  {cppE_str}  {shdev:>6.3f}  {cdev:>6.3f}  "
                  f"{score:>6.3f}  {zc}{ema_strain:>6.3f}  {ema_zone:>7}{ZONE_COLOR['reset']}  {base_cpp:>8.3f}  {zc}{bar}{ZONE_COLOR['reset']}{cal}")

    # Summary — use EMA-smoothed scores (matches what the frontend displays)
    voiced = [f for f in frames if f["zone"] != "silent"]
    if not voiced:
        print("  (no voiced frames)")
        return

    ema_scores = [f["ema_score"] for f in voiced]

    zone_counts = {"green": 0, "yellow": 0, "red": 0}
    for s in ema_scores:
        z = zone_of(s)
        if z in zone_counts:
            zone_counts[z] += 1

    p80 = float(np.percentile(ema_scores, 80))
    dominant = zone_of(p80)

    # Also show base_cpp drift
    cpp_vals = [f.get("base_cpp", 0) for f in voiced if f.get("base_cpp")]
    cpp_drift = ""
    if cpp_vals:
        cpp_drift = f"  cpp_baseline drift: {cpp_vals[0]:.3f} → {cpp_vals[-1]:.3f} (Δ{cpp_vals[-1]-cpp_vals[0]:+.3f})"

    print(f"\n  SUMMARY (EMA-smoothed): mean={np.mean(ema_scores):.3f}  p80={p80:.3f}  max={np.max(ema_scores):.3f}  "
          f"dominant={dominant}  (green={zone_counts['green']}  "
          f"yellow={zone_counts['yellow']}  red={zone_counts['red']}  "
          f"voiced={len(voiced)})")

    warm_str = "warmed up" if clean_frames >= BASELINE_WARM_N else f"adapting ({clean_frames} clean frames)"
    print(f"  Baseline: shimmer={base_shim:.2f}%  cpp={base_cpp:.3f}  [{warm_str}]")
    if cpp_drift:
        print(f" {cpp_drift}")

    ok = "✓" if dominant == true_label or true_label == "mixed" else "✗"
    dc = ZONE_COLOR.get(dominant, "")
    tc = ZONE_COLOR.get(true_label, "")
    print(f"  Result: {ok}  predicted={dc}{dominant}{ZONE_COLOR['reset']}  "
          f"true={tc}{true_label}{ZONE_COLOR['reset']}")

    return {"dominant": dominant, "true": true_label, "correct": dominant == true_label or true_label == "mixed",
            "mean_score": float(np.mean(ema_scores)), "max_score": float(np.max(ema_scores)),
            "baseline_warm": clean_frames >= BASELINE_WARM_N,
            "clean_frames": clean_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",    help="Single anchor filename to run")
    ap.add_argument("--songs",   action="store_true")
    ap.add_argument("--all",     action="store_true")
    ap.add_argument("--quiet",   action="store_true", help="Summary only, no per-frame output")
    args = ap.parse_args()

    clips = []
    if args.file:
        path = ANCHOR_DIR / args.file
        if not path.exists():
            path = SONG_DIR / args.file
        clips = [(path, "?")]
    elif args.songs or args.all:
        clips = [(SONG_DIR / f, lbl) for f, lbl in SONGS]
        if args.all:
            clips = [(ANCHOR_DIR / f, lbl) for f, lbl in ANCHORS] + clips
    else:
        clips = [(ANCHOR_DIR / f, lbl) for f, lbl in ANCHORS]

    results = []
    for path, label in clips:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        r = analyze_clip(path, label, show_frames=not args.quiet)
        if r:
            results.append(r)

    if len(results) > 1:
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{'═'*70}")
        print(f"  OVERALL: {correct}/{len(results)} correct ({100*correct/len(results):.0f}%)")
        for r in results:
            ok = "✓" if r["correct"] else "✗"
            print(f"    {ok} true={r['true']:>7}  pred={r['dominant']:>7}  "
                  f"mean={r['mean_score']:.3f}  max={r['max_score']:.3f}  "
                  f"clean_frames={r['clean_frames']}  {'warm' if r['baseline_warm'] else 'adapting'}")


if __name__ == "__main__":
    main()
