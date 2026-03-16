#!/usr/bin/env python3
"""
Diagnostic: WHY is the model over-detecting on Daniel's green chunks?

Hypothesis A: Unvoiced frames (consonants, breaths, glottal onsets) are passing
              the RMS gate but producing garbage shimmer/CPP values.
              Fix: voiced-fraction gate — only score frames where pitch is stable.

Hypothesis B: Specific frequency bands in the room/mic response are adding noise
              to the CPP or shimmer computation.
              Fix: bandpass filter before shimmer/CPP.

This script tests both and tells you which hypothesis is supported.

Usage:
    .venv/bin/python3 scripts/diagnose_false_positives.py
"""

import json, math, sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as pcall

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SR = 44100
CHUNK = 4410    # 100ms
SILENCE_RMS = 0.008
CHUNK_S = 2.0

GT_PATH = Path("data/ground_truth/lizajane_labels.json")
SEED_SHIM, SEED_CPP = 5.26, 0.22
BOOTSTRAP_N = 20


# ─── Acoustic feature extraction ──────────────────────────────────────────────

def voiced_fraction(chunk: np.ndarray, sr=SR) -> float:
    """Fraction of the chunk that has a detectable F0 (0-1).
    Unvoiced frames (consonants, silence, breaths) → near 0.
    Steady vowel → near 1."""
    try:
        snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0 = pitch.selected_array['frequency']
        return float(np.mean(f0 > 0))
    except Exception:
        return 0.0


def compute_cpp(chunk, sr=SR):
    try:
        N = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N // 2]
        q_min, q_max = int(sr / 600), int(sr / 75)
        if q_max >= len(cepstrum): return float('nan')
        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max + 1]))
        q_axis = np.arange(len(cepstrum)) / float(sr)
        coeffs = np.polyfit(q_axis[q_min:q_max + 1], cepstrum[q_min:q_max + 1], 1)
        return float(cepstrum[peak_idx] - np.polyval(coeffs, q_axis[peak_idx]))
    except Exception:
        return float('nan')


def analyze_frame(chunk, shim_base, cpp_base, onset_gated, bandpass=False):
    """Returns (v8_strain, voiced_frac, shimmer, cpp) for this 100ms chunk."""
    x = chunk.astype(np.float64)

    if bandpass:
        # 100-1200 Hz bandpass — isolates fundamental + lower harmonics, removes
        # broadband consonant noise and low-freq mic rumble
        from scipy.signal import butter, sosfilt
        sos = butter(4, [100, 1200], btype='bandpass', fs=SR, output='sos')
        x = sosfilt(sos, x)

    vf = voiced_fraction(x)
    cpp = compute_cpp(x)

    shimmer_pct = float('nan')
    try:
        snd = parselmouth.Sound(x, sampling_frequency=float(SR))
        pp = pcall(snd, "To PointProcess (periodic, cc)", 75, 600)
        s = pcall([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_pct = (s or 0.0) * 100.0
    except Exception:
        pass

    sg = shimmer_pct if not math.isnan(shimmer_pct) else shim_base
    cg = cpp         if not math.isnan(cpp)         else cpp_base

    if onset_gated:
        shim_dev, cpp_dev = 0.0, 0.0
    else:
        shim_dev = max(0.0, sg - shim_base) / 10.0
        cpp_dev  = max(0.0, cpp_base - cg)  / 0.5

    v8 = min(1.0, max(shim_dev, cpp_dev))
    return v8, vf, shimmer_pct, cpp


# ─── Main analysis ─────────────────────────────────────────────────────────────

def run(bandpass=False, voiced_gate=0.0, label="baseline"):
    """Run per-frame analysis and return per-chunk P80s vs ground truth labels."""
    gt = json.loads(GT_PATH.read_text())
    chunks_gt = gt["chunks"]

    y, _ = librosa.load(gt["path"], sr=SR, mono=True)
    n_frames = len(y) // CHUNK

    shim_base, cpp_base, base_n, voiced_run = SEED_SHIM, SEED_CPP, 0, 0
    frame_data = {}  # chunk_idx → list of (v8, voiced_frac, shim, cpp)

    for fi in range(n_frames):
        ci = int(fi * 0.1 // CHUNK_S)
        chunk = y[fi * CHUNK:(fi + 1) * CHUNK]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        if rms < SILENCE_RMS:
            voiced_run = 0
            continue

        voiced_run += 1
        onset_gated = voiced_run < 3

        v8, vf, shim, cpp = analyze_frame(chunk, shim_base, cpp_base, onset_gated, bandpass=bandpass)

        # Voiced gate: skip frames where voice is not detected reliably
        voiced_gate_pass = vf >= voiced_gate

        # Baseline update (same as backend)
        sg = shim if not math.isnan(shim) else shim_base
        cg = cpp  if not math.isnan(cpp)  else cpp_base
        in_boot = base_n < BOOTSTRAP_N
        ts = max(0, sg - shim_base) / 10
        tc = max(0, cpp_base - cg) / 0.5
        clean = not onset_gated and (in_boot or max(ts, tc) < 0.25)
        if clean:
            a = 0.15 if in_boot else 0.05
            shim_base = (1 - a) * shim_base + a * sg
            cpp_base  = (1 - a) * cpp_base  + a * cg
            base_n += 1

        if ci not in frame_data:
            frame_data[ci] = []
        frame_data[ci].append((v8 if voiced_gate_pass else 0.0, vf, shim, cpp))

    # Aggregate per chunk
    results = {}  # chunk_idx → {label, p80, voiced_frac_mean, ...}
    for ci_str, gt_label in chunks_gt.items():
        ci = int(ci_str)
        if gt_label == "skip":
            continue
        frames = frame_data.get(ci, [])
        if not frames:
            continue
        v8s = [f[0] for f in frames]
        vfs = [f[1] for f in frames]
        results[ci] = {
            "label": gt_label,
            "p80":   float(np.percentile(v8s, 80)),
            "mean":  float(np.mean(v8s)),
            "vf":    float(np.mean(vfs)),   # mean voiced fraction
            "n":     len(frames),
        }
    return results


def accuracy(results, green_thresh, yellow_thresh):
    """Count exact matches, over/under given thresholds."""
    exact = over = under = total = 0
    ZONE_ORDER = {"green": 0, "yellow": 1, "red": 2}

    def zone(p80):
        return "green" if p80 < green_thresh else "yellow" if p80 < yellow_thresh else "red"

    for ci, r in results.items():
        gt_ord    = ZONE_ORDER[r["label"]]
        model_ord = ZONE_ORDER[zone(r["p80"])]
        diff = model_ord - gt_ord
        if diff == 0: exact += 1
        elif diff > 0: over += 1
        else: under += 1
        total += 1
    return exact, over, under, total


def main():
    print("\n" + "═" * 65)
    print("  FALSE POSITIVE DIAGNOSIS — Liza Jane")
    print("═" * 65)

    # ── Hypothesis A: are false positive frames unvoiced? ─────────────────────
    print("\n── HYPOTHESIS A: Unvoiced frames causing false positives ──\n")

    base = run(bandpass=False, voiced_gate=0.0)

    # Split by label and look at voiced fraction of false-positive chunks
    fp_chunks = {ci: r for ci, r in base.items()
                 if r["label"] == "green" and r["p80"] >= 0.35}
    tp_chunks = {ci: r for ci, r in base.items()
                 if r["label"] == "green" and r["p80"] < 0.35}

    if fp_chunks:
        fp_vf = np.mean([r["vf"] for r in fp_chunks.values()])
        tp_vf = np.mean([r["vf"] for r in tp_chunks.values()]) if tp_chunks else 0
        print(f"  False positive GREEN chunks  (model ≥ 0.35, you said green): {len(fp_chunks)}")
        print(f"    Mean voiced fraction: {fp_vf:.2f}  (1.0 = fully voiced, 0 = silence/consonant)")
        print(f"  True positive GREEN chunks   (model < 0.35, you said green): {len(tp_chunks)}")
        print(f"    Mean voiced fraction: {tp_vf:.2f}")
        print()

        if fp_vf < tp_vf - 0.1:
            print("  → HYPOTHESIS A SUPPORTED: false positive chunks have lower voiced fraction.")
            print("    A voiced gate would reduce false positives.")
        else:
            print("  → Hypothesis A WEAK: voiced fractions are similar. Unvoiced frames are not the main issue.")

    # Per-chunk voiced fraction for context
    print(f"\n  {'Chunk':>5}  {'Time':>8}  {'Label':<8}  {'P80':>6}  {'VoicedFrac':>11}  {'FP?':>4}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*11}  {'─'*4}")
    for ci, r in sorted(base.items()):
        fp = "FP" if r["label"] == "green" and r["p80"] >= 0.35 else ""
        z_c = "\033[91m" if fp else "\033[92m" if r["label"]=="green" else "\033[93m" if r["label"]=="yellow" else "\033[91m"
        rst = "\033[0m"
        print(f"  {ci:>5}  {ci*2:>3}-{(ci+1)*2:<3}s  {z_c}{r['label']:<8}{rst}  "
              f"{r['p80']:>6.3f}  {r['vf']:>11.2f}  {fp:>4}")

    # ── Hypothesis B: does bandpass filtering help? ────────────────────────────
    print(f"\n{'─'*65}")
    print("── HYPOTHESIS B: Frequency content — test 100-1200Hz bandpass ──\n")
    print("  (This takes ~60s — analyzing each frame with filter...)")

    bp = run(bandpass=True, voiced_gate=0.0)

    base_ex, base_ov, base_un, base_tot = accuracy(base, 0.35, 0.55)
    bp_ex,   bp_ov,   bp_un,   bp_tot   = accuracy(bp,   0.35, 0.55)

    print(f"\n  {'Condition':<30}  {'Exact':>6}  {'Over':>6}  {'Under':>6}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}")
    print(f"  {'No filter (current)':<30}  {base_ex:>5}/{base_tot}  {base_ov:>5}/{base_tot}  {base_un:>5}/{base_tot}")
    print(f"  {'100-1200Hz bandpass':<30}  {bp_ex:>5}/{bp_tot}  {bp_ov:>5}/{bp_tot}  {bp_un:>5}/{bp_tot}")

    diff = bp_ex - base_ex
    if diff > 2:
        print(f"\n  → HYPOTHESIS B SUPPORTED: bandpass filter improves exact matches by {diff}.")
        print("    Recommend: apply 100-1200Hz bandpass before shimmer/CPP computation.")
    elif diff > 0:
        print(f"\n  → Hypothesis B MARGINAL: +{diff} exact matches. Minor improvement.")
    else:
        print(f"\n  → Hypothesis B REJECTED: bandpass filter does not improve accuracy (Δ={diff}).")

    # ── Test voiced gate thresholds ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("── VOICED GATE TEST — what voiced-fraction threshold helps most? ──\n")
    print(f"  {'Gate threshold':>16}  {'Exact':>6}  {'Over':>6}  {'Under':>6}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*6}")

    best_exact, best_gate = base_ex, 0.0
    for gate in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = run(bandpass=False, voiced_gate=gate)
        ex, ov, un, tot = accuracy(r, 0.35, 0.55)
        marker = " ← best" if ex > best_exact else ""
        print(f"  voiced ≥ {gate:.1f}         {ex:>5}/{tot}  {ov:>5}/{tot}  {un:>5}/{tot}{marker}")
        if ex > best_exact:
            best_exact = ex
            best_gate = gate

    if best_gate > 0:
        print(f"\n  → Best voiced gate: {best_gate:.1f}  (+{best_exact-base_ex} exact matches vs no gate)")
        print(f"    Meaning: only score strain on frames where ≥{best_gate*100:.0f}% of the window is voiced.")
    else:
        print(f"\n  → Voiced gate does not improve accuracy at any threshold tested.")

    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()
