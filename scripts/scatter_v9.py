#!/usr/bin/env python3
"""
Wavelet Scattering Strain Score — v9 exploration

Strategy:
  1. Compute 1D wavelet scattering coefficients (34-dim) per 100ms frame
  2. Build baseline from Easy 2 anchor (clean singing)
  3. Per-frame Mahalanobis-approx distance from baseline = scatter strain score
  4. Fuse with v8 (CPP + shimmer) to create hybrid score
  5. Test on all 3 song clips — especially Liza Jane (verses green, chorus yellow/red)

Wavelet scattering captures multi-scale amplitude modulation:
  - Healthy singing: stable AM patterns at all scales (vibrato, natural dynamics)
  - Strained singing: irregular AM bursts, scale-inconsistent patterns
  This is what the 2025 paper (86.1% accuracy) found as the best feature set.

Usage:
    .venv/bin/python3 scripts/scatter_v9.py              # songs only
    .venv/bin/python3 scripts/scatter_v9.py --all        # anchors + songs
    .venv/bin/python3 scripts/scatter_v9.py --calibrate  # inspect baseline stats
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Wavelet scattering — bypass broken 3D module (scipy.special.sph_harm removed in 1.17)
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D

ROOT      = Path(__file__).parent.parent
ANCHOR_DIR = ROOT / "Vocal test recording sessions" / "Anchors"
SONG_DIR   = ROOT / "Vocal test recording sessions"

ANCHORS = [
    ("Easy 2.m4a",    "green"),
    ("Medium 1.m4a",  "green"),
    ("hard push.m4a", "yellow"),
    ("Rough 1.m4a",   "red"),
]
SONGS = [
    ("Danny - Runnin down a dream R1.m4a",  "green"),
    ("Danny - Chris Young R1.m4a",          "yellow"),
    ("Danny - Liza Jane R1 (longer).m4a",   "mixed"),
]

SR         = 44100
CHUNK      = 4410     # 100ms
SCATTER_N  = 4096     # must be power-of-2 ≥ CHUNK — pad/trim chunk to this
SILENCE_RMS = 0.008

# v8 thresholds and seeds
STRAIN_GREEN  = 0.35
STRAIN_YELLOW = 0.55
DANIEL_SHIM_SEED = 5.26
DANIEL_CPP_SEED  = 0.22
BASELINE_EMA_ALPHA   = 0.05
BASELINE_MAX_SCORE   = 0.25
BASELINE_WARM_N      = 20

# Scattering: J=7 → 7 octaves of modulation, Q=1 → simpler, 34 coefficients
# Q=1 gives 34 dims vs Q=8's 176 — more robust with small baselines (~26 voiced frames)
# shape=4096 → ~93ms at 44100Hz (close to our 100ms chunk)
J = 7
Q = 1
_scatter = Scattering1D(J=J, shape=SCATTER_N, Q=Q)

ZONE_COLOR = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m", "reset": "\033[0m"}
ZONE_BAR   = {"green": "█", "yellow": "▓", "red": "░"}


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def praat_shimmer(chunk, sr=SR):
    snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(sr))
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return (shim or 0.0) * 100.0
    except Exception:
        return float('nan')


def compute_cpp(chunk, sr=SR):
    """Cepstral Peak Prominence — loudness-robust. High=healthy, Low=strained."""
    try:
        N = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N // 2]
        q_axis = np.arange(len(cepstrum)) / float(sr)
        q_min = int(sr / 600)
        q_max = int(sr / 75)
        if q_max >= len(cepstrum):
            return float('nan')
        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max + 1]))
        qs = q_axis[q_min:q_max + 1]
        cs = cepstrum[q_min:q_max + 1]
        coeffs = np.polyfit(qs, cs, 1)
        regression_at_peak = np.polyval(coeffs, q_axis[peak_idx])
        return float(cepstrum[peak_idx] - regression_at_peak)
    except Exception:
        return float('nan')


def scatter_features(chunk):
    """Extract 34-dim scattering coefficients from a chunk.
    Pads/trims to SCATTER_N samples. Returns 1D array of log-compressed means across time.

    Log compression is critical: raw scattering spans 3+ orders of magnitude,
    making L2 distance dominated by high-energy dims. Log-space equalizes dynamic range.
    RMS normalization before scatter → loudness invariance.
    """
    x = chunk.astype(np.float64)
    if len(x) < SCATTER_N:
        x = np.pad(x, (0, SCATTER_N - len(x)))
    else:
        x = x[:SCATTER_N]
    # RMS-normalize: removes overall loudness, leaves timbre/modulation structure
    rms = np.sqrt(np.mean(x ** 2))
    if rms > 1e-8:
        x = x / rms
    Sx = _scatter(x)  # shape: (34, 32)
    # Mean over time axis → 34-dim feature vector, then log-compress
    feat = np.mean(Sx, axis=1)
    return np.log(np.abs(feat) + 1e-10)  # log-compress to equalize dynamic range


# ---------------------------------------------------------------------------
# Baseline calibration from Easy 2 anchor
# ---------------------------------------------------------------------------

def _load_voiced_features(path: Path) -> list:
    """Load voiced scatter features from a clip (skipping onset frames)."""
    if not path.exists():
        return []
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    n_chunks = len(y) // CHUNK
    voiced_run = 0
    features = []
    for i in range(n_chunks):
        chunk = y[i * CHUNK:(i + 1) * CHUNK]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < SILENCE_RMS:
            voiced_run = 0
            continue
        voiced_run += 1
        if voiced_run < 3:
            continue
        features.append(scatter_features(chunk))
    return features


def build_scatter_baseline(verbose=False):
    """Baseline from Easy 2 + Medium 1 (both green anchors) for robustness.

    Log-compressed features are used so dims are comparable (equalized dynamic range).
    Per-dim z-score normalization for scatter_strain() distance.
    """
    all_features = []
    for name, label in ANCHORS:
        if label == "green":
            feats = _load_voiced_features(ANCHOR_DIR / name)
            if verbose:
                print(f"  {name}: {len(feats)} voiced frames")
            all_features.extend(feats)

    if not all_features:
        print("ERROR: no voiced frames in baseline clips")
        return None, None

    feats = np.stack(all_features)  # (N, 34) log-compressed
    baseline_mean = np.mean(feats, axis=0)
    baseline_std  = np.std(feats, axis=0) + 1e-8   # per-dim, for z-score

    if verbose:
        print(f"Baseline: {len(all_features)} total voiced frames, {feats.shape[1]} dims")
        print(f"  Mean range (log): [{baseline_mean.min():.3f}, {baseline_mean.max():.3f}]")
        print(f"  Std range (log):  [{baseline_std.min():.4f}, {baseline_std.max():.4f}]")
        # Self-score sanity check
        scores = []
        for feat in all_features:
            z = (feat - baseline_mean) / baseline_std
            raw = float(np.mean(np.abs(z)))
            scores.append(min(1.0, max(0.0, (raw - 1.0) / 2.0)))
        print(f"  Self-score on baseline (should be ~0): mean={np.mean(scores):.3f}  "
              f"p80={float(np.percentile(scores, 80)):.3f}  max={np.max(scores):.3f}")

    return baseline_mean, baseline_std


def scatter_strain(feat, baseline_mean, baseline_std):
    """Mean absolute z-score across scattering dims as strain score.
    Returns score in [0, 1] where 0=at baseline, 1=highly deviant.

    z = (feat - mean) / std, then mean|z| across 34 dims.
    At baseline: mean|z| ≈ 1.0 (by definition of std).
    For strained singing (different modulation pattern): mean|z| > 1.
    Scale: (mean|z| - 1) / 2 → 0 at baseline, 0.5 at 2× deviation, 1.0 at 3×.

    Uses log-compressed features so dims are in comparable scale.
    baseline_std is per-dim (34-dim array).
    """
    z = (feat - baseline_mean) / baseline_std
    raw = float(np.mean(np.abs(z)))
    return min(1.0, max(0.0, (raw - 1.0) / 2.0))


def strain_v8(shimmer_pct, cpp, base_shim, base_cpp):
    shim_dev = max(0.0, shimmer_pct - base_shim) / 10.0 if not math.isnan(shimmer_pct) else 0.0
    cpp_dev  = max(0.0, base_cpp - cpp) / 0.5 if not math.isnan(cpp) else 0.0
    return min(1.0, max(shim_dev, cpp_dev)), shim_dev, cpp_dev


def zone_of(score):
    return "green" if score < STRAIN_GREEN else "yellow" if score < STRAIN_YELLOW else "red"


# ---------------------------------------------------------------------------
# Per-clip analysis
# ---------------------------------------------------------------------------

def analyze_clip_v9(path: Path, true_label: str,
                    anchor_mean=None, anchor_std=None,
                    show_frames=True,
                    fusion_weight=0.5):
    """
    Hybrid v8 + session-adaptive scatter baseline.

    The scatter baseline is built WITHIN each clip from v8-gated clean frames,
    exactly like the shimmer/CPP EMA baseline. This avoids cross-clip acoustic
    mismatch (different room, mic, phrase length) that made anchor-based scatter
    fail.

    fusion_weight: 0.0 = pure scatter, 1.0 = pure v8, 0.5 = equal blend
    scatter is disabled until enough in-song baseline frames are collected (SCATTER_WARM_N).
    """
    SCATTER_WARM_N = 15   # frames needed before scatter is meaningful
    SCATTER_ALPHA  = 0.05  # EMA alpha for scatter baseline adaptation (same as v8)

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    n_chunks = len(y) // CHUNK

    # v8 EMA baseline
    base_shim = DANIEL_SHIM_SEED
    base_cpp  = DANIEL_CPP_SEED
    # Scatter baseline (session-adaptive from clean frames)
    sc_baseline_feats = []   # collect clean frames first
    sc_mean = None           # initialized after SCATTER_WARM_N clean frames
    sc_std  = None

    clean_frames = 0
    voiced_run   = 0
    frames = []

    print(f"\n{'─'*75}")
    c = ZONE_COLOR
    lc = c.get(true_label, "")
    print(f"  {path.name}   (true: {lc}{true_label.upper()}{c['reset']})")
    print(f"  {n_chunks} frames × 100ms = {n_chunks * 0.1:.1f}s | fusion_weight={fusion_weight}")
    print(f"{'─'*75}")

    if show_frames:
        print(f"  {'t':>5}  {'shim':>6}  {'cpp':>7}  {'scat':>6}  {'v8':>6}  {'fuse':>6}  zone   bar")

    for i in range(n_chunks):
        chunk = y[i * CHUNK:(i + 1) * CHUNK]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        t = i * 0.1

        if rms < SILENCE_RMS:
            voiced_run = 0
            frames.append({"t": t, "zone": "silent", "score": 0.0, "v8": 0.0, "scatter": 0.0})
            if show_frames:
                print(f"  {t:>5.1f}  {'—':>6}  {'—':>7}  {'—':>6}  {'—':>6}  {'—':>6}  silent")
            continue

        voiced_run += 1
        onset_gated = voiced_run < 3

        # Always compute scatter features (onset artifacts are minor after log-compression)
        feat = scatter_features(chunk)

        # v8 signals — gate onset for shimmer/CPP
        if onset_gated:
            shimmer_pct = float('nan')
            cpp = float('nan')
        else:
            shimmer_pct = praat_shimmer(chunk)
            cpp = compute_cpp(chunk)

        # v8 tentative score (for gating baseline updates)
        t_score, _, _ = strain_v8(shimmer_pct, cpp, base_shim, base_cpp)
        is_clean = (t_score < BASELINE_MAX_SCORE and
                    not math.isnan(shimmer_pct) and
                    not math.isnan(cpp))

        if is_clean:
            # Update v8 EMA baseline
            a = BASELINE_EMA_ALPHA
            base_shim = (1 - a) * base_shim + a * shimmer_pct
            base_cpp = (1 - a) * base_cpp + a * cpp if cpp > base_cpp else (1 - 0.01) * base_cpp + 0.01 * cpp

            # Collect scatter baseline (warm-up phase)
            if sc_mean is None:
                sc_baseline_feats.append(feat)
                if len(sc_baseline_feats) >= SCATTER_WARM_N:
                    feats_arr = np.stack(sc_baseline_feats)
                    sc_mean = np.mean(feats_arr, axis=0)
                    sc_std  = np.std(feats_arr, axis=0) + 1e-8
                    if show_frames:
                        print(f"  >>> SCATTER BASELINE WARM at t={t:.1f}s "
                              f"({len(sc_baseline_feats)} clean frames)")
            else:
                # EMA update: only if this frame scores clean on scatter too
                sc_score_check = scatter_strain(feat, sc_mean, sc_std)
                if sc_score_check < 0.3:
                    sc_mean = (1 - SCATTER_ALPHA) * sc_mean + SCATTER_ALPHA * feat
                    # std doesn't update (keep baseline spread fixed after warmup)

            clean_frames += 1

        # Compute final scores
        v8_score, _, _ = strain_v8(shimmer_pct, cpp, base_shim, base_cpp)

        if sc_mean is not None:
            sc_score = scatter_strain(feat, sc_mean, sc_std)
            # Max-fusion: strain if EITHER signal fires.
            # fusion_weight blends between max (0.0) and weighted-avg (1.0):
            #   fusion_weight=0.0 → max(v8, scatter)  (most sensitive)
            #   fusion_weight=1.0 → v8 only
            max_score  = max(v8_score, sc_score)
            wavg_score = fusion_weight * v8_score + (1 - fusion_weight) * sc_score
            # Use weighted interpolation between max and wavg
            # fusion_weight controls how much we trust max vs wavg
            fuse_score = min(1.0, (1 - fusion_weight) * max_score + fusion_weight * wavg_score)
        else:
            # Scatter not warmed up yet — use v8 only
            sc_score = float('nan')
            fuse_score = v8_score

        zone = zone_of(fuse_score)
        frames.append({
            "t": t, "zone": zone, "score": fuse_score,
            "v8": v8_score, "scatter": sc_score,
            "shim": shimmer_pct, "cpp": cpp,
        })

        if show_frames:
            zc = ZONE_COLOR.get(zone, "")
            bar_ch = ZONE_BAR.get(zone, "?")
            bar_w = min(20, max(1, int(fuse_score * 20)))
            bar = bar_ch * bar_w
            shim_s = f"{shimmer_pct:>6.2f}" if not math.isnan(shimmer_pct) else f"{'nan':>6}"
            cpp_s  = f"{cpp:>7.3f}"          if not math.isnan(cpp) else f"{'nan':>7}"
            sc_s   = f"{sc_score:>6.3f}"     if not math.isnan(sc_score) else f"{'--':>6}"
            print(f"  {t:>5.1f}  {shim_s}  {cpp_s}  {sc_s}  {v8_score:>6.3f}  "
                  f"{zc}{fuse_score:>6.3f}{ZONE_COLOR['reset']}  {zc}{zone:<7}{ZONE_COLOR['reset']}  {zc}{bar}{ZONE_COLOR['reset']}")

    # Summary
    voiced = [f for f in frames if f["zone"] != "silent"]
    if not voiced:
        print("  (no voiced frames)")
        return None

    raw = [f["score"] for f in voiced]
    smoothed = [float(np.median(raw[max(0, j-1):j+2])) for j in range(len(raw))]
    p80 = float(np.percentile(smoothed, 80))
    dominant = zone_of(p80)

    zone_counts = {"green": 0, "yellow": 0, "red": 0}
    for s in smoothed:
        z = zone_of(s)
        if z in zone_counts:
            zone_counts[z] += 1

    # Pure scatter P80 (only frames where scatter was warmed up)
    sc_vals = [f["scatter"] for f in voiced if not (isinstance(f["scatter"], float) and math.isnan(f["scatter"]))]
    sc_p80 = float(np.percentile(sc_vals, 80)) if sc_vals else float('nan')
    v8_p80 = float(np.percentile([f["v8"] for f in voiced], 80))

    print(f"\n  SUMMARY: mean={np.mean(smoothed):.3f}  p80={p80:.3f}  max={np.max(smoothed):.3f}  "
          f"dominant={dominant}")
    print(f"  Breakdown: scatter_p80={sc_p80:.3f}  v8_p80={v8_p80:.3f}  "
          f"scatter_warmed={'yes' if sc_mean is not None else 'no'}  clean_frames={clean_frames}")
    print(f"  Zones: green={zone_counts['green']}  yellow={zone_counts['yellow']}  "
          f"red={zone_counts['red']}  voiced={len(voiced)}")

    dc = ZONE_COLOR.get(dominant, "")
    tc = ZONE_COLOR.get(true_label, "")
    ok = "✓" if dominant == true_label or true_label == "mixed" else "✗"
    print(f"  Result: {ok}  predicted={dc}{dominant}{ZONE_COLOR['reset']}  "
          f"true={tc}{true_label}{ZONE_COLOR['reset']}")

    return {"dominant": dominant, "true": true_label,
            "p80": p80, "scatter_p80": sc_p80, "v8_p80": v8_p80,
            "zone_counts": zone_counts, "clean_frames": clean_frames}


# ---------------------------------------------------------------------------
# Liza Jane segment analysis — the key test
# ---------------------------------------------------------------------------

def analyze_liza_jane_segments(baseline_mean=None, baseline_std=None, fusion_weight=0.5):
    """
    Liza Jane: Daniel sang verses easy, chorus hard.
    The longer recording is ~67s. Analyze in 10s segments to see verse/chorus separation.
    Uses session-adaptive scatter baseline (same as analyze_clip_v9).
    """
    SCATTER_WARM_N = 15
    SCATTER_ALPHA  = 0.05

    path = SONG_DIR / "Danny - Liza Jane R1 (longer).m4a"
    if not path.exists():
        print(f"MISSING: {path}")
        return

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / SR
    seg_size = 10.0  # 10s segments

    print(f"\n{'═'*75}")
    print(f"  LIZA JANE SEGMENT ANALYSIS (10s windows)  |  total={duration:.1f}s")
    print(f"  (verses=easy, chorus=harder — expecting green→yellow/red transition)")
    print(f"  fusion_weight={fusion_weight}")
    print(f"{'═'*75}")

    base_shim = DANIEL_SHIM_SEED
    base_cpp  = DANIEL_CPP_SEED
    sc_baseline_feats = []
    sc_mean = None
    sc_std  = None

    voiced_run   = 0
    clean_frames = 0

    seg_start = 0.0
    seg_results = []
    n_chunks = len(y) // CHUNK
    seg_scores_v   = []  # fused scores for current segment
    seg_scores_sc  = []  # scatter-only
    seg_scores_v8  = []  # v8-only

    for i in range(n_chunks):
        chunk = y[i * CHUNK:(i + 1) * CHUNK]
        t = i * 0.1
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Segment boundary
        if (t >= seg_start + seg_size or i == n_chunks - 1) and seg_scores_v:
            smoothed = [float(np.median(seg_scores_v[max(0, j-1):j+2]))
                        for j in range(len(seg_scores_v))]
            p80 = float(np.percentile(smoothed, 80))
            sc_p80 = float(np.percentile(seg_scores_sc, 80)) if seg_scores_sc else float('nan')
            v8_p80 = float(np.percentile(seg_scores_v8, 80)) if seg_scores_v8 else float('nan')
            z = zone_of(p80)
            zc = ZONE_COLOR.get(z, "")
            bar_w = min(30, max(1, int(p80 * 30)))
            bar = ZONE_BAR.get(z, "?") * bar_w
            warm_str = "warm" if sc_mean is not None else "v8only"
            sc_str = f"{sc_p80:.3f}" if not math.isnan(sc_p80) else "--   "
            print(f"  t={seg_start:>4.0f}-{seg_start+seg_size:>4.0f}s  "
                  f"p80={p80:.3f}  sc={sc_str}  v8={v8_p80:.3f}  "
                  f"{zc}{z:<7}{ZONE_COLOR['reset']}  {zc}{bar}{ZONE_COLOR['reset']}  [{warm_str}]")
            seg_results.append({"t_start": seg_start, "p80": p80, "zone": z,
                                 "sc_p80": sc_p80, "v8_p80": v8_p80})
            seg_start += seg_size
            seg_scores_v, seg_scores_sc, seg_scores_v8 = [], [], []

        if rms < SILENCE_RMS:
            voiced_run = 0
            continue

        voiced_run += 1
        onset_gated = voiced_run < 3

        feat = scatter_features(chunk)

        if onset_gated:
            shimmer_pct = float('nan')
            cpp = float('nan')
        else:
            shimmer_pct = praat_shimmer(chunk)
            cpp = compute_cpp(chunk)

        t_score, _, _ = strain_v8(shimmer_pct, cpp, base_shim, base_cpp)
        is_clean = (t_score < BASELINE_MAX_SCORE and
                    not math.isnan(shimmer_pct) and not math.isnan(cpp))

        if is_clean:
            a = BASELINE_EMA_ALPHA
            base_shim = (1 - a) * base_shim + a * shimmer_pct
            base_cpp = (1 - a) * base_cpp + a * cpp if cpp > base_cpp else (1 - 0.01) * base_cpp + 0.01 * cpp
            if sc_mean is None:
                sc_baseline_feats.append(feat)
                if len(sc_baseline_feats) >= SCATTER_WARM_N:
                    feats_arr = np.stack(sc_baseline_feats)
                    sc_mean = np.mean(feats_arr, axis=0)
                    sc_std  = np.std(feats_arr, axis=0) + 1e-8
                    print(f"  [scatter warm at t={t:.1f}s, {len(sc_baseline_feats)} clean frames]")
            else:
                sc_score_check = scatter_strain(feat, sc_mean, sc_std)
                if sc_score_check < 0.3:
                    sc_mean = (1 - SCATTER_ALPHA) * sc_mean + SCATTER_ALPHA * feat
            clean_frames += 1

        v8_score, _, _ = strain_v8(shimmer_pct, cpp, base_shim, base_cpp)
        if sc_mean is not None:
            sc_score = scatter_strain(feat, sc_mean, sc_std)
            max_score  = max(v8_score, sc_score)
            wavg_score = fusion_weight * v8_score + (1 - fusion_weight) * sc_score
            fuse_score = min(1.0, (1 - fusion_weight) * max_score + fusion_weight * wavg_score)
        else:
            sc_score = float('nan')
            fuse_score = v8_score

        seg_scores_v.append(fuse_score)
        if not math.isnan(sc_score):
            seg_scores_sc.append(sc_score)
        seg_scores_v8.append(v8_score)

    print(f"\n  Interpretation (clean_frames={clean_frames}):")
    if seg_results:
        zones = [r["zone"] for r in seg_results]
        has_progression = any(z in ("yellow", "red") for z in zones[len(zones)//2:])
        if has_progression:
            print(f"  ✓ Verse→chorus progression detected")
        else:
            print(f"  ✗ No clear verse/chorus separation")
        print(f"  Segments: {' → '.join(z.upper()[:1] for z in zones)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all",       action="store_true", help="Include anchor clips")
    ap.add_argument("--calibrate", action="store_true", help="Show baseline stats only")
    ap.add_argument("--quiet",     action="store_true", help="No per-frame output")
    ap.add_argument("--weight",    type=float, default=0.5,
                    help="Fusion weight: 0=pure scatter, 1=pure v8 (default 0.5)")
    ap.add_argument("--liza",      action="store_true", help="Liza Jane segment analysis only")
    args = ap.parse_args()

    print("Building scatter baseline from Easy 2 anchor...")
    baseline_mean, baseline_std = build_scatter_baseline(verbose=True)
    if baseline_mean is None:
        print("Failed to build baseline.")
        return

    if args.calibrate:
        return

    if args.liza:
        analyze_liza_jane_segments(baseline_mean, baseline_std, fusion_weight=args.weight)
        return

    clips = []
    if args.all:
        clips = [(ANCHOR_DIR / f, lbl) for f, lbl in ANCHORS]
    clips += [(SONG_DIR / f, lbl) for f, lbl in SONGS]

    results = []
    for path, label in clips:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        r = analyze_clip_v9(path, label, baseline_mean, baseline_std,
                            show_frames=not args.quiet,
                            fusion_weight=args.weight)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n{'═'*75}")
        print(f"  OVERALL SUMMARY  (fusion_weight={args.weight})")
        for r in results:
            dc = ZONE_COLOR.get(r["dominant"], "")
            print(f"    {r['true']:>7} → {dc}{r['dominant']:<7}{ZONE_COLOR['reset']}  "
                  f"p80={r['p80']:.3f}  scatter={r['scatter_p80']:.3f}  v8={r['v8_p80']:.3f}")


if __name__ == "__main__":
    main()
