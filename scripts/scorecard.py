#!/usr/bin/env python3
"""
Strain formula accuracy scorecard.

Loads reviewed annotations + anchor clips, runs HNR/shimmer via parselmouth,
compares predicted zone vs. ground truth, reports % accuracy.

Usage:
    python scripts/scorecard.py             # current formula (v3)
    python scripts/scorecard.py --adaptive  # + session-adaptive variant
    python scripts/scorecard.py --sweep     # threshold sweep to find best values
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ANNOT_DIR = ROOT / "docs" / "annotations"
AUDIO_DIR = ROOT / "Vocal test recording sessions"
ANCHOR_DIR = AUDIO_DIR / "Anchors"

# Reviewed annotation files
REVIEWED = [
    "Danny - Chris Young R1_reviewed.json",
    "Danny - Liza Jane R1 (longer)_reviewed.json",
    "Danny - Runnin down a dream R1_reviewed.json",
]

# Anchor clips: (filename, true zone)
ANCHORS = [
    ("Easy 2.m4a",    "green"),
    ("Medium 1.m4a",  "green"),
    ("hard push.m4a", "yellow"),
    ("Rough 1.m4a",   "red"),
]

# ── Formula v3 thresholds ──────────────────────────────────────────────────────
STRAIN_GREEN  = 0.35
STRAIN_YELLOW = 0.55


def load_audio(path: Path, sr=44100):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def praat_features(y, sr=44100):
    """Return (mean_hnr_db, mean_shimmer_pct) for a numpy float32 array."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_db = call(harmonicity, "Get mean", 0.0, 0.0)
    if hnr_db is None or np.isnan(hnr_db):
        hnr_db = 0.0

    # Shimmer
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    try:
        shimmer = call([snd, pp], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_pct = (shimmer if shimmer else 0.0) * 100.0
    except Exception:
        shimmer_pct = 0.0

    return hnr_db, shimmer_pct


def strain_v3(hnr_db, shimmer_pct):
    """Current formula: shimmer*0.6 + hnr_norm*0.4 (absolute thresholds)."""
    hnr_norm  = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
    shim_norm = min(1.0, shimmer_pct / 10.0)
    return shim_norm * 0.6 + hnr_norm * 0.4


def strain_adaptive(hnr_db, shimmer_pct, base_hnr, base_shim):
    """Session-adaptive: measure deviation above personal baseline (old direction — HNR drop)."""
    hnr_dev  = max(0.0, base_hnr - hnr_db)
    shim_dev = max(0.0, shimmer_pct - base_shim)
    hnr_norm  = min(1.0, hnr_dev  / 10.0)
    shim_norm = min(1.0, shim_dev / 10.0)
    return shim_norm * 0.6 + hnr_norm * 0.4


def strain_v4(hnr_db, shimmer_pct, base_hnr, base_shim):
    """Formula v4: Daniel-specific pressed + rough phonation model.

    Two independent strain signals:
    - HNR RISE above baseline = pressed phonation (Daniel squeezes, HNR goes UP)
    - shimmer SPIKE above baseline = rough phonation (blown-out, HNR may drop)

    Either signal alone can indicate strain.
    Normalization: 10dB HNR rise → 1.0, 10% shimmer rise → 1.0
    """
    hnr_press = max(0.0, hnr_db - base_hnr) / 10.0       # HNR rise above baseline
    shim_dev  = max(0.0, shimmer_pct - base_shim) / 10.0  # shimmer rise above baseline
    return min(1.0, hnr_press + shim_dev)


def score_to_zone(score, green_thresh=STRAIN_GREEN, yellow_thresh=STRAIN_YELLOW):
    if score < green_thresh:
        return "green"
    elif score < yellow_thresh:
        return "yellow"
    else:
        return "red"


def evaluate(samples, formula_fn, label="formula"):
    """
    samples: list of (true_zone, hnr_db, shimmer_pct, [base_hnr, base_shim], clip_name)
    Returns (correct, total, per_zone_results)
    """
    correct = 0
    total = 0
    per_zone = {"green": [0, 0], "yellow": [0, 0], "red": [0, 0]}  # [correct, total]
    rows = []

    for item in samples:
        true_zone = item["true_zone"]
        hnr, shim = item["hnr"], item["shim"]
        name = item["name"]

        if formula_fn.__code__.co_varnames[:4] == ("hnr_db", "shimmer_pct", "base_hnr", "base_shim"):
            score = formula_fn(hnr, shim, item.get("base_hnr", 18.0), item.get("base_shim", 7.0))
        else:
            score = formula_fn(hnr, shim)

        pred_zone = score_to_zone(score)
        ok = pred_zone == true_zone
        correct += ok
        total += 1
        per_zone[true_zone][1] += 1
        per_zone[true_zone][0] += ok
        rows.append((name, true_zone, pred_zone, score, hnr, shim, ok))

    return correct, total, per_zone, rows


def print_results(rows, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Clip':<32} {'True':>6} {'Pred':>6} {'Score':>6} {'HNR':>6} {'Shim':>6}  {'':>3}")
    print(f"  {'-'*67}")
    for (name, true_z, pred_z, score, hnr, shim, ok) in rows:
        tick = "✓" if ok else "✗"
        name_s = name[:31]
        print(f"  {name_s:<32} {true_z:>6} {pred_z:>6} {score:>6.3f} {hnr:>6.1f} {shim:>6.2f}  {tick}")


def threshold_sweep(samples):
    """Find the green/yellow threshold pair that maximizes accuracy."""
    best = (0, 0.30, 0.50)
    for g in np.arange(0.20, 0.55, 0.025):
        for y in np.arange(g + 0.05, 0.90, 0.025):
            correct = sum(
                1 for s in samples
                if score_to_zone(strain_v3(s["hnr"], s["shim"]), g, y) == s["true_zone"]
            )
            if correct > best[0]:
                best = (correct, g, y)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", action="store_true", help="Also run session-adaptive formula")
    ap.add_argument("--sweep",    action="store_true", help="Threshold sweep on v3")
    ap.add_argument("--anchors-only", action="store_true", help="Only run anchor clips")
    args = ap.parse_args()

    samples = []

    # ── Anchor clips ────────────────────────────────────────────────────────────
    print("\n[Loading anchor clips...]")
    anchor_samples = []
    for fname, true_zone in ANCHORS:
        path = ANCHOR_DIR / fname
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        y = load_audio(path)
        hnr, shim = praat_features(y)
        anchor_samples.append({"name": fname, "true_zone": true_zone, "hnr": hnr, "shim": shim})
        print(f"  {fname:<28}  HNR={hnr:>6.1f}dB  shim={shim:>5.2f}%  label={true_zone}")

    if args.anchors_only:
        samples = anchor_samples
    else:
        # ── Reviewed annotation segments ──────────────────────────────────────────
        print("\n[Loading reviewed annotation segments...]")
        for ann_file in REVIEWED:
            ann_path = ANNOT_DIR / ann_file
            if not ann_path.exists():
                print(f"  MISSING: {ann_path}")
                continue

            with open(ann_path) as f:
                ann = json.load(f)

            audio_fname = ann["file"]
            audio_path = AUDIO_DIR / audio_fname
            if not audio_path.exists():
                print(f"  MISSING audio: {audio_path}")
                continue

            y_full = load_audio(audio_path)
            sr = 44100
            segs = ann.get("segment_ratings", [])
            print(f"\n  {audio_fname} — {len(segs)} segments")

            for seg in segs:
                t0, t1 = seg["t_start"], seg["t_end"]
                true_zone = seg["zone"]
                i0 = int(t0 * sr)
                i1 = min(int(t1 * sr), len(y_full))
                chunk = y_full[i0:i1]

                # Skip very quiet segments (silence/between phrases)
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.005:
                    continue
                if true_zone not in ("green", "yellow", "red"):
                    continue

                hnr, shim = praat_features(chunk)
                name = f"{audio_fname[:20]}..  {t0:.0f}-{t1:.0f}s"
                samples.append({
                    "name": name,
                    "true_zone": true_zone,
                    "hnr": hnr,
                    "shim": shim,
                    "source": "annotation",
                })

        samples.extend(anchor_samples)

    if not samples:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    total_n = len(samples)
    print(f"\nTotal samples: {total_n} "
          f"({sum(1 for s in samples if s['true_zone']=='green')} green, "
          f"{sum(1 for s in samples if s['true_zone']=='yellow')} yellow, "
          f"{sum(1 for s in samples if s['true_zone']=='red')} red)")

    # ── Evaluate v3 ─────────────────────────────────────────────────────────────
    correct_v3, total_v3, per_zone_v3, rows_v3 = evaluate(samples, strain_v3)
    print_results(rows_v3, f"Formula v3  (shim*0.6 + hnr_norm*0.4, thresholds {STRAIN_GREEN}/{STRAIN_YELLOW})")
    print(f"\n  ACCURACY: {correct_v3}/{total_v3} = {100*correct_v3/total_v3:.1f}%")
    for z in ["green", "yellow", "red"]:
        c, t = per_zone_v3[z]
        pct = f"{100*c/t:.0f}%" if t else "N/A"
        print(f"    {z:>6}: {c}/{t} ({pct})")

    # ── Evaluate v4 (Daniel-specific pressed + rough model) ─────────────────────
    if args.adaptive:
        green_anchors = [s for s in anchor_samples if s["true_zone"] == "green"]
        if green_anchors:
            base_hnr  = np.mean([s["hnr"] for s in green_anchors])
            base_shim = np.mean([s["shim"] for s in green_anchors])
        else:
            base_hnr, base_shim = 18.0, 7.0
        print(f"\n  Session baseline: HNR={base_hnr:.1f}dB, shimmer={base_shim:.2f}%")
        for s in samples:
            s["base_hnr"] = base_hnr
            s["base_shim"] = base_shim

        def _v4(hnr_db, shimmer_pct):
            return strain_v4(hnr_db, shimmer_pct, base_hnr, base_shim)

        correct_v4, total_v4, per_zone_v4, rows_v4 = evaluate(samples, _v4)
        print_results(rows_v4, f"Formula v4  (hnr_press + shim_dev, baseline HNR={base_hnr:.1f} shim={base_shim:.2f})")
        print(f"\n  ACCURACY: {correct_v4}/{total_v4} = {100*correct_v4/total_v4:.1f}%")
        for z in ["green", "yellow", "red"]:
            c, t = per_zone_v4[z]
            pct = f"{100*c/t:.0f}%" if t else "N/A"
            print(f"    {z:>6}: {c}/{t} ({pct})")

        # v4 anchor deep-dive
        print("\n  --- v4 ANCHOR DEEP-DIVE ---")
        for item in anchor_samples:
            score = _v4(item["hnr"], item["shim"])
            press = max(0.0, item["hnr"] - base_hnr) / 10.0
            shim_d = max(0.0, item["shim"] - base_shim) / 10.0
            pred = score_to_zone(score)
            ok = "✓" if pred == item["true_zone"] else "✗"
            print(f"  {item['name']:<28} HNR={item['hnr']:>5.1f}(+{item['hnr']-base_hnr:+.1f}) "
                  f"shim={item['shim']:>5.2f}(+{item['shim']-base_shim:+.2f}) "
                  f"press={press:.3f} shim_dev={shim_d:.3f} score={score:.3f} "
                  f"pred={pred} true={item['true_zone']} {ok}")

    # ── Evaluate adaptive (old) ──────────────────────────────────────────────────
    if False and args.adaptive:
        # Compute baseline from green anchor clips (Easy 2, Medium 1)
        green_anchors = [s for s in anchor_samples if s["true_zone"] == "green"]
        if green_anchors:
            base_hnr = np.mean([s["hnr"] for s in green_anchors])
            base_shim = np.mean([s["shim"] for s in green_anchors])
        else:
            base_hnr, base_shim = 18.0, 7.0
        print(f"\n  Session baseline: HNR={base_hnr:.1f}dB, shimmer={base_shim:.2f}%")

        # Inject baseline into samples
        for s in samples:
            s["base_hnr"] = base_hnr
            s["base_shim"] = base_shim

        def _adaptive(hnr_db, shimmer_pct):
            return strain_adaptive(hnr_db, shimmer_pct, base_hnr, base_shim)

        correct_ad, total_ad, per_zone_ad, rows_ad = evaluate(samples, _adaptive)
        print_results(rows_ad, f"Adaptive  (deviation above baseline HNR={base_hnr:.1f} shim={base_shim:.2f})")
        print(f"\n  ACCURACY: {correct_ad}/{total_ad} = {100*correct_ad/total_ad:.1f}%")
        for z in ["green", "yellow", "red"]:
            c, t = per_zone_ad[z]
            pct = f"{100*c/t:.0f}%" if t else "N/A"
            print(f"    {z:>6}: {c}/{t} ({pct})")

    # ── Threshold sweep ──────────────────────────────────────────────────────────
    if args.sweep:
        print("\n[Threshold sweep on v3...]")
        best_n, best_g, best_y = threshold_sweep(samples)
        print(f"  Best: green<{best_g:.3f}, yellow<{best_y:.3f} → {best_n}/{total_n} correct ({100*best_n/total_n:.1f}%)")

        # Re-evaluate with best thresholds
        def _v3_best(hnr_db, shimmer_pct):
            s = strain_v3(hnr_db, shimmer_pct)
            return s  # use the sweep result for display only

        rows_best = []
        for item in samples:
            score = strain_v3(item["hnr"], item["shim"])
            pred = score_to_zone(score, best_g, best_y)
            ok = pred == item["true_zone"]
            rows_best.append((item["name"], item["true_zone"], pred, score, item["hnr"], item["shim"], ok))
        print_results(rows_best, f"v3 with optimal thresholds green<{best_g:.3f} yellow<{best_y:.3f}")


if __name__ == "__main__":
    main()
