#!/usr/bin/env python3
"""
Calibration analysis: find optimal EARS thresholds from reviewed annotations.

Loads all *_reviewed.json AND yt_*.json annotation files, runs EARS on audio,
then for each rated segment computes the EARS score and shows distributions
per zone. Suggests thresholds that maximize agreement.

Usage:
  python scripts/calibrate.py
"""
import os
import sys
import json
import math
import tempfile
import subprocess

import numpy as np

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "docs", "annotations")
RECORDINGS_DIR  = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception", "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SAMPLE_RATE  = 44100
WINDOW_S     = 1.0
HOP_S        = 0.25
SILENCE_RMS  = 0.008


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


def run_ears(audio):
    """Run EARS strain analysis → list of {t, strain, voiced}."""
    from mel_extractor import MelExtractor
    from frequency_explorer import compute_emotion_properties
    try:
        import parselmouth
        from parselmouth import praat
        PRAAT = True
    except ImportError:
        PRAAT = False

    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    window_samples = int(WINDOW_S * SAMPLE_RATE)
    hop_samples    = int(HOP_S * SAMPLE_RATE)
    results = []
    i = 0
    while i + window_samples <= len(audio):
        chunk = audio[i:i + window_samples]
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        t     = i / SAMPLE_RATE

        if rms < SILENCE_RMS:
            results.append({"t": t, "strain": None, "voiced": False})
            i += hop_samples
            continue

        try:
            mel    = mel_ex.extract_from_audio(chunk)
            em     = compute_emotion_properties(mel)
            tonos_raw = em.get("tension", 0.5)
            f = float(tonos_raw) if tonos_raw is not None else 0.5
            tonos = 0.5 if (math.isnan(f) or math.isinf(f)) else max(0.0, min(1.0, f))
        except Exception:
            tonos = 0.5

        hnr_db, shimmer_pct = 20.0, 0.0
        if PRAAT:
            try:
                snd  = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
                harm = snd.to_harmonicity()
                vals = harm.values[0]
                valid = vals[vals > -200]
                hnr_db = float(np.mean(valid)) if len(valid) else 20.0
                hnr_db = max(0.0, hnr_db)
                pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
                shimmer_pct = max(0.0, praat.call(
                    [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
                ) * 100)
            except Exception:
                pass

        hnr_norm     = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
        shimmer_norm = min(1.0, shimmer_pct / 10.0)
        strain       = tonos * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10
        results.append({"t": t, "strain": strain, "voiced": True, "tonos": tonos,
                         "hnr_db": hnr_db, "hnr_norm": hnr_norm,
                         "shimmer_pct": shimmer_pct, "shimmer_norm": shimmer_norm})
        i += hop_samples

    return results


def ears_score_at(results, t_center):
    """Return the EARS strain score nearest to t_center."""
    voiced = [r for r in results if r["voiced"]]
    if not voiced:
        return None
    nearest = min(voiced, key=lambda r: abs(r["t"] - t_center))
    return nearest


def run_ears_on_chunk(audio):
    """Run EARS on a single audio chunk (already loaded). Returns one result dict."""
    from mel_extractor import MelExtractor
    from frequency_explorer import compute_emotion_properties
    try:
        import parselmouth
        from parselmouth import praat
        PRAAT = True
    except ImportError:
        PRAAT = False

    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    # Use center 1s of the chunk (or full if shorter)
    ws = int(1.0 * SAMPLE_RATE)
    if len(audio) >= ws:
        start = (len(audio) - ws) // 2
        chunk = audio[start:start + ws]
    else:
        chunk = audio

    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < SILENCE_RMS:
        return None

    try:
        mel    = mel_ex.extract_from_audio(chunk)
        em     = compute_emotion_properties(mel)
        tonos_raw = em.get("tension", 0.5)
        f = float(tonos_raw) if tonos_raw is not None else 0.5
        tonos = 0.5 if (math.isnan(f) or math.isinf(f)) else max(0.0, min(1.0, f))
    except Exception:
        tonos = 0.5

    hnr_db, shimmer_pct = 20.0, 0.0
    if PRAAT:
        try:
            snd  = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
            harm = snd.to_harmonicity()
            vals = harm.values[0]
            valid = vals[vals > -200]
            hnr_db = float(np.mean(valid)) if len(valid) else 20.0
            hnr_db = max(0.0, hnr_db)
            pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
            shimmer_pct = max(0.0, praat.call(
                [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100)
        except Exception:
            pass

    hnr_norm     = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
    shimmer_norm = min(1.0, shimmer_pct / 10.0)
    strain       = tonos * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10
    return {"strain": strain, "voiced": True, "tonos": tonos,
            "hnr_db": hnr_db, "hnr_norm": hnr_norm,
            "shimmer_pct": shimmer_pct, "shimmer_norm": shimmer_norm}


def main():
    # Find reviewed annotation files (personal recordings)
    reviewed_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.endswith("_reviewed.json")
    ])
    # Find YouTube harvest annotation files
    yt_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.startswith("yt_") and f.endswith(".json")
    ])

    ann_files = reviewed_files + yt_files
    if not ann_files:
        print("No annotation files found — run review.py or yt_harvest.py first.")
        sys.exit(1)

    print(f"Found: {len(reviewed_files)} reviewed files, {len(yt_files)} YT harvest files")

    all_points = []   # {zone, strain, tonos, hnr_norm, shimmer_norm, file, t, source}

    # ── Personal reviewed recordings ──────────────────────────────────────────
    for ann_file in reviewed_files:
        ann = json.load(open(os.path.join(ANNOTATIONS_DIR, ann_file)))
        name = ann["file"]
        segs = ann.get("segment_ratings", [])
        rated = [s for s in segs if s["zone"]]
        if not rated:
            continue

        audio_path = os.path.join(RECORDINGS_DIR, name)
        if not os.path.exists(audio_path):
            for f in os.listdir(RECORDINGS_DIR):
                if os.path.splitext(name)[0].lower() in f.lower():
                    audio_path = os.path.join(RECORDINGS_DIR, f)
                    break
        if not os.path.exists(audio_path):
            print(f"  Audio not found: {name} — skipping")
            continue

        print(f"Analyzing {name} ({len(rated)} rated segments)...")
        audio = load_audio(audio_path)
        ears  = run_ears(audio)

        for seg in rated:
            t_mid = (seg["t_start"] + seg["t_end"]) / 2.0
            r     = ears_score_at(ears, t_mid)
            if r is None:
                continue
            all_points.append({
                "zone":         seg["zone"],
                "strain":       r["strain"],
                "tonos":        r.get("tonos", 0),
                "hnr_db":       r.get("hnr_db", 20.0),
                "hnr_norm":     r.get("hnr_norm", 0),
                "shimmer_pct":  r.get("shimmer_pct", 0),
                "shimmer_norm": r.get("shimmer_norm", 0),
                "file":         name,
                "t":            t_mid,
                "replays":      seg.get("replays", 0),
                "source":       "personal",
            })

    # ── YouTube harvest segments ───────────────────────────────────────────────
    import librosa
    for ann_file in yt_files:
        ann = json.load(open(os.path.join(ANNOTATIONS_DIR, ann_file)))
        segs = ann.get("segment_ratings", [])
        rated = [s for s in segs if s.get("zone") and s.get("audio")]
        if not rated:
            continue

        title = ann.get("title", ann_file)[:50]
        print(f"Analyzing YT '{title}' ({len(rated)} segments)...")
        ok = 0
        for seg in rated:
            seg_path = seg["audio"]
            if not os.path.exists(seg_path):
                continue
            try:
                audio, _ = librosa.load(seg_path, sr=SAMPLE_RATE, mono=True)
                r = run_ears_on_chunk(audio)
                if r is None:
                    continue
                all_points.append({
                    "zone":         seg["zone"],
                    "strain":       r["strain"],
                    "tonos":        r.get("tonos", 0),
                    "hnr_db":       r.get("hnr_db", 20.0),
                    "hnr_norm":     r.get("hnr_norm", 0),
                    "shimmer_pct":  r.get("shimmer_pct", 0),
                    "shimmer_norm": r.get("shimmer_norm", 0),
                    "file":         f"YT:{ann.get('video_id','')}",
                    "t":            seg["t_start"],
                    "replays":      0,
                    "source":       "yt",
                    "keyword":      seg.get("keyword", ""),
                })
                ok += 1
            except Exception as e:
                pass
        print(f"  {ok}/{len(rated)} segments processed")

    if not all_points:
        print("No data points found.")
        sys.exit(1)

    personal = [p for p in all_points if p.get("source") == "personal"]
    yt_pts   = [p for p in all_points if p.get("source") == "yt"]
    print(f"\nTotal: {len(all_points)} points ({len(personal)} personal, {len(yt_pts)} YT)")

    # ── Per-zone statistics ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("EARS SCORE DISTRIBUTION BY ANNOTATION ZONE (ALL DATA)")
    print("=" * 60)

    zones = ["green", "yellow", "red"]
    zone_data = {z: [p["strain"] for p in all_points if p["zone"] == z] for z in zones}

    for z in zones:
        vals = zone_data[z]
        if not vals:
            print(f"\n{z.upper()}: no data")
            continue
        arr = np.array(vals)
        print(f"\n{z.upper()} (n={len(vals)}):")
        print(f"  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}  "
              f"median={np.median(arr):.3f}  std={arr.std():.3f}")
        print(f"  25th pct={np.percentile(arr,25):.3f}  75th pct={np.percentile(arr,75):.3f}")
        # ASCII histogram
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(arr, bins=bins)
        bar_max = max(hist) if max(hist) > 0 else 1
        for i, (lo, hi, count) in enumerate(zip(bins, bins[1:], hist)):
            bar = "█" * int(count / bar_max * 20)
            print(f"  {lo:.1f}-{hi:.1f}  {bar:20s} {count}")

    # ── Suggest thresholds ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("THRESHOLD SUGGESTIONS")
    print("=" * 60)

    g_vals = np.array(zone_data["green"])   if zone_data["green"]  else np.array([])
    y_vals = np.array(zone_data["yellow"])  if zone_data["yellow"] else np.array([])
    r_vals = np.array(zone_data["red"])     if zone_data["red"]    else np.array([])

    if len(g_vals) and len(y_vals):
        # Green/yellow boundary: midpoint between green 75th pct and yellow 25th pct
        g75 = float(np.percentile(g_vals, 75))
        y25 = float(np.percentile(y_vals, 25))
        green_thresh = round((g75 + y25) / 2.0, 3)
        print(f"\nGreen 75th pct:  {g75:.3f}")
        print(f"Yellow 25th pct: {y25:.3f}")
        print(f"→ Suggested STRAIN_GREEN threshold: {green_thresh:.3f}  (was 0.50)")

        # Sweep to find best accuracy on green/yellow split
        best_acc, best_t = 0, 0.50
        for t in np.linspace(0.2, 0.8, 61):
            correct = sum(1 for v in g_vals if v < t) + sum(1 for v in y_vals if v >= t)
            acc = correct / (len(g_vals) + len(y_vals))
            if acc > best_acc:
                best_acc, best_t = acc, t
        print(f"→ Best accuracy threshold G/Y: {best_t:.3f}  (accuracy {best_acc*100:.0f}%)")

    if len(y_vals) and len(r_vals):
        y75 = float(np.percentile(y_vals, 75))
        r25 = float(np.percentile(r_vals, 25))
        yellow_thresh = round((y75 + r25) / 2.0, 3)
        print(f"\nYellow 75th pct: {y75:.3f}")
        print(f"Red 25th pct:    {r25:.3f}")
        print(f"→ Suggested STRAIN_YELLOW threshold: {yellow_thresh:.3f}  (was 0.68)")

        best_acc, best_t = 0, 0.68
        for t in np.linspace(0.3, 0.9, 61):
            correct = sum(1 for v in y_vals if v < t) + sum(1 for v in r_vals if v >= t)
            acc = correct / (len(y_vals) + len(r_vals))
            if acc > best_acc:
                best_acc, best_t = acc, t
        print(f"→ Best accuracy threshold Y/R: {best_t:.3f}  (accuracy {best_acc*100:.0f}%)")

    # ── Feature breakdown ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FEATURE MEANS BY ZONE")
    print("=" * 60)
    print(f"  {'Zone':8s}  {'Strain':8s}  {'Tonos':7s}  {'HNR_db':8s}  {'HNR_norm':9s}  {'Shimmer%':9s}  {'Shim_norm':9s}")
    for z in zones:
        pts = [p for p in all_points if p["zone"] == z]
        if not pts:
            continue
        s_mean   = np.mean([p["strain"] for p in pts])
        t_mean   = np.mean([p["tonos"] for p in pts])
        h_mean   = np.mean([p["hnr_db"] for p in pts])
        hn_mean  = np.mean([p["hnr_norm"] for p in pts])
        sh_mean  = np.mean([p["shimmer_pct"] for p in pts])
        shn_mean = np.mean([p["shimmer_norm"] for p in pts])
        print(f"  {z.upper():8s}  {s_mean:.3f}     {t_mean:.3f}    {h_mean:6.1f}    {hn_mean:.3f}      {sh_mean:6.2f}    {shn_mean:.3f}")

    # ── HNR vs Zone analysis ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RAW HNR_DB vs ANNOTATION ZONE (should decrease GREEN→RED if HNR tracks strain)")
    print("=" * 60)
    print(f"  Clinical norm: HNR > 20 dB = healthy vocal fold closure")
    print(f"  Lower HNR = more noise = more strain expected")
    print()
    for z in zones:
        pts = [p for p in all_points if p["zone"] == z]
        if not pts:
            continue
        vals = np.array([p["hnr_db"] for p in pts])
        indicator = "↓ good" if z == "red" and np.mean(vals) < np.mean([p["hnr_db"] for p in all_points if p["zone"] == "green"]) else ""
        print(f"  {z.upper():8s}  n={len(vals):2d}  mean={vals.mean():5.1f} dB  "
              f"min={vals.min():5.1f}  max={vals.max():5.1f}  std={vals.std():.1f}  {indicator}")

    # ── Correlation analysis ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FEATURE CORRELATIONS WITH ZONE (zone encoded green=0, yellow=1, red=2)")
    print("=" * 60)
    zone_num = {"green": 0, "yellow": 1, "red": 2}

    for label_set, pts in [("ALL DATA", all_points), ("PERSONAL ONLY", personal), ("YT ONLY", yt_pts)]:
        if not pts:
            continue
        print(f"\n  -- {label_set} (n={len(pts)}) --")
        z_vals = np.array([zone_num[p["zone"]] for p in pts])
        for feat, label in [
            ("strain",      "EARS strain score"),
            ("tonos",       "tonos (tension)   "),
            ("hnr_db",      "HNR dB (raw)      "),
            ("hnr_norm",    "HNR norm          "),
            ("shimmer_pct", "shimmer %         "),
        ]:
            f_vals = np.array([p[feat] for p in pts])
            corr   = float(np.corrcoef(f_vals, z_vals)[0, 1])
            direction = "↑ with strain (good)" if corr > 0.1 else "↓ with strain (inverted!)" if corr < -0.1 else "  ~flat (no signal)"
            print(f"  {label}  r={corr:+.3f}   {direction}")

    # ── Raw data table ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RAW SEGMENT DATA")
    print("=" * 60)
    print(f"  {'File':35s}  {'t':5s}  {'Zone':8s}  {'Strain':7s}  {'Tonos':6s}  {'HNR_db':7s}  {'Shim%':6s}")
    for p in sorted(all_points, key=lambda x: (x["file"], x["t"])):
        flag = " ⟵" if p["replays"] >= 2 else ""
        print(f"  {p['file']:35s}  {p['t']:5.1f}  {p['zone']:8s}  "
              f"{p['strain']:.3f}    {p['tonos']:.3f}   {p['hnr_db']:5.1f}    {p['shimmer_pct']:5.2f}{flag}")

    print()
    print(f"Total data points: {len(all_points)}")
    print()


if __name__ == "__main__":
    main()
