#!/usr/bin/env python3
"""
Scan ALL EARS dimensions across annotated segments.
For each 2s segment, takes both MEAN and MAX of each dim across all frames in window.
Ranks by correlation with annotation zone (green=0, yellow=1, red=2).
"""
import os, sys, json, math, tempfile, subprocess
import numpy as np

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "docs", "annotations")
RECORDINGS_DIR  = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception", "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SAMPLE_RATE = 44100
WINDOW_S    = 1.0
HOP_S       = 0.25
SILENCE_RMS = 0.008


def load_audio(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
                   capture_output=True, check=True)
    import librosa
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    return audio


def run_full_ears(audio):
    """Run full analyze_mel on each hop → list of {t, dims: {name: value}}."""
    from mel_extractor import MelExtractor
    from frequency_explorer import analyze_mel, compute_emotion_properties, compute_tactile_properties

    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    ws     = int(WINDOW_S * SAMPLE_RATE)
    hs     = int(HOP_S * SAMPLE_RATE)
    frames = []
    i = 0
    while i + ws <= len(audio):
        chunk = audio[i:i+ws]
        rms   = float(np.sqrt(np.mean(chunk**2)))
        t     = i / SAMPLE_RATE
        if rms < SILENCE_RMS:
            i += hs
            continue
        try:
            mel  = mel_ex.extract_from_audio(chunk)
            # Full analysis — all dimensions
            res  = analyze_mel(mel)
            dims = {}
            for mod_name, mod_dict in res.get("modalities", {}).items():
                if isinstance(mod_dict, dict):
                    for k, v in mod_dict.items():
                        if isinstance(v, (int, float)):
                            f = float(v)
                            if not (math.isnan(f) or math.isinf(f)):
                                dims[f"{mod_name}.{k}"] = f
            # Also add fast-path dims explicitly
            em = compute_emotion_properties(mel)
            tc = compute_tactile_properties(mel)
            for k, v in em.items():
                if isinstance(v, (int, float)):
                    f = float(v)
                    if not (math.isnan(f) or math.isinf(f)):
                        dims[f"emotion.{k}"] = f
            for k, v in tc.items():
                if isinstance(v, (int, float)):
                    f = float(v)
                    if not (math.isnan(f) or math.isinf(f)):
                        dims[f"tactile.{k}"] = f
            frames.append({"t": t, "dims": dims})
        except Exception as e:
            pass
        i += hs
    return frames


def dims_in_window(frames, t_start, t_end):
    """Return all frames whose center falls within [t_start, t_end]."""
    return [f for f in frames if t_start <= f["t"] <= t_end]


def main():
    ann_files = sorted(f for f in os.listdir(ANNOTATIONS_DIR) if "_reviewed" in f)
    if not ann_files:
        print("No *_reviewed.json files found.")
        sys.exit(1)

    # Collect: zone + dict of all dim values per segment
    records = []   # {zone, dims_mean: {}, dims_max: {}}

    for ann_file in ann_files:
        ann   = json.load(open(os.path.join(ANNOTATIONS_DIR, ann_file)))
        rated = [s for s in ann.get("segment_ratings", []) if s["zone"]]
        if not rated:
            continue

        audio_path = os.path.join(RECORDINGS_DIR, ann["file"])
        if not os.path.exists(audio_path):
            print(f"  Audio not found: {ann['file']} — skipping")
            continue

        print(f"Running full EARS on {ann['file']} ({len(rated)} segments)...")
        audio  = load_audio(audio_path)
        frames = run_full_ears(audio)
        print(f"  {len(frames)} voiced frames extracted")

        for seg in rated:
            window_frames = dims_in_window(frames, seg["t_start"], seg["t_end"])
            if not window_frames:
                # Expand search to ±1s if window was silent
                window_frames = dims_in_window(frames, seg["t_start"] - 1.0, seg["t_end"] + 1.0)
            if not window_frames:
                continue

            # Collect all dim values across frames in this window
            all_dim_names = set()
            for f in window_frames:
                all_dim_names.update(f["dims"].keys())

            dims_mean, dims_max = {}, {}
            for dim in all_dim_names:
                vals = [f["dims"][dim] for f in window_frames if dim in f["dims"]]
                if vals:
                    dims_mean[dim] = float(np.mean(vals))
                    dims_max[dim]  = float(np.max(vals))

            records.append({
                "zone":      seg["zone"],
                "dims_mean": dims_mean,
                "dims_max":  dims_max,
                "file":      ann["file"],
                "t_start":   seg["t_start"],
                "n_frames":  len(window_frames),
            })

    if not records:
        print("No data.")
        sys.exit(1)

    print(f"\nTotal annotated segments with data: {len(records)}")
    zone_num = {"green": 0, "yellow": 1, "red": 2}
    z_vals   = np.array([zone_num[r["zone"]] for r in records])

    # Collect all dim names
    all_dims = set()
    for r in records:
        all_dims.update(r["dims_mean"].keys())

    # Compute correlation for mean and max of each dim
    results = []
    for dim in sorted(all_dims):
        mean_vals = np.array([r["dims_mean"].get(dim, np.nan) for r in records])
        max_vals  = np.array([r["dims_max"].get(dim, np.nan)  for r in records])

        # Skip dims with too many missing values
        mean_ok = ~np.isnan(mean_vals)
        max_ok  = ~np.isnan(max_vals)
        if mean_ok.sum() < 10:
            continue

        try:
            r_mean = float(np.corrcoef(mean_vals[mean_ok], z_vals[mean_ok])[0, 1])
        except Exception:
            r_mean = 0.0
        try:
            r_max = float(np.corrcoef(max_vals[max_ok], z_vals[max_ok])[0, 1])
        except Exception:
            r_max = 0.0

        if math.isnan(r_mean):
            r_mean = 0.0
        if math.isnan(r_max):
            r_max = 0.0

        best_r = r_mean if abs(r_mean) >= abs(r_max) else r_max
        results.append((dim, r_mean, r_max, best_r))

    # Sort by absolute best correlation
    results.sort(key=lambda x: abs(x[3]), reverse=True)

    print()
    print("=" * 72)
    print("ALL EARS DIMENSIONS RANKED BY CORRELATION WITH STRAIN ZONE")
    print("(zone: green=0, yellow=1, red=2 — positive r = increases with strain)")
    print("=" * 72)
    print(f"  {'Dimension':40s}  {'r_mean':7s}  {'r_max':7s}  Direction")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*25}")

    for dim, r_mean, r_max, best_r in results:
        if abs(best_r) < 0.05:
            continue  # skip flat dims
        arrow = "↑ increases with strain" if best_r > 0 else "↓ decreases with strain"
        flag  = " ★" if abs(best_r) >= 0.30 else ""
        print(f"  {dim:40s}  {r_mean:+.3f}    {r_max:+.3f}    {arrow}{flag}")

    # Summary: top candidates
    print()
    print("=" * 72)
    print("TOP CANDIDATES (|r| ≥ 0.30)")
    print("=" * 72)
    top = [(d, rm, rx, br) for d, rm, rx, br in results if abs(br) >= 0.30]
    if top:
        for dim, r_mean, r_max, best_r in top:
            zone_means = {}
            for z in ["green", "yellow", "red"]:
                vals = [r["dims_mean"].get(dim) for r in records
                        if r["zone"] == z and r["dims_mean"].get(dim) is not None]
                zone_means[z] = round(float(np.mean(vals)), 4) if vals else None
            print(f"\n  {dim}  (r_mean={r_mean:+.3f}, r_max={r_max:+.3f})")
            print(f"    green={zone_means['green']}  yellow={zone_means['yellow']}  red={zone_means['red']}")
    else:
        print("  None above 0.30 threshold.")

    print()


if __name__ == "__main__":
    main()
