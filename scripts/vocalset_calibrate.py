#!/usr/bin/env python3
"""
VocalSet calibration — runs EARS + parselmouth on professional singing technique clips.

Techniques tested:
  GREEN:   straight, vibrato
  YELLOW:  belt
  RED:     vocal_fry
  BREATHY: breathy  (separate axis — not on strain meter)

Also runs Daniel's labeled anchor clips for cross-validation.

Usage:
  python scripts/vocalset_calibrate.py          # full run, 40 clips/technique
  python scripts/vocalset_calibrate.py --n 15   # quick run, 15 clips/technique
"""
import os, sys, json, math, io, random, argparse, zipfile, tempfile
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception", "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SAMPLE_RATE  = 44100
WINDOW_S     = 1.5          # center window to analyze per clip
SILENCE_RMS  = 0.006
ZIP_PATH     = os.path.join(PROJECT_ROOT, "data", "vocalset", "VocalSet.zip")
ANCHORS_DIR  = os.path.join(PROJECT_ROOT, "Vocal test recording sessions", "Anchors")

# Technique → expected zone label for accuracy reporting
TECHNIQUE_ZONE = {
    "straight":  "green",
    "vibrato":   "green",
    "belt":      "yellow",
    "vocal_fry": "red",
    "breathy":   "breathy",   # separate axis
}

ZONE_NUM = {"green": 0, "yellow": 1, "red": 2, "breathy": -1}

ANCHOR_LABELS = {
    "Easy 2.m4a":          "green",
    "Medium 1.m4a":        "green",
    "hard push.m4a":       "yellow",
    "Rough 1.m4a":         "red",
    # legacy names in case not renamed yet
    "Easy 3.m4a":          "green",
    "hard push 2.m4a":     "red",
}


def load_wav_from_zip(zf, path) -> np.ndarray:
    """Load a wav file from the zip into a float32 numpy array."""
    import soundfile as sf
    data = zf.read(path)
    audio, sr = sf.read(io.BytesIO(data))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def load_audio_file(path) -> np.ndarray:
    """Load any audio file (m4a/wav/etc) via ffmpeg → librosa."""
    import librosa, subprocess
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
                   capture_output=True, check=True)
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    return audio


def center_chunk(audio: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    """Take center N seconds of audio."""
    n = int(window_s * SAMPLE_RATE)
    if len(audio) <= n:
        return audio
    start = (len(audio) - n) // 2
    return audio[start:start + n]


def analyze_clip(audio: np.ndarray) -> dict:
    """Run EARS + parselmouth on a chunk. Returns feature dict."""
    from mel_extractor import MelExtractor
    from frequency_explorer import compute_emotion_properties

    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    chunk = center_chunk(audio)

    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < SILENCE_RMS:
        return None

    # EARS — tonos
    try:
        mel   = mel_ex.extract_from_audio(chunk)
        em    = compute_emotion_properties(mel)
        tonos = float(em.get("tension", 0.5))
        if math.isnan(tonos) or math.isinf(tonos):
            tonos = 0.5
        tonos = max(0.0, min(1.0, tonos))
    except Exception:
        tonos = 0.5

    # Parselmouth — HNR, shimmer
    hnr_db, shimmer_pct = 20.0, 0.0
    try:
        import parselmouth
        from parselmouth import praat
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

    # Spectral tilt: ratio of low-freq energy to total (breathy indicator)
    # Low freqs = <500Hz bins on mel spectrogram, normalized
    try:
        mel_lin = np.exp(mel_ex.extract_from_audio(chunk))
        n_mels  = mel_lin.shape[1] if mel_lin.ndim == 2 else len(mel_lin)
        low_cut = int(n_mels * 0.15)   # ~bottom 15% of mel bands ≈ <500Hz
        if mel_lin.ndim == 2:
            low_e  = float(mel_lin[:, :low_cut].mean())
            tot_e  = float(mel_lin.mean()) + 1e-8
        else:
            low_e  = float(mel_lin[:low_cut].mean())
            tot_e  = float(mel_lin.mean()) + 1e-8
        spectral_tilt = low_e / tot_e
    except Exception:
        spectral_tilt = 0.5

    # Current strain formula (session-adaptive with fixed defaults as baseline)
    hnr_norm     = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
    shimmer_norm = min(1.0, shimmer_pct / 10.0)
    strain_v1    = tonos * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10

    return {
        "tonos":          tonos,
        "hnr_db":         hnr_db,
        "shimmer_pct":    shimmer_pct,
        "hnr_norm":       hnr_norm,
        "shimmer_norm":   shimmer_norm,
        "spectral_tilt":  spectral_tilt,
        "strain_v1":      strain_v1,
        "rms":            rms,
    }


def sample_technique(zf, technique: str, n: int, wavs: list) -> list:
    """Return up to n randomly sampled analyzed clips for a technique."""
    candidates = [w for w in wavs if f"/{technique}/" in w]
    random.shuffle(candidates)
    results = []
    for path in candidates:
        if len(results) >= n:
            break
        try:
            audio = load_wav_from_zip(zf, path)
            feat  = analyze_clip(audio)
            if feat:
                feat["path"] = path
                results.append(feat)
        except Exception:
            pass
    return results


def print_stats(label: str, pts: list, features=("hnr_db","shimmer_pct","tonos","spectral_tilt","strain_v1")):
    if not pts:
        print(f"  {label}: no data")
        return
    print(f"\n  {label} (n={len(pts)}):")
    for f in features:
        vals = np.array([p[f] for p in pts])
        print(f"    {f:<18} mean={vals.mean():6.3f}  std={vals.std():5.3f}  "
              f"min={vals.min():6.3f}  max={vals.max():6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="clips per technique")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(ZIP_PATH):
        print(f"VocalSet zip not found: {ZIP_PATH}")
        sys.exit(1)

    print(f"Opening VocalSet ({args.n} clips/technique)…")
    zf   = zipfile.ZipFile(ZIP_PATH)
    wavs = [n for n in zf.namelist() if n.endswith(".wav")]

    techniques = ["straight", "vibrato", "belt", "vocal_fry", "breathy"]
    data = {}

    for tech in techniques:
        print(f"  Sampling {tech}…", flush=True)
        data[tech] = sample_technique(zf, tech, args.n, wavs)
        print(f"    {len(data[tech])} clips analyzed")

    zf.close()

    # ── Daniel's anchor clips ──────────────────────────────────────────────────
    print("\n  Loading anchor clips…")
    anchors = {}
    if os.path.exists(ANCHORS_DIR):
        for fname in os.listdir(ANCHORS_DIR):
            if fname in ANCHOR_LABELS:
                zone = ANCHOR_LABELS[fname]
                try:
                    audio = load_audio_file(os.path.join(ANCHORS_DIR, fname))
                    feat  = analyze_clip(audio)
                    if feat:
                        feat["label"] = zone
                        feat["file"]  = fname
                        anchors[fname] = feat
                        print(f"    {fname} ({zone}) → HNR={feat['hnr_db']:.1f} "
                              f"shim={feat['shimmer_pct']:.2f}% tonos={feat['tonos']:.3f}")
                except Exception as e:
                    print(f"    {fname}: ERROR {e}")

    # ── Per-technique stats ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FEATURE MEANS BY TECHNIQUE")
    print("(Expected: straight/vibrato=GREEN, belt=YELLOW, vocal_fry=RED, breathy=BREATHY)")
    print("=" * 70)

    for tech in techniques:
        print_stats(tech.upper(), data[tech])

    # ── Correlation analysis ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STRAIN SIGNAL CORRELATIONS")
    print("(Using GREEN=0, YELLOW=1, RED=2 encoding; breathy excluded from strain axis)")
    print("=" * 70)

    strain_pts = []
    for tech in ["straight", "vibrato", "belt", "vocal_fry"]:
        zone_n = ZONE_NUM[TECHNIQUE_ZONE[tech]]
        for p in data[tech]:
            strain_pts.append({**p, "zone_n": zone_n, "tech": tech})

    z_arr = np.array([p["zone_n"] for p in strain_pts])
    for feat in ("strain_v1", "hnr_db", "hnr_norm", "shimmer_pct", "tonos", "spectral_tilt"):
        f_arr = np.array([p[feat] for p in strain_pts])
        corr  = float(np.corrcoef(f_arr, z_arr)[0, 1])
        direction = "↑ with strain ✓" if corr > 0.1 else "↓ with strain (inverted!)" if corr < -0.1 else "~flat"
        print(f"  {feat:<20} r={corr:+.3f}   {direction}")

    # Breathy vs non-breathy
    print()
    print("BREATHY DETECTION SIGNALS")
    print("(1 = breathy, 0 = non-breathy — using straight as clean reference)")
    breathy_pts = [(p, 1) for p in data["breathy"]] + [(p, 0) for p in data["straight"]]
    b_arr = np.array([b for _, b in breathy_pts])
    pts_only = [p for p, _ in breathy_pts]
    for feat in ("hnr_db", "shimmer_pct", "tonos", "spectral_tilt", "hnr_norm"):
        f_arr = np.array([p[feat] for p in pts_only])
        corr  = float(np.corrcoef(f_arr, b_arr)[0, 1])
        direction = "↑ with breathy ✓" if corr > 0.1 else "↓ with breathy" if corr < -0.1 else "~flat"
        print(f"  {feat:<20} r={corr:+.3f}   {direction}")

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("THRESHOLD SWEEP — strain_v1 (find best G/Y and Y/R cutoffs)")
    print("=" * 70)
    g_pts = [p["strain_v1"] for p in data["straight"]] + [p["strain_v1"] for p in data["vibrato"]]
    y_pts = [p["strain_v1"] for p in data["belt"]]
    r_pts = [p["strain_v1"] for p in data["vocal_fry"]]

    g_arr = np.array(g_pts)
    y_arr = np.array(y_pts)
    r_arr = np.array(r_pts)

    print(f"\n  GREEN (straight+vibrato): mean={g_arr.mean():.3f} std={g_arr.std():.3f}  "
          f"75th={np.percentile(g_arr,75):.3f}")
    print(f"  YELLOW (belt):            mean={y_arr.mean():.3f} std={y_arr.std():.3f}  "
          f"25th={np.percentile(y_arr,25):.3f}  75th={np.percentile(y_arr,75):.3f}")
    print(f"  RED (vocal_fry):          mean={r_arr.mean():.3f} std={r_arr.std():.3f}  "
          f"25th={np.percentile(r_arr,25):.3f}")

    best_gy, best_yr, best_gy_acc, best_yr_acc = 0.5, 0.7, 0.0, 0.0
    for t in np.linspace(0.2, 0.9, 71):
        acc = (sum(1 for v in g_arr if v < t) + sum(1 for v in y_arr if v >= t)) / (len(g_arr) + len(y_arr))
        if acc > best_gy_acc:
            best_gy_acc, best_gy = acc, t
    for t in np.linspace(0.3, 1.0, 71):
        acc = (sum(1 for v in y_arr if v < t) + sum(1 for v in r_arr if v >= t)) / (len(y_arr) + len(r_arr))
        if acc > best_yr_acc:
            best_yr_acc, best_yr = acc, t

    print(f"\n  Best G/Y threshold: {best_gy:.3f}  (accuracy {best_gy_acc*100:.0f}%)")
    print(f"  Best Y/R threshold: {best_yr:.3f}  (accuracy {best_yr_acc*100:.0f}%)")

    # ── Formula v2 exploration ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FORMULA EXPLORATION — which feature combo gives best separation?")
    print("=" * 70)

    combos = [
        ("hnr_norm + shimmer_norm",       lambda p: p["hnr_norm"] * 0.6 + p["shimmer_norm"] * 0.4),
        ("tonos only",                    lambda p: p["tonos"]),
        ("strain_v1 (current)",           lambda p: p["strain_v1"]),
        ("hnr_norm*0.5 + shim*0.3 + t*0.2", lambda p: p["hnr_norm"]*0.5 + p["shimmer_norm"]*0.3 + p["tonos"]*0.2),
        ("shim*0.6 + hnr_norm*0.4",       lambda p: p["shimmer_norm"]*0.6 + p["hnr_norm"]*0.4),
        ("(1-hnr/40)*0.5 + shim/20*0.5",  lambda p: max(0,min(1,(1-p["hnr_db"]/40)))*0.5 + min(1,p["shimmer_pct"]/20)*0.5),
    ]

    for name, fn in combos:
        try:
            scores = np.array([fn(p) for p in strain_pts])
            corr   = float(np.corrcoef(scores, z_arr)[0, 1])
            g_mean = np.mean([fn(p) for p in data["straight"]] + [fn(p) for p in data["vibrato"]])
            y_mean = np.mean([fn(p) for p in data["belt"]])
            r_mean = np.mean([fn(p) for p in data["vocal_fry"]])
            sep    = r_mean - g_mean   # separation: bigger = better
            print(f"  {name:<42} r={corr:+.3f}  G={g_mean:.3f} Y={y_mean:.3f} R={r_mean:.3f}  sep={sep:+.3f}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # ── Anchor cross-check ─────────────────────────────────────────────────────
    if anchors:
        print()
        print("=" * 70)
        print("DANIEL'S ANCHOR CLIPS vs CURRENT FORMULA")
        print("=" * 70)
        for fname, feat in sorted(anchors.items(), key=lambda x: ZONE_NUM.get(x[1]["label"], 0)):
            zone = feat["label"]
            s    = feat["strain_v1"]
            pred = "green" if s < 0.50 else "yellow" if s < 0.68 else "red"
            match = "✅" if pred == zone else "❌"
            print(f"  {fname:<25} ({zone:6}) → strain={s:.3f} pred={pred} {match}")

    print()


if __name__ == "__main__":
    main()
