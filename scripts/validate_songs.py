#!/usr/bin/env python3
"""
Song-only validation — the only ground truth that matters.

Tests the full v8+scatter+phonation_classifier pipeline on singing recordings.
No anchor clips. No CVT studio recordings.

Ground truth (Daniel's subjective expectations):
  Runnin' Down a Dream  → should be mostly GREEN (comfortable country rock)
  Chris Young           → mostly GREEN, couple YELLOW phrases (more pushed)
  Liza Jane             → verses GREEN, chorus YELLOW/RED (intentionally pushed chorus)

The Liza Jane segment test is the clearest signal:
  0-20s  = verse  → should be GREEN  (Daniel went easy)
  20-40s = chorus → should be YELLOW/RED (Daniel pushed intentionally)
  repeat...

Usage:
    .venv/bin/python3 scripts/validate_songs.py
    .venv/bin/python3 scripts/validate_songs.py --segments   # Liza Jane 10s segments
    .venv/bin/python3 scripts/validate_songs.py --phonation  # show per-frame classifier output
"""

import argparse
import math
import json
import math
import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mel_extractor import MelExtractor
    from frequency_explorer import analyze_mel
    _mel_extractor = MelExtractor(sample_rate=44100)
    EARS_OK = True
except Exception as _e:
    EARS_OK = False
    print(f"[WARN] EARS not available: {_e}")

SR          = 44100
CHUNK       = 4410    # 100ms — same as backend
SILENCE_RMS   = 0.008
LOW_ENERGY_RMS = 0.016   # 2x silence floor — phrase tails produce garbage shimmer/CPP
SCATTER_CHUNK = 4096  # scatter expects this size

STRAIN_GREEN  = 0.40
STRAIN_YELLOW = 0.60

SONGS = {
    "runnin": {
        "path": "Vocal test recording sessions/Danny - Runnin down a dream R1.m4a",
        "expect": "GREEN",
        "note": "comfortable country rock, should be just green",
    },
    "chrisyoung": {
        "path": "Vocal test recording sessions/Danny - Chris Young R1.m4a",
        "expect": "YELLOW",
        "note": "mostly green, couple yellow — more pushed",
    },
    "lizajane": {
        "path": "Vocal test recording sessions/Danny - Liza Jane R1 (longer).m4a",
        "expect": "RED",
        "note": "verses green, chorus pushed — should see structure",
    },
}


# ─── Feature extraction (mirrors backend exactly) ─────────────────────────────

try:
    from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
    _scatter = ScatteringNumPy1D(J=7, shape=SCATTER_CHUNK, Q=1)
    SCATTER_OK = True
except Exception as e:
    SCATTER_OK = False
    print(f"[WARN] scatter unavailable: {e}")

try:
    import joblib as _jl
    _MODEL_PATH = Path(__file__).parent.parent / "models" / "phonation_classifier.joblib"
    if _MODEL_PATH.exists():
        _bundle = _jl.load(_MODEL_PATH)
        _model = _bundle["model"]
        _int_to_label = _bundle["int_to_label"]
        _strain_map_model = _bundle["strain_map"]
        CLASSIFIER_OK = True
        _model_ver = _bundle.get("version", 1)
        _model_acc = _bundle.get("binary_accuracy", _bundle.get("accuracy", 0))
        print(f"[INFO] Classifier v{_model_ver} loaded  binary_acc={_model_acc:.3f}")
    else:
        CLASSIFIER_OK = False
        _model = None
        print(f"[WARN] No classifier at {_MODEL_PATH}")
except Exception as e:
    CLASSIFIER_OK = False
    _model = None
    print(f"[WARN] Classifier load failed: {e}")


def scatter_features(chunk: np.ndarray):
    if not SCATTER_OK:
        return None
    x = chunk.astype(np.float64)
    if len(x) < SCATTER_CHUNK:
        x = np.pad(x, (0, SCATTER_CHUNK - len(x)))
    else:
        x = x[:SCATTER_CHUNK]
    rms = float(np.sqrt(np.mean(x**2)))
    if rms < 1e-8:
        return None
    x /= rms
    Sx = _scatter(x)
    feat = np.mean(Sx, axis=1)
    return np.log(np.abs(feat) + 1e-10)


def am_fast_power(chunk, sr=SR):
    """10-30Hz AM power in 75-300Hz band (Eulerian audio motion amplification)."""
    try:
        from scipy.signal import butter, sosfilt
        from scipy.signal import hilbert as _hilbert
        sos = butter(4, [75, 300], btype='bandpass', fs=sr, output='sos')
        env = np.abs(_hilbert(sosfilt(sos, chunk.astype(np.float64))))
        env_rms = float(np.sqrt(np.mean(env**2)))
        if env_rms < 1e-8:
            return 0.0
        env_n = env / env_rms
        fft = np.fft.rfft(env_n)
        freqs = np.fft.rfftfreq(len(env_n), d=1.0/sr)
        mask = (freqs >= 10.0) & (freqs < 30.0)
        return float(np.sum(np.abs(fft[mask])**2)) / (len(fft) + 1e-9)
    except Exception:
        return 0.0


def compute_cpp(chunk):
    try:
        N = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        win = np.hanning(N)
        spec = np.fft.rfft(pre * win, n=N)
        log_pow = np.log(np.abs(spec)**2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_pow))[:N//2]
        q_min, q_max = int(SR/600), int(SR/75)
        if q_max >= len(cepstrum):
            return float('nan')
        peak_idx = q_min + int(np.argmax(cepstrum[q_min:q_max+1]))
        q_axis = np.arange(len(cepstrum)) / float(SR)
        coeffs = np.polyfit(q_axis[q_min:q_max+1], cepstrum[q_min:q_max+1], 1)
        return float(cepstrum[peak_idx] - np.polyval(coeffs, q_axis[peak_idx]))
    except Exception:
        return float('nan')


# ─── Per-frame analysis (mirrors backend signal pipeline) ─────────────────────

def analyze_frames(y: np.ndarray, show_phonation: bool = False):
    """Yield per-frame analysis dicts at 10Hz."""
    n_frames = len(y) // CHUNK
    scatter_baseline_feats = []
    scatter_mean = None
    scatter_std  = None
    SCATTER_WARM_N = 15

    shim_baseline = 5.26
    cpp_baseline  = 0.22
    baseline_clean_n = 0
    voiced_run = 0

    # EARS v11 baselines
    elast_baseline = 0.30
    anham_baseline = 0.15
    am_baseline    = 0.0

    for i in range(n_frames):
        t = i * 0.1
        chunk = y[i*CHUNK:(i+1)*CHUNK]
        rms = float(np.sqrt(np.mean(chunk**2)))

        if rms < SILENCE_RMS:
            voiced_run = 0
            yield {"t": t, "active": False}
            continue

        voiced_run += 1
        onset_gated = voiced_run < 3
        low_energy  = rms < LOW_ENERGY_RMS   # phrase tail: suppress acoustic features

        # Parselmouth shimmer + HNR — suppress at onset AND low-energy (phrase tails)
        shimmer_pct = float('nan')
        hnr_db = 0.0
        if not onset_gated and not low_energy:
            try:
                snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SR))
                harm = snd.to_harmonicity()
                vals = harm.values[0]
                valid = vals[vals > -200]
                hnr_db = float(np.mean(valid)) if len(valid) > 0 else 0.0
                pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
                shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_pct = (shim or 0.0) * 100.0
            except Exception:
                pass

        cpp = compute_cpp(chunk) if not onset_gated and not low_energy else float('nan')

        # Baseline adaptation
        # Fast bootstrap: first 20 voiced frames always update baseline (simulates real-session warm-up
        # where singer hums before singing). After frame 20, switch to strict gating (strain gate).
        BOOTSTRAP_N = 20
        shim_g = shimmer_pct if not math.isnan(shimmer_pct) else shim_baseline
        cpp_g  = cpp         if not math.isnan(cpp)         else cpp_baseline

        tent_shim = max(0.0, shim_g - shim_baseline) / 10.0
        tent_cpp  = max(0.0, cpp_baseline - cpp_g) / 0.5
        in_bootstrap = baseline_clean_n < BOOTSTRAP_N
        is_clean  = not onset_gated and not low_energy and (in_bootstrap or max(tent_shim, tent_cpp) < 0.25)

        # EARS v11 features — disabled in validate_songs (backend uses 1s ring buffer;
        # 100ms chunks here give unreliable temporal features like elastikos/anharmonia).
        # The backend includes these signals correctly via all_dims from full 1s window.
        ears_v11 = 0.0

        if is_clean:
            a = 0.15 if in_bootstrap else 0.05   # faster convergence during bootstrap
            # Shimmer: asymmetric — fast UP (shimmer increase = new norm), slow DOWN
            # (prevents drift when soft phrases over-tune sensitivity)
            if shim_g >= shim_baseline:
                shim_baseline = (1 - a) * shim_baseline + a * shim_g
            else:
                shim_baseline = (1 - 0.01) * shim_baseline + 0.01 * shim_g
            cpp_baseline  = (1 - a) * cpp_baseline  + a * cpp_g
            baseline_clean_n += 1

        if onset_gated or low_energy:
            shim_dev, cpp_dev = 0.0, 0.0
        else:
            shim_dev = max(0.0, shim_g - shim_baseline) / 10.0
            cpp_dev  = max(0.0, cpp_baseline - cpp_g) / 0.5

        v8_strain = min(1.0, max(shim_dev, cpp_dev))

        # Scatter
        scatter_score = 0.0
        sf = scatter_features(chunk)
        if sf is not None:
            if scatter_mean is not None:
                z = (sf - scatter_mean) / scatter_std
                raw = float(np.mean(np.abs(z)))
                scatter_score = min(1.0, max(0.0, (raw - 1.0) / 2.0))
                if scatter_score < 0.3:
                    scatter_mean = (1 - 0.05) * scatter_mean + 0.05 * sf
            elif is_clean:
                scatter_baseline_feats.append(sf)
                if len(scatter_baseline_feats) >= SCATTER_WARM_N:
                    arr = np.stack(scatter_baseline_feats)
                    scatter_mean = np.mean(arr, axis=0)
                    scatter_std  = np.std(arr, axis=0) + 1e-8

        # Phonation classifier
        phonation_score = 0.0
        if CLASSIFIER_OK and sf is not None:
            try:
                # v2 model needs 65-dim features; v1 needs 34-dim scatter only
                model_ver = _bundle.get("version", 1)
                if model_ver == 1:
                    feat_in = sf
                else:
                    # Build 65-dim for v2
                    import librosa as _lr
                    x32 = chunk.astype(np.float32)
                    if len(x32) < SCATTER_CHUNK:
                        x32 = np.pad(x32, (0, SCATTER_CHUNK - len(x32)))
                    mfcc = _lr.feature.mfcc(y=x32, sr=SR, n_mfcc=13, n_fft=512, hop_length=128)
                    delta = _lr.feature.delta(mfcc)
                    mfcc_v = np.concatenate([np.mean(mfcc, axis=1), np.mean(delta, axis=1)])
                    # H1*-H2*
                    h1h2 = np.array([0.0])
                    try:
                        snd2 = parselmouth.Sound(chunk.astype(np.float64)[:SCATTER_CHUNK], sampling_frequency=float(SR))
                        pitch = snd2.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
                        f0_vals = pitch.selected_array['frequency']
                        f0_vals = f0_vals[f0_vals > 0]
                        if len(f0_vals) > 0:
                            f0 = float(np.median(f0_vals))
                            spec = snd2.to_spectrum()
                            freqs = np.array(spec.xs())
                            amps_db = 20 * np.log10(np.abs(spec.values[0]) + 1e-12)
                            def amp_at(freq):
                                idx = np.argmin(np.abs(freqs - freq))
                                lo, hi = max(0, idx-5), min(len(amps_db), idx+6)
                                return float(np.max(amps_db[lo:hi]))
                            h1h2 = np.array([amp_at(f0) - amp_at(2*f0)])
                    except Exception:
                        pass
                    # Spectral
                    centroid  = float(np.mean(_lr.feature.spectral_centroid(y=x32, sr=SR, n_fft=512, hop_length=128)))
                    bandwidth = float(np.mean(_lr.feature.spectral_bandwidth(y=x32, sr=SR, n_fft=512, hop_length=128)))
                    rolloff   = float(np.mean(_lr.feature.spectral_rolloff(y=x32, sr=SR, n_fft=512, hop_length=128)))
                    zcr       = float(np.mean(_lr.feature.zero_crossing_rate(y=x32, hop_length=128)))
                    spec_v = np.array([centroid/SR, bandwidth/SR, rolloff/SR, zcr])
                    feat_in = np.concatenate([sf, mfcc_v, h1h2, spec_v])

                proba = _model.predict_proba(feat_in.reshape(1, -1))[0]
                phonation_score = float(proba[0] + proba[1])
            except Exception:
                pass

        # Fusion — v8 + scatter + phonation (scaled) + EARS v11
        # Phonation scaled by 0.6: domain-shifted (CVT close-mic → iPhone), supporting role only
        ph_scaled = phonation_score * 0.6
        if scatter_mean is not None and sf is not None:
            w = 0.5
            max_s  = max(v8_strain, scatter_score, ph_scaled)
            wavg_s = (v8_strain + scatter_score + ph_scaled) / 3.0
            strain = min(1.0, (1 - w) * max_s + w * wavg_s)
        else:
            strain = min(1.0, max(v8_strain, ph_scaled))

        zone = "green" if strain < STRAIN_GREEN else "yellow" if strain < STRAIN_YELLOW else "red"

        yield {
            "t": t, "active": True,
            "strain": strain, "zone": zone,
            "v8": v8_strain, "scatter": scatter_score,
            "phonation": phonation_score, "ears_v11": ears_v11,
            "shimmer": shimmer_pct, "cpp": cpp,
            "shim_baseline": shim_baseline, "cpp_baseline": cpp_baseline,
        }


def p80(scores):
    if not scores:
        return 0.0
    return float(np.percentile(scores, 80))


def zone_from_p80(s):
    return "GREEN" if s < STRAIN_GREEN else "YELLOW" if s < STRAIN_YELLOW else "RED"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments",     action="store_true", help="10s segment breakdown")
    ap.add_argument("--phonation",    action="store_true", help="Show per-frame phonation score")
    ap.add_argument("--song",         choices=list(SONGS.keys()), help="Only run one song")
    ap.add_argument("--ground-truth", action="store_true",
                    help="Compare model output against your manual labels (data/ground_truth/)")
    args = ap.parse_args()

    songs_to_run = {args.song: SONGS[args.song]} if args.song else SONGS

    print(f"\n{'═'*65}")
    print(f"  SONG VALIDATION — v8 + scatter + phonation classifier")
    if CLASSIFIER_OK:
        _ver = _bundle.get("version", 1)
        _bacc = _bundle.get("binary_accuracy", _bundle.get("accuracy", 0))
        print(f"  Classifier: v{_ver}  binary_acc={_bacc:.3f}")
    else:
        print("  Classifier: NOT LOADED")
    print(f"{'═'*65}")

    for name, info in songs_to_run.items():
        path = Path(info["path"])
        if not path.exists():
            print(f"\n[SKIP] {path.name} not found")
            continue

        print(f"\n{'─'*65}")
        print(f"  {path.stem}")
        print(f"  Expected: {info['expect']}  ({info['note']})")
        print(f"{'─'*65}")

        y, _ = librosa.load(str(path), sr=SR, mono=True)
        duration = len(y) / SR
        print(f"  Duration: {duration:.1f}s")

        frames = list(analyze_frames(y, show_phonation=args.phonation))
        active = [f for f in frames if f.get("active")]

        if not active:
            print("  No active frames detected.")
            continue

        strain_vals = [f["strain"] for f in active]
        v8_vals     = [f["v8"]     for f in active]
        sc_vals     = [f["scatter"] for f in active]
        ph_vals     = [f["phonation"] for f in active]

        p = p80(strain_vals)
        zone = zone_from_p80(p)
        match = "✓" if zone == info["expect"] else "✗"

        print(f"\n  P80 strain:    {p:.3f}  →  {zone} {match}")
        print(f"  P80 v8:        {p80(v8_vals):.3f}")
        print(f"  P80 scatter:   {p80(sc_vals):.3f}")
        print(f"  P80 phonation: {p80(ph_vals):.3f}")
        print(f"  Active frames: {len(active)}/{len(frames)}")

        if args.phonation:
            print(f"\n  Per-frame phonation scores (>0.2 highlighted):")
            for f in active:
                if f["phonation"] > 0.2:
                    print(f"    t={f['t']:5.1f}s  strain={f['strain']:.3f}  phonation={f['phonation']:.3f}  {f['zone']}")

        # ── Ground-truth comparison ──────────────────────────────────────────
        if args.ground_truth:
            labels_path = Path("data/ground_truth") / f"{name}_labels.json"
            if not labels_path.exists():
                print(f"\n  [NO LABELS] Run first: .venv/bin/python3 scripts/label_song.py --song {name}")
            else:
                gt = json.loads(labels_path.read_text())
                chunks_gt = gt.get("chunks", {})
                chunk_s = 2.0
                chunk_samples_gt = int(SR * chunk_s)
                n_chunks = int(len(y) / SR / chunk_s)

                ZONE_ORDER = {"green": 0, "yellow": 1, "red": 2}
                matches = exact = over = under = 0
                total = 0

                print(f"\n  GROUND TRUTH COMPARISON  (2s chunks, skips excluded)")
                print(f"  {'Time':>8}  {'Your label':>12}  {'Model':>8}  {'Match':>6}")
                print(f"  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*6}")

                for ci in range(n_chunks):
                    gt_label = chunks_gt.get(str(ci))
                    if gt_label is None or gt_label == "skip":
                        continue
                    t_start = ci * chunk_s
                    t_end   = (ci + 1) * chunk_s
                    seg_frames = [f for f in frames
                                  if f.get("active") and t_start <= f["t"] < t_end]
                    if not seg_frames:
                        continue
                    model_p80 = p80([f["strain"] for f in seg_frames])
                    model_zone = zone_from_p80(model_p80).lower()

                    gt_ord    = ZONE_ORDER.get(gt_label, 1)
                    model_ord = ZONE_ORDER.get(model_zone, 1)
                    diff = model_ord - gt_ord

                    if diff == 0:
                        match_sym = "✓"
                        matches += 1
                    elif diff > 0:
                        match_sym = f"↑{diff}"   # model more severe than you
                        over += 1
                    else:
                        match_sym = f"↓{abs(diff)}"  # model less severe than you (missed strain)
                        under += 1
                    total += 1

                    gt_c = {"green":"\033[92m","yellow":"\033[93m","red":"\033[91m"}.get(gt_label,"")
                    m_c  = {"green":"\033[92m","yellow":"\033[93m","red":"\033[91m"}.get(model_zone,"")
                    rst  = "\033[0m"
                    print(f"  {t_start:>4.0f}-{t_end:<3.0f}s  "
                          f"{gt_c}{gt_label:>12}{rst}  "
                          f"{m_c}{model_zone:>8}{rst}  "
                          f"{match_sym:>6}")

                print(f"\n  Total chunks: {total}")
                if total > 0:
                    print(f"  Exact match:  {matches}/{total} = {matches/total*100:.0f}%")
                    print(f"  Over-detect:  {over}/{total}  (model more severe than you — false alarms)")
                    print(f"  Under-detect: {under}/{total}  (model less severe than you — missed strain)")

    # ── Segment breakdown ──────────────────────────────────────────────────────
    if args.segments:
        info = SONGS["lizajane"]
        path = Path(info["path"])
        if not path.exists():
            print(f"\n[SKIP] Liza Jane not found for segment analysis")
            return

        print(f"\n{'═'*65}")
        print(f"  LIZA JANE — 10s SEGMENT BREAKDOWN")
        print(f"  Ground truth: verses=GREEN, chorus=YELLOW/RED")
        print(f"{'═'*65}")

        y, _ = librosa.load(str(path), sr=SR, mono=True)
        frames = list(analyze_frames(y))

        seg_len = 10.0  # seconds
        n_segs = int(len(y) / SR / seg_len)

        print(f"\n  {'Seg':>4}  {'Time':>8}  {'P80':>6}  {'Zone':>7}  {'v8':>6}  {'scatter':>8}  {'phonation':>10}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*10}")

        for s in range(n_segs):
            t_start = s * seg_len
            t_end   = (s + 1) * seg_len
            seg_frames = [f for f in frames
                          if f.get("active") and t_start <= f["t"] < t_end]
            if not seg_frames:
                continue
            p = p80([f["strain"] for f in seg_frames])
            z = zone_from_p80(p)
            pv8 = p80([f["v8"]       for f in seg_frames])
            psc = p80([f["scatter"]  for f in seg_frames])
            pph = p80([f["phonation"] for f in seg_frames])
            color = "\033[92m" if z=="GREEN" else "\033[93m" if z=="YELLOW" else "\033[91m"
            reset = "\033[0m"
            print(f"  {s:>4}  {t_start:>4.0f}-{t_end:<3.0f}s  {p:>6.3f}  {color}{z:>7}{reset}  {pv8:>6.3f}  {psc:>8.3f}  {pph:>10.3f}")

        print()


if __name__ == "__main__":
    main()
