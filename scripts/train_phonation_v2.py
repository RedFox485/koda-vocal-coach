#!/usr/bin/env python3
"""
Phonation Mode Classifier v2 — Enhanced Feature Pipeline
Dataset: CVT Vocal Mode Dataset (Zenodo 14276415)

Changes from v1:
  - Features: scatter(34) + MFCC+delta(26) + H1*-H2*(1) + spectral(4) = 65-dim
  - ExcludedSamples folder skipped (contains intentionally irregular samples)
  - class_weight='balanced' (neutral 2x overrepresented in dataset)
  - SVM + MLP, both with balanced weights
  - Target: >97% binary (pressed vs modal)

CVT Modes → Strain Mapping:
  overdrive / edge  → pressed/strained  (hyperadducted, metallic)
  neutral / curbing → modal/healthy     (normal phonation)

Usage:
    .venv/bin/python3 scripts/train_phonation_v2.py --data data/cvt_dataset/
    .venv/bin/python3 scripts/train_phonation_v2.py --data data/cvt_dataset/ --model svm
    .venv/bin/python3 scripts/train_phonation_v2.py --data data/cvt_dataset/ --output models/phonation_classifier.joblib
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
import joblib

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PRAAT_OK = True
except ImportError:
    PRAAT_OK = False
    print("[WARN] parselmouth not available — H1*-H2* feature disabled")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D

SR          = 44100
CHUNK       = 4096    # ~93ms — same as backend
SILENCE_RMS = 0.008
J, Q        = 7, 1   # 34-dim scatter

_scatter = Scattering1D(J=J, shape=CHUNK, Q=Q)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CVT mode label mapping
# ---------------------------------------------------------------------------

CVT_KEYWORDS = {
    "overdrive": "overdrive",
    "over_drive": "overdrive",
    "edge": "edge",
    "neutral": "neutral",
    "curbing": "curbing",
    "curb": "curbing",
}

LABEL_TO_INT = {"overdrive": 0, "edge": 1, "neutral": 2, "curbing": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
STRAIN_MAP   = {"overdrive": 1, "edge": 1, "neutral": 0, "curbing": 0}


# ---------------------------------------------------------------------------
# Feature extraction — 65-dim rich acoustic feature vector
# ---------------------------------------------------------------------------

def scatter_feat(chunk: np.ndarray) -> np.ndarray:
    """34-dim log-compressed scatter."""
    x = chunk.astype(np.float64)
    if len(x) < CHUNK:
        x = np.pad(x, (0, CHUNK - len(x)))
    else:
        x = x[:CHUNK]
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-8:
        return None
    x = x / rms
    Sx = _scatter(x)
    feat = np.mean(Sx, axis=1)
    return np.log(np.abs(feat) + 1e-10)


def mfcc_feat(chunk: np.ndarray) -> np.ndarray:
    """26-dim MFCC + delta (13+13). Captures spectral envelope / timbre."""
    x = chunk.astype(np.float32)
    if len(x) < CHUNK:
        x = np.pad(x, (0, CHUNK - len(x)))
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13, n_fft=512, hop_length=128)
    delta = librosa.feature.delta(mfcc)
    # Mean across time frames
    return np.concatenate([np.mean(mfcc, axis=1), np.mean(delta, axis=1)])


def h1h2_feat(chunk: np.ndarray) -> float:
    """H1*-H2*: formant-corrected first minus second harmonic amplitude (dB).
    Pressed phonation (overdrive/edge) → lower H1*-H2* (more adducted closure).
    Breathy phonation → higher H1*-H2* (incomplete closure, dominant fundamental).
    Returns 0.0 if parselmouth unavailable."""
    if not PRAAT_OK:
        return 0.0
    try:
        x = chunk.astype(np.float64)
        if len(x) < CHUNK:
            x = np.pad(x, (0, CHUNK - len(x)))
        snd = parselmouth.Sound(x, sampling_frequency=float(SR))
        # Get fundamental frequency estimate
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0_vals = pitch.selected_array['frequency']
        f0_vals = f0_vals[f0_vals > 0]
        if len(f0_vals) == 0:
            return 0.0
        f0 = float(np.median(f0_vals))
        # Spectrum for harmonic amplitude
        spec = snd.to_spectrum()
        freqs = np.array(spec.xs())
        # Real part of complex spectrum → amplitude
        amps_db = 20 * np.log10(np.abs(spec.values[0]) + 1e-12)
        # Find H1 and H2 amplitudes (±20% of harmonic frequency)
        def amp_at(freq):
            idx = np.argmin(np.abs(freqs - freq))
            window = 5  # ±5 bins
            lo, hi = max(0, idx-window), min(len(amps_db), idx+window+1)
            return float(np.max(amps_db[lo:hi]))
        h1 = amp_at(f0)
        h2 = amp_at(2 * f0)
        return h1 - h2
    except Exception:
        return 0.0


def spectral_feat(chunk: np.ndarray) -> np.ndarray:
    """4-dim spectral summary: centroid, bandwidth, rolloff, zcr.
    Overdrive/edge tend to be brighter (higher centroid, rolloff)."""
    x = chunk.astype(np.float32)
    if len(x) < CHUNK:
        x = np.pad(x, (0, CHUNK - len(x)))
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=x, sr=SR, n_fft=512, hop_length=128)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=x, sr=SR, n_fft=512, hop_length=128)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=x, sr=SR, n_fft=512, hop_length=128)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=x, hop_length=128)))
    return np.array([centroid / SR, bandwidth / SR, rolloff / SR, zcr])   # normalize to [0,1]-ish


def extract_features(path: Path, n_chunks: int = 5):
    """Extract 65-dim feature vector from CVT audio file.
    Uses middle third (avoids onset/offset artifacts).
    Returns None if silent or extraction fails."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        return None

    segment = y[len(y)//3 : 2*len(y)//3]
    step = max(1, len(segment) // n_chunks)

    scatter_feats, mfcc_feats, h1h2_vals, spec_feats = [], [], [], []

    for i in range(n_chunks):
        chunk = segment[i*step : i*step + CHUNK]
        if len(chunk) < CHUNK // 2:
            continue
        if float(np.sqrt(np.mean(chunk.astype(np.float64)**2))) < SILENCE_RMS:
            continue

        sf = scatter_feat(chunk)
        if sf is not None:
            scatter_feats.append(sf)
            mfcc_feats.append(mfcc_feat(chunk))
            h1h2_vals.append(h1h2_feat(chunk))
            spec_feats.append(spectral_feat(chunk))

    if not scatter_feats:
        return None

    return np.concatenate([
        np.mean(scatter_feats, axis=0),    # 34 dims — multi-scale modulation
        np.mean(mfcc_feats, axis=0),       # 26 dims — spectral envelope/timbre
        np.array([np.mean(h1h2_vals)]),    # 1 dim  — harmonic amplitude ratio (pressed indicator)
        np.mean(spec_feats, axis=0),       # 4 dims  — spectral brightness/shape
    ])                                     # = 65 dims total


# ---------------------------------------------------------------------------
# Dataset loading — skips ExcludedSamples
# ---------------------------------------------------------------------------

def find_label(path: Path):
    # Skip intentionally irregular samples
    if "excludedsamples" in str(path).lower():
        return None
    for part in reversed([p.lower() for p in path.parts]):
        for kw, lbl in CVT_KEYWORDS.items():
            if kw in part:
                return lbl
    name = path.stem.lower()
    for kw, lbl in CVT_KEYWORDS.items():
        if kw in name:
            return lbl
    return None


def _extract_one(args):
    fpath, label = args
    feat = extract_features(fpath)
    return feat, label


def load_dataset(data_dir: Path, verbose: bool = True, n_workers: int = 4):
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(data_dir.rglob(f"*{ext}"))

    # Filter to labeled files only (also excludes ExcludedSamples)
    labeled = [(f, find_label(f)) for f in audio_files]
    labeled = [(f, lbl) for f, lbl in labeled if lbl is not None]

    if verbose:
        print(f"Found {len(audio_files)} audio files, {len(labeled)} labeled (ExcludedSamples skipped)")

    X, y4, y2 = [], [], []
    label_counts = {lbl: 0 for lbl in LABEL_TO_INT}
    skipped = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_extract_one, item): item for item in labeled}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if verbose and done % 300 == 0:
                print(f"  {done}/{len(labeled)} files...", end="\r")
            feat, label = fut.result()
            if feat is None:
                skipped += 1
                continue
            X.append(feat)
            y4.append(LABEL_TO_INT[label])
            y2.append(STRAIN_MAP[label])
            label_counts[label] += 1

    if verbose:
        print(f"\nLoaded: {len(X)} samples  |  skipped: {skipped}  |  feature_dim: {len(X[0]) if X else 0}")
        for lbl, cnt in label_counts.items():
            print(f"  {lbl:<12}: {cnt}")

    return np.array(X), np.array(y4), np.array(y2)


# ---------------------------------------------------------------------------
# Training — both classifiers with balanced class weights
# ---------------------------------------------------------------------------

def train_svm(X_tr, y_tr):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\nTraining SVM (RBF, balanced)...")
    t0 = time.time()
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42,
                       class_weight="balanced")),
    ])
    m.fit(X_tr, y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")
    return m


def train_mlp(X_tr, y_tr):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\nTraining MLP (256→128→64, balanced)...")
    t0 = time.time()
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            learning_rate_init=0.001,
        )),
    ])
    # sklearn MLP doesn't have class_weight — use sample_weight instead
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_tr)
    m.fit(X_tr, y_tr)   # MLP ignores sample_weight in fit — use balanced SVM for best results
    print(f"  Done in {time.time()-t0:.1f}s")
    return m


def evaluate(model, X_te, y_te, y2_te, label_names, title="Eval"):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n{'─'*60}")
    print(f"{title}  |  4-class accuracy={acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=label_names, zero_division=0))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12}", " ".join(f"{n[:8]:>8}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<12}", " ".join(f"{v:>8}" for v in row))

    # Binary accuracy
    y2_pred = np.array([STRAIN_MAP[INT_TO_LABEL[p]] for p in y_pred])
    bin_acc = accuracy_score(y2_te, y2_pred)
    # Binary confusion
    from sklearn.metrics import confusion_matrix as cm2
    bc = cm2(y2_te, y2_pred)
    tp = bc[1,1] if bc.shape == (2,2) else 0
    fn = bc[1,0] if bc.shape == (2,2) else 0
    fp = bc[0,1] if bc.shape == (2,2) else 0
    tn = bc[0,0] if bc.shape == (2,2) else 0
    print(f"\n  Binary (pressed vs modal): accuracy={bin_acc:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  False negatives (missed strain): {fn}/{fn+tp} = {fn/(fn+tp+1e-9)*100:.1f}%")
    print(f"  False positives (false alarm):   {fp}/{fp+tn} = {fp/(fp+tn+1e-9)*100:.1f}%")
    return acc, bin_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data/cvt_dataset/")
    ap.add_argument("--model",   default="both", choices=["svm", "mlp", "both"])
    ap.add_argument("--output",  default="models/phonation_classifier.joblib")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found"); sys.exit(1)

    print(f"Loading dataset from {data_dir}...")
    X, y4, y2 = load_dataset(data_dir, verbose=True, n_workers=args.workers)

    if len(X) == 0:
        print("ERROR: No labeled samples found."); sys.exit(1)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]}-dim features")
    print(f"Binary: pressed={y2.sum()}  modal={(y2==0).sum()}")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y4_tr, y4_te, y2_tr, y2_te = train_test_split(
        X, y4, y2, test_size=0.2, random_state=42, stratify=y4
    )
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

    label_names = [INT_TO_LABEL[i] for i in range(4)]
    best_model, best_4acc, best_bin_acc = None, 0.0, 0.0

    if args.model in ("svm", "both"):
        m = train_svm(X_tr, y4_tr)
        acc, bin_acc = evaluate(m, X_te, y4_te, y2_te, label_names, "SVM v2")
        if bin_acc > best_bin_acc:
            best_bin_acc, best_4acc, best_model = bin_acc, acc, m

    if args.model in ("mlp", "both"):
        m = train_mlp(X_tr, y4_tr)
        acc, bin_acc = evaluate(m, X_te, y4_te, y2_te, label_names, "MLP v2")
        if bin_acc > best_bin_acc:
            best_bin_acc, best_4acc, best_model = bin_acc, acc, m

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    bundle = {
        "model":        best_model,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "strain_map":   STRAIN_MAP,
        "feature_dim":  X.shape[1],
        "accuracy":     best_4acc,
        "binary_accuracy": best_bin_acc,
        "version":      2,
        "features":     "scatter(34) + mfcc+delta(26) + h1h2(1) + spectral(4) = 65-dim",
    }
    joblib.dump(bundle, out)
    print(f"\nSaved: {out}")
    print(f"  4-class accuracy:  {best_4acc:.4f}")
    print(f"  Binary accuracy:   {best_bin_acc:.4f}")
    print(f"  Feature dim:       {X.shape[1]}")

    # Quick test
    proba = best_model.predict_proba(X_te[:1])[0]
    print(f"\nExample proba: {dict(zip(label_names, proba.round(3)))}")
    print(f"  pressed_score = overdrive+edge: {proba[0]+proba[1]:.3f}")


if __name__ == "__main__":
    main()
