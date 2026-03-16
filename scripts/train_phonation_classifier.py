#!/usr/bin/env python3
"""
Phonation Mode Classifier — Training Script
Dataset: CVT Vocal Mode Dataset (Zenodo 14276415)
         ~3,752 unique productions, 4 CVT vocal modes, 4 microphones

Pipeline:
  1. Scan dataset directory for labeled audio + annotations
  2. Extract 34-dim log-compressed wavelet scatter features per sample
  3. Train: SVM baseline, then small MLP
  4. Evaluate: accuracy, confusion matrix, per-class scores
  5. Save model to models/phonation_classifier.joblib (joblib format)

CVT Modes → Strain Mapping:
  overdrive / edge  → pressed/strained
  neutral / curbing → modal/healthy

Usage:
    .venv/bin/python3 scripts/train_phonation_classifier.py --data data/cvt_dataset/
    .venv/bin/python3 scripts/train_phonation_classifier.py --data data/cvt_dataset/ --model mlp
    .venv/bin/python3 scripts/train_phonation_classifier.py --data data/cvt_dataset/ --eval-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D

SR          = 44100
CHUNK       = 4096    # ~93ms — same as backend
SILENCE_RMS = 0.008
J, Q        = 7, 1   # 34-dim scatter features

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
# Feature extraction
# ---------------------------------------------------------------------------

def scatter_features(chunk: np.ndarray):
    """34-dim log-compressed scatter features. RMS-normalized, loudness-invariant."""
    x = chunk.astype(np.float64)
    if len(x) < CHUNK:
        x = np.pad(x, (0, CHUNK - len(x)))
    else:
        x = x[:CHUNK]
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-8:
        return None
    x = x / rms
    Sx  = _scatter(x)
    feat = np.mean(Sx, axis=1)
    return np.log(np.abs(feat) + 1e-10)


def extract_features_from_file(path: Path, n_chunks: int = 5):
    """Load audio, extract scatter from middle third (avoids onset/offset).
    Returns mean 34-dim feature or None if silent."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception as e:
        return None

    segment = y[len(y)//3 : 2*len(y)//3]
    step = max(1, len(segment) // n_chunks)
    feats = []
    for i in range(n_chunks):
        chunk = segment[i*step : i*step + CHUNK]
        if len(chunk) < CHUNK // 2:
            continue
        if float(np.sqrt(np.mean(chunk**2))) < SILENCE_RMS:
            continue
        feat = scatter_features(chunk)
        if feat is not None:
            feats.append(feat)

    return np.mean(np.stack(feats), axis=0) if feats else None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def find_label_from_path(path: Path):
    for part in reversed([p.lower() for p in path.parts]):
        for keyword, label in CVT_KEYWORDS.items():
            if keyword in part:
                return label
    name = path.stem.lower()
    for keyword, label in CVT_KEYWORDS.items():
        if keyword in name:
            return label
    return None


def load_annotation_file(ann_path: Path):
    try:
        if ann_path.suffix == ".json":
            data = json.loads(ann_path.read_text())
            for key in ("mode", "label", "class", "annotation", "vocal_mode"):
                if key in data:
                    val = str(data[key]).lower()
                    for kw, lbl in CVT_KEYWORDS.items():
                        if kw in val:
                            return lbl
        else:
            text = ann_path.read_text().strip().lower()
            for kw, lbl in CVT_KEYWORDS.items():
                if kw in text:
                    return lbl
    except Exception:
        pass
    return None


def load_dataset(data_dir: Path, verbose: bool = True):
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(data_dir.rglob(f"*{ext}"))

    if verbose:
        print(f"Found {len(audio_files)} audio files in {data_dir}")

    # Try to find annotation summary files
    ann_lookup = {}
    for ann_file in data_dir.rglob("*.json"):
        if "annotation" in ann_file.name.lower() or "label" in ann_file.name.lower():
            try:
                data = json.loads(ann_file.read_text())
                if isinstance(data, list):
                    for item in data:
                        fname = item.get("filename", item.get("file", ""))
                        for key in ("mode", "label", "class", "annotation"):
                            if key in item:
                                val = str(item[key]).lower()
                                for kw, lbl in CVT_KEYWORDS.items():
                                    if kw in val:
                                        ann_lookup[Path(fname).stem] = lbl
                                        break
            except Exception:
                pass

    X, y4, y2 = [], [], []
    label_counts = {lbl: 0 for lbl in LABEL_TO_INT}
    skipped = 0

    for i, fpath in enumerate(audio_files):
        if verbose and i % 200 == 0:
            print(f"  {i}/{len(audio_files)} files...", end="\r")

        label = ann_lookup.get(fpath.stem) or find_label_from_path(fpath)
        if label is None:
            for ext in (".json", ".txt"):
                ann = fpath.with_suffix(ext)
                if ann.exists():
                    label = load_annotation_file(ann)
                    if label:
                        break
        if label is None:
            skipped += 1
            continue

        feat = extract_features_from_file(fpath)
        if feat is None:
            skipped += 1
            continue

        X.append(feat)
        y4.append(LABEL_TO_INT[label])
        y2.append(STRAIN_MAP[label])
        label_counts[label] += 1

    if verbose:
        print(f"\nLoaded: {len(X)} samples  |  skipped: {skipped}")
        for lbl, cnt in label_counts.items():
            print(f"  {lbl:<12}: {cnt}")

    return np.array(X), np.array(y4), np.array(y2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_svm(X_tr, y_tr):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\nTraining SVM (RBF kernel)...")
    t0 = time.time()
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
    ])
    m.fit(X_tr, y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")
    return m


def train_mlp(X_tr, y_tr):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\nTraining MLP (128→64→4)...")
    t0 = time.time()
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])
    m.fit(X_tr, y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")
    return m


def evaluate(model, X_te, y_te, label_names, title="Eval"):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n{'─'*55}")
    print(f"{title}  |  accuracy={acc:.3f}")
    print(classification_report(y_te, y_pred, target_names=label_names, zero_division=0))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12}", " ".join(f"{n[:6]:>8}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<12}", " ".join(f"{v:>8}" for v in row))
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       default="data/cvt_dataset/")
    ap.add_argument("--model",      default="both", choices=["svm", "mlp", "both"])
    ap.add_argument("--eval-only",  action="store_true")
    ap.add_argument("--output",     default="models/phonation_classifier.joblib")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Loading dataset from {data_dir}...")
    X, y4, y2 = load_dataset(data_dir, verbose=True)

    if len(X) == 0:
        print("ERROR: No labeled samples found.")
        print("Expected: audio files with CVT mode labels in directory/file names or annotation JSON.")
        print("Looking for keywords: overdrive, edge, neutral, curbing")
        sys.exit(1)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]}-dim scatter features")
    print(f"Binary: strained={y2.sum()}  healthy={(y2==0).sum()}")

    if args.eval_only:
        out = Path(args.output)
        if not out.exists():
            print(f"ERROR: model not found: {out}")
            sys.exit(1)
        bundle = joblib.load(out)
        label_names = [bundle["int_to_label"][i] for i in range(4)]
        evaluate(bundle["model"], X, y4, label_names, title="Full dataset (no train/test split)")
        return

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y4_tr, y4_te, y2_tr, y2_te = train_test_split(
        X, y4, y2, test_size=0.2, random_state=42, stratify=y4
    )
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

    label_names_4 = [INT_TO_LABEL[i] for i in range(4)]
    best_model, best_acc = None, 0.0

    if args.model in ("svm", "both"):
        m = train_svm(X_tr, y4_tr)
        acc = evaluate(m, X_te, y4_te, label_names_4, title="SVM 4-class")
        from sklearn.metrics import accuracy_score
        y2_pred = np.array([STRAIN_MAP[INT_TO_LABEL[p]] for p in m.predict(X_te)])
        print(f"  Binary strain accuracy: {accuracy_score(y2_te, y2_pred):.3f}")
        if acc > best_acc:
            best_acc, best_model = acc, m

    if args.model in ("mlp", "both"):
        m = train_mlp(X_tr, y4_tr)
        acc = evaluate(m, X_te, y4_te, label_names_4, title="MLP 4-class")
        from sklearn.metrics import accuracy_score
        y2_pred = np.array([STRAIN_MAP[INT_TO_LABEL[p]] for p in m.predict(X_te)])
        print(f"  Binary strain accuracy: {accuracy_score(y2_te, y2_pred):.3f}")
        if acc > best_acc:
            best_acc, best_model = acc, m

    # Save
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    bundle = {
        "model":        best_model,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "strain_map":   STRAIN_MAP,
        "feature_dim":  X.shape[1],
        "accuracy":     best_acc,
        "dataset_size": len(X),
    }
    joblib.dump(bundle, out)
    print(f"\nSaved: {out}  (accuracy={best_acc:.3f})")

    # Inference snippet
    print("\n── Inference ──")
    proba = best_model.predict_proba(X_te[:1])[0]
    print(f"Example proba: {dict(zip(label_names_4, proba.round(3)))}")
    print(f"  overdrive+edge = pressed_strain_score: {proba[0]+proba[1]:.3f}")


if __name__ == "__main__":
    main()
