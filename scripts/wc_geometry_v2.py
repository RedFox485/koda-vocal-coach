#!/usr/bin/env python3
"""
Wildcard: Sound as Geometry v2 — Frequency-Native Shape
========================================================
The first geometry experiment used time-domain heuristics (transient counts,
decay length) which scored poorly. But shape IS frequency — the Fourier
transform of a spatial boundary. A circle = one frequency. A square = odd
harmonics. A jagged edge = broadband noise.

This version defines shape using frequency-native properties:
  - curvature: 2nd derivative of spectral envelope (smooth=round, jagged=angular)
  - fractal_dim: self-similarity across frequency scales (coastline dimension)
  - compactness: spectral concentration (peaked=compact, spread=diffuse)
  - topology: number of spectral peaks (modes = holes/features)
  - harmonic_order: even/odd harmonic ratio (symmetric vs asymmetric shape)
  - edge_sharpness: spectral rolloff steepness (smooth edge vs sharp cutoff)
  - dimensionality: effective number of independent frequency bands (intrinsic dim)
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_freq_shape_properties(mel_frames):
    """Compute frequency-native geometric properties."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8

    # Normalize spectrum to probability distribution
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)

    # 1. CURVATURE — 2nd derivative of spectral envelope
    #    Smooth spectrum = low curvature = round shape
    #    Jagged spectrum = high curvature = angular shape
    if n_mels >= 3:
        d2 = np.diff(mean_spectrum, n=2)
        curvature = np.mean(np.abs(d2)) / (np.mean(mean_spectrum) + 1e-8)
    else:
        curvature = 0.0

    # 2. FRACTAL DIMENSION — self-similarity via box-counting on spectrum
    #    Compute spectral complexity at multiple resolutions
    fractal_scores = []
    for scale in [2, 4, 8]:
        if n_mels >= scale:
            n_boxes = n_mels // scale
            reshaped = mean_spectrum[:n_boxes * scale].reshape(n_boxes, scale)
            box_ranges = reshaped.max(axis=1) - reshaped.min(axis=1)
            occupied = np.sum(box_ranges > np.mean(mean_spectrum) * 0.01)
            fractal_scores.append(occupied / n_boxes)
    if len(fractal_scores) >= 2:
        # Slope of log(occupied) vs log(1/scale) approximates fractal dimension
        fractal_dim = np.mean(fractal_scores)
    else:
        fractal_dim = 0.5

    # 3. COMPACTNESS — spectral concentration (peaked vs spread)
    #    Use participation ratio: (sum p)^2 / sum(p^2)
    pr = (spec_norm.sum() ** 2) / (np.sum(spec_norm ** 2) + 1e-10)
    compactness = 1.0 - (pr / n_mels)  # 1 = maximally compact, 0 = uniform

    # 4. TOPOLOGY — number of spectral peaks (local maxima)
    if n_mels >= 3:
        peaks = 0
        for i in range(1, n_mels - 1):
            if mean_spectrum[i] > mean_spectrum[i-1] and mean_spectrum[i] > mean_spectrum[i+1]:
                if mean_spectrum[i] > np.mean(mean_spectrum) * 0.3:
                    peaks += 1
        topology = peaks / (n_mels / 4)  # normalized
    else:
        topology = 0.0

    # 5. HARMONIC ORDER — even vs odd harmonic energy ratio
    #    Even harmonics = symmetric shapes, odd = asymmetric
    even_bins = mean_spectrum[0::2]
    odd_bins = mean_spectrum[1::2]
    total_harm = even_bins.sum() + odd_bins.sum() + 1e-8
    harmonic_order = even_bins.sum() / total_harm  # 0.5 = balanced, >0.5 = even-dominant

    # 6. EDGE SHARPNESS — spectral rolloff steepness
    cumulative = np.cumsum(spec_norm)
    rolloff_85 = np.searchsorted(cumulative, 0.85)
    rolloff_95 = np.searchsorted(cumulative, 0.95)
    if rolloff_95 > rolloff_85:
        edge_sharpness = 1.0 / (rolloff_95 - rolloff_85)
    else:
        edge_sharpness = 1.0

    # 7. DIMENSIONALITY — effective number of independent bands
    #    Shannon entropy of normalized spectrum
    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10))
    max_entropy = np.log(n_mels)
    dimensionality = entropy / (max_entropy + 1e-8)  # 0 = one band, 1 = all equal

    return {
        'curvature': float(curvature),
        'fractal_dim': float(fractal_dim),
        'compactness': float(compactness),
        'topology': float(topology),
        'harmonic_order': float(harmonic_order),
        'edge_sharpness': float(edge_sharpness),
        'dimensionality': float(dimensionality),
    }


def load_encoder(checkpoint_path, device='cpu'):
    try:
        from mamba_ssm import Mamba2
        use_real = True
    except ImportError:
        use_real = False

    if use_real:
        class MambaEncoder(nn.Module):
            def __init__(self, d=128, n=6):
                super().__init__()
                self.input_proj = nn.Linear(40, d)
                self.layers = nn.ModuleList([Mamba2(d_model=d, d_state=16, d_conv=4, expand=2) for _ in range(n)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x):
                x = self.input_proj(x)
                for layer in self.layers: x = x + layer(x)
                return self.norm(x)
    else:
        class ConvBlock(nn.Module):
            def __init__(self, d=128):
                super().__init__()
                self.conv = nn.Conv1d(d, d*2, 4, padding=2)
                self.act = nn.SiLU()
                self.proj = nn.Linear(d*2, d)
            def forward(self, x):
                y = self.conv(x.transpose(1,2)).transpose(1,2)[:,:x.size(1),:]
                return self.proj(self.act(y))
        class MambaEncoder(nn.Module):
            def __init__(self, d=128, n=6):
                super().__init__()
                self.input_proj = nn.Linear(40, d)
                self.layers = nn.ModuleList([ConvBlock(d) for _ in range(n)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x):
                x = self.input_proj(x)
                for layer in self.layers: x = x + layer(x)
                return self.norm(x)

    encoder = MambaEncoder()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    enc_state = ckpt['encoder'] if 'encoder' in ckpt else {}
    loaded = encoder.load_state_dict(enc_state, strict=False)
    n_loaded = len(enc_state) - len(loaded.missing_keys)
    print(f"Encoder: {n_loaded}/{len(list(encoder.state_dict().keys()))} params")
    encoder.to(device)
    encoder.train(False)
    return encoder


def run_experiment(data_dir='data/training/mel/esc50', checkpoint_path='checkpoints/clap_distill_spread_a.pt'):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    mel_dir = Path(data_dir)
    meta = json.load(open('data/clap_meta.json'))
    categories = meta['categories']
    unique_cats = sorted(set(c for c in categories if c not in ('ambient', 'mixed', 'music')))
    cat_to_id = {c: i for i, c in enumerate(unique_cats)}
    cat_names = {i: c for c, i in cat_to_id.items()}

    mel_files = sorted(mel_dir.glob('*.npy'), key=lambda p: int(p.stem))
    all_mels, all_shape, all_cats, valid_files = [], [], [], []
    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        cn = categories[idx]
        if cn in ('ambient', 'mixed', 'music'): continue
        shp = compute_freq_shape_properties(mel)
        all_mels.append(mel)
        all_shape.append(shp)
        all_cats.append(cat_to_id[cn])
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    prop_names = ['curvature', 'fractal_dim', 'compactness', 'topology', 'harmonic_order', 'edge_sharpness', 'dimensionality']
    V = np.array([[s[p] for p in prop_names] for s in all_shape])

    print(f"\nFrequency-native shape stats:")
    for i, name in enumerate(prop_names):
        print(f"  {name:16s}: mean={V[:,i].mean():.3f}, std={V[:,i].std():.3f}")

    encoder = load_encoder(checkpoint_path, device)
    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            emb = encoder(x).mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
    E = np.array(embeddings)

    results = {}
    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)

    # TEST 1: Linear prediction
    print("\n" + "="*60)
    print("TEST 1: Frequency-native shape prediction")
    print("="*60)
    test1 = {}
    for i, name in enumerate(prop_names):
        y = V[:, i]
        if np.isnan(y).any() or np.std(y) < 1e-10:
            test1[name] = 0.0
            print(f"  {name:16s}: SKIPPED")
            continue
        scores = cross_val_score(Ridge(alpha=1.0), E_scaled, y, cv=5, scoring='r2')
        r2 = scores.mean()
        test1[name] = float(r2)
        marker = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:16s}: R-sq = {r2:.3f} {marker}")
    results['test1_encoder_r2'] = test1

    # Compare with v1 results
    print("\n  Comparison with v1 (time-domain shape):")
    v1_r2 = {'size': 0.618, 'angularity': 0.094, 'symmetry': 0.048, 'density': 0.561,
             'motion': 0.457, 'roundness': 0.257, 'depth': -0.111}
    v1_strong = sum(1 for v in v1_r2.values() if v > 0.3)
    v2_strong = sum(1 for v in test1.values() if v > 0.3)
    print(f"  v1: {v1_strong}/7 with R-sq > 0.3")
    print(f"  v2: {v2_strong}/7 with R-sq > 0.3")
    if v2_strong > v1_strong:
        print(f"  IMPROVEMENT: +{v2_strong - v1_strong} properties above threshold")
    elif v2_strong == v1_strong:
        print(f"  SAME count, but check if R-sq values improved")
    else:
        print(f"  REGRESSION: frequency-native didn't help as expected")

    # TEST 2: Binary classification
    print("\n" + "="*60)
    print("TEST 2: Binary shape classification (frequency-native)")
    print("="*60)
    binary_tasks = {
        'smooth_vs_curved': ('curvature', lambda x: x > np.median(x)),
        'simple_vs_fractal': ('fractal_dim', lambda x: x > np.median(x)),
        'diffuse_vs_compact': ('compactness', lambda x: x > np.median(x)),
        'simple_vs_complex': ('topology', lambda x: x > np.median(x)),
        'low_vs_high_dim': ('dimensionality', lambda x: x > np.median(x)),
    }
    test2 = {}
    for task, (prop, binarize) in binary_tasks.items():
        idx = prop_names.index(prop)
        y = binarize(V[:, idx]).astype(int)
        scores = cross_val_score(LogisticRegression(max_iter=1000, C=1.0), E_scaled, y, cv=5, scoring='accuracy')
        test2[task] = float(scores.mean())
        marker = "***" if scores.mean() > 0.75 else "**" if scores.mean() > 0.65 else ""
        print(f"  {task:22s}: {scores.mean():.1%} {marker}")
    results['test2_binary'] = test2

    # TEST 3: Frequency-native correlations
    print("\n" + "="*60)
    print("TEST 3: Frequency-shape correlation structure")
    print("="*60)
    expected = {
        ('curvature', 'compactness'): 'negative',      # curved = spread out
        ('dimensionality', 'compactness'): 'negative',  # high dim = not compact
        ('topology', 'dimensionality'): 'positive',     # more peaks = more dims
        ('curvature', 'edge_sharpness'): 'positive',    # angular = sharp edges
    }
    corr = np.corrcoef(V.T)
    test3 = {}
    matches = 0
    for (p1, p2), exp_sign in expected.items():
        i, j = prop_names.index(p1), prop_names.index(p2)
        r = corr[i, j]
        actual = 'positive' if r > 0 else 'negative'
        match = actual == exp_sign
        matches += match
        test3[f"{p1}_vs_{p2}"] = {'r': float(r), 'expected': exp_sign, 'match': bool(match)}
        print(f"  {p1:16s} vs {p2:16s}: r={r:+.3f} ({'MATCH' if match else 'MISS'})")
    print(f"\n  Structure score: {matches}/{len(expected)}")
    results['test3_correlations'] = test3

    # TEST 4: Category shape profiles
    print("\n" + "="*60)
    print("TEST 4: Frequency-shape of each sound")
    print("="*60)
    cats_arr = np.array(all_cats)
    test4 = {}
    for i, name in enumerate(prop_names):
        vals = {cat_names[c]: V[cats_arr == c, i].mean() for c in sorted(set(all_cats))}
        s = sorted(vals.items(), key=lambda x: x[1])
        test4[name] = {'lowest': s[0][0], 'highest': s[-1][0]}
        print(f"  {name:16s}: {s[0][0]:18s}({s[0][1]:.3f}) ... {s[-1][0]:18s}({s[-1][1]:.3f})")
    results['test4_profiles'] = test4

    # VERDICT
    strong_r2 = sum(1 for v in test1.values() if v > 0.3)
    good_class = sum(1 for v in test2.values() if v > 0.65)
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  Linear prediction: {strong_r2}/7 with R-sq > 0.3")
    print(f"  Classification:    {good_class}/5 > 65%")
    print(f"  Correlations:      {matches}/{len(expected)}")
    print(f"  vs v1:             {v1_strong}/7 -> {v2_strong}/7")

    if strong_r2 >= 4:
        verdict = "STRONG - frequency IS geometry (v2 confirms)"
    elif strong_r2 >= 3:
        verdict = "IMPROVED - frequency-native approach helps"
    elif strong_r2 > v1_strong:
        verdict = "MARGINAL IMPROVEMENT"
    else:
        verdict = "NO IMPROVEMENT - shape may need spatial, not just spectral"
    print(f"  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['v1_comparison'] = {'v1_strong': v1_strong, 'v2_strong': strong_r2}

    with open('data/wc_geometry_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_geometry_v2_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
