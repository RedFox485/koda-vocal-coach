#!/usr/bin/env python3
"""
Wildcard Experiment: Sound as Geometry
=======================================
Can audio embeddings predict geometric/shape properties?

The bouba/kiki effect proves humans map sound to shape automatically.
This experiment tests whether our encoder does the same without training.

Shape-analog properties computed from mel spectrograms:
  - size: low-frequency energy (large objects = low resonance)
  - angularity: transient density (sharp = angular, smooth = round)
  - symmetry: harmonic regularity (periodic = symmetric, chaotic = asymmetric)
  - density: spectral fill (more bins active = denser)
  - motion: spectral flux direction (changing = moving, static = still)
  - roundness: spectral smoothness (smooth envelope = round, jagged = spiky)
  - depth: reverberance / decay length (long decay = deep/hollow)

Expected: roundness and angularity should be strongly anti-correlated.
Size and depth should correlate (big things sound deep/hollow).
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


def compute_shape_properties(mel_frames):
    """Compute geometric/shape-analog properties from log-mel spectrogram."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # 1. SIZE - low-frequency energy ratio (large = bass-heavy)
    low_cut = n_mels // 3
    size = np.sum(mean_spectrum[:low_cut]) / (np.sum(mean_spectrum) + 1e-8)

    # 2. ANGULARITY - transient density (sharp attacks per unit time)
    if T >= 3:
        energy_diff = np.abs(np.diff(frame_energy))
        threshold = np.mean(energy_diff) + np.std(energy_diff)
        transients = np.sum(energy_diff > threshold) / T
        angularity = np.clip(transients, 0, 1)
    else:
        angularity = 0.0

    # 3. SYMMETRY - harmonic regularity via autocorrelation peak strength
    if T >= 10:
        spec_centered = mel_linear - mel_linear.mean(axis=0)
        # Average temporal autocorrelation across bands
        sym_scores = []
        for b in range(0, n_mels, 4):  # sample every 4th band
            band = spec_centered[:, b]
            norm = np.sum(band ** 2)
            if norm > 1e-10:
                ac = np.correlate(band, band, mode='full')
                ac = ac[len(ac)//2:]
                ac = ac / (norm + 1e-8)
                # Peak in autocorrelation = regularity
                if len(ac) > 2:
                    peaks = ac[1:].max()
                    sym_scores.append(max(peaks, 0))
        symmetry = np.mean(sym_scores) if sym_scores else 0.0
    else:
        symmetry = 0.0

    # 4. DENSITY - spectral fill (what fraction of bins are active)
    active_threshold = np.mean(mean_spectrum) * 0.1
    density = np.sum(mean_spectrum > active_threshold) / n_mels

    # 5. MOTION - spectral flux (how much the spectrum changes over time)
    if T >= 2:
        flux = np.mean(np.sqrt(np.sum(np.diff(mel_linear, axis=0) ** 2, axis=1)))
        motion = flux / (np.mean(np.sqrt(np.sum(mel_linear ** 2, axis=1))) + 1e-8)
    else:
        motion = 0.0

    # 6. ROUNDNESS - spectral envelope smoothness (smooth = round, jagged = spiky)
    spec_diff = np.abs(np.diff(mean_spectrum))
    max_diff = np.max(spec_diff) + 1e-8
    roundness = 1.0 - (np.mean(spec_diff) / max_diff)

    # 7. DEPTH - decay length (how long energy persists after peak)
    if T >= 4:
        peak_idx = np.argmax(frame_energy)
        if peak_idx < T - 2:
            tail = frame_energy[peak_idx:]
            if tail[0] > 1e-8:
                # Time to decay to 1/e of peak
                decay_level = tail[0] / np.e
                below = np.where(tail < decay_level)[0]
                if len(below) > 0:
                    depth = below[0] / T
                else:
                    depth = 1.0  # never decays = very deep
            else:
                depth = 0.0
        else:
            depth = 0.0
    else:
        depth = 0.0

    return {
        'size': float(size),
        'angularity': float(angularity),
        'symmetry': float(symmetry),
        'density': float(density),
        'motion': float(motion),
        'roundness': float(roundness),
        'depth': float(depth),
    }


def load_encoder(checkpoint_path, device='cpu'):
    """Load encoder with conv_stub fallback."""
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
        shp = compute_shape_properties(mel)
        all_mels.append(mel)
        all_shape.append(shp)
        all_cats.append(cat_to_id[cn])
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    prop_names = ['size', 'angularity', 'symmetry', 'density', 'motion', 'roundness', 'depth']
    V = np.array([[s[p] for p in prop_names] for s in all_shape])

    print(f"\nShape property stats:")
    for i, name in enumerate(prop_names):
        print(f"  {name:12s}: mean={V[:,i].mean():.3f}, std={V[:,i].std():.3f}")

    encoder = load_encoder(checkpoint_path, device)
    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            emb = encoder(x).mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
    E = np.array(embeddings)

    # CLAP
    clap_embs = None
    clap_path = Path('data/clap_embeddings.npz')
    if clap_path.exists():
        clap_data = np.load(clap_path)
        clap_E = clap_data['embeddings']
        matched = [clap_E[int(f.replace('.npy',''))] for f in valid_files if int(f.replace('.npy','')) < len(clap_E)]
        if len(matched) == len(valid_files):
            clap_embs = np.array(matched)

    results = {}
    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)

    # TEST 1: Linear prediction
    print("\n" + "="*60)
    print("TEST 1: Can embeddings predict geometric properties?")
    print("="*60)
    test1 = {}
    for i, name in enumerate(prop_names):
        y = V[:, i]
        if np.isnan(y).any() or np.std(y) < 1e-10:
            test1[name] = 0.0
            continue
        scores = cross_val_score(Ridge(alpha=1.0), E_scaled, y, cv=5, scoring='r2')
        r2 = scores.mean()
        test1[name] = float(r2)
        marker = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:12s}: R-sq = {r2:.3f} {marker}")
    results['test1_encoder_r2'] = test1

    if clap_embs is not None:
        print("\n  CLAP comparison:")
        clap_scaled = StandardScaler().fit_transform(clap_embs)
        test1c = {}
        for i, name in enumerate(prop_names):
            y = V[:, i]
            if np.isnan(y).any() or np.std(y) < 1e-10: continue
            scores = cross_val_score(Ridge(alpha=1.0), clap_scaled, y, cv=5, scoring='r2')
            test1c[name] = float(scores.mean())
            print(f"  {name:12s}: R-sq = {scores.mean():.3f}")
        results['test1_clap_r2'] = test1c

    # TEST 2: Binary shape classification
    print("\n" + "="*60)
    print("TEST 2: Binary shape classification (bouba/kiki)")
    print("="*60)
    binary_tasks = {
        'small_vs_large': ('size', lambda x: x > np.median(x)),
        'round_vs_angular': ('angularity', lambda x: x > np.median(x)),
        'regular_vs_chaotic': ('symmetry', lambda x: x > np.median(x)),
        'sparse_vs_dense': ('density', lambda x: x > np.median(x)),
        'still_vs_moving': ('motion', lambda x: x > np.median(x)),
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

    # TEST 3: Cross-modal structure
    print("\n" + "="*60)
    print("TEST 3: Shape correlation structure")
    print("="*60)
    expected = {
        ('roundness', 'angularity'): 'negative',  # round vs angular
        ('size', 'depth'): 'positive',             # big = deep
        ('symmetry', 'roundness'): 'positive',     # regular = round
        ('motion', 'angularity'): 'positive',      # moving = sharp transients
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
        test3[f"{p1}_vs_{p2}"] = {'r': float(r), 'expected': exp_sign, 'match': match}
        print(f"  {p1:12s} vs {p2:12s}: r={r:+.3f} ({exp_sign}, {'MATCH' if match else 'MISMATCH'})")
    print(f"\n  Shape structure score: {matches}/{len(expected)}")
    results['test3_correlations'] = test3

    # TEST 4: Per-category shape profiles
    print("\n" + "="*60)
    print("TEST 4: What shape is each sound?")
    print("="*60)
    cats_arr = np.array(all_cats)
    test4 = {}
    for i, name in enumerate(prop_names):
        vals = {cat_names[c]: V[cats_arr == c, i].mean() for c in sorted(set(all_cats))}
        s = sorted(vals.items(), key=lambda x: x[1])
        test4[name] = {'lowest': s[0][0], 'highest': s[-1][0]}
        print(f"  {name:12s}: lowest={s[0][0]:20s} ({s[0][1]:.3f})  highest={s[-1][0]:20s} ({s[-1][1]:.3f})")
    results['test4_profiles'] = test4

    # BOUBA/KIKI specific test
    print("\n" + "="*60)
    print("TEST 5: Bouba/Kiki mapping")
    print("="*60)
    # Which sounds are most bouba (round, smooth, large) vs kiki (angular, sharp, small)?
    bouba_score = V[:, prop_names.index('roundness')] + V[:, prop_names.index('size')] - V[:, prop_names.index('angularity')]
    kiki_score = V[:, prop_names.index('angularity')] + V[:, prop_names.index('motion')] - V[:, prop_names.index('roundness')]

    cat_bouba = {}
    cat_kiki = {}
    for c in sorted(set(all_cats)):
        mask = cats_arr == c
        cat_bouba[cat_names[c]] = float(bouba_score[mask].mean())
        cat_kiki[cat_names[c]] = float(kiki_score[mask].mean())

    most_bouba = sorted(cat_bouba.items(), key=lambda x: x[1], reverse=True)[:5]
    most_kiki = sorted(cat_kiki.items(), key=lambda x: x[1], reverse=True)[:5]

    print("  Most BOUBA (round, large, smooth):")
    for name, score in most_bouba:
        print(f"    {name:20s}: {score:.3f}")
    print("  Most KIKI (angular, sharp, small):")
    for name, score in most_kiki:
        print(f"    {name:20s}: {score:.3f}")

    results['test5_bouba'] = [{'cat': n, 'score': s} for n, s in most_bouba]
    results['test5_kiki'] = [{'cat': n, 'score': s} for n, s in most_kiki]

    # VERDICT
    strong_r2 = sum(1 for v in test1.values() if v > 0.3)
    good_class = sum(1 for v in test2.values() if v > 0.65)
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  Linear prediction: {strong_r2}/7 with R-sq > 0.3")
    print(f"  Classification:    {good_class}/5 > 65%")
    print(f"  Correlations:      {matches}/{len(expected)} match shape physics")

    if strong_r2 >= 4 and matches >= 3:
        verdict = "STRONG - sound encodes geometric shape"
    elif strong_r2 >= 2 or matches >= 2:
        verdict = "PARTIAL - some shape dimensions present"
    else:
        verdict = "WEAK - shape mapping is weak"
    print(f"  VERDICT: {verdict}")
    results['verdict'] = verdict

    with open('data/wc_sound_as_geometry_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_sound_as_geometry_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
