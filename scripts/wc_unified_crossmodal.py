#!/usr/bin/env python3
"""
Wildcard: Unified Cross-Modal Space
=====================================
If frequency is the fabric of reality, then light, touch, geometry, and
emotion shouldn't be independent modalities — they should be projections
of the SAME underlying frequency structure.

This experiment tests:
1. Do all cross-modal properties share a low-dimensional subspace?
   (PCA on combined property matrix — if unified, few components explain most variance)
2. Can we predict one modality from another?
   (If touch predicts light, they share structure)
3. What is the shared frequency basis?
   (Which spectral features span the unified space?)
4. Redundancy analysis: how many truly independent cross-modal dimensions exist?
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def compute_all_properties(mel_frames):
    """Compute ALL cross-modal properties from log-mel spectrogram."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # Shared computations
    centroid = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)
    geo_mean = np.exp(np.mean(np.log(mean_spectrum + 1e-10)))
    flatness = geo_mean / (np.mean(mean_spectrum) + 1e-8)
    rms = np.sqrt(np.mean(mel_linear ** 2))
    third = n_mels // 3
    low_ratio = np.sum(mean_spectrum[:third]) / (np.sum(mean_spectrum) + 1e-8)

    bw_sq = np.sum(((freq_bins - centroid) ** 2) * mean_spectrum) / np.sum(mean_spectrum)
    bandwidth = np.sqrt(max(bw_sq, 0))

    spec_diff = np.abs(np.diff(mean_spectrum))
    smoothness = 1.0 - (np.mean(spec_diff) / (np.max(spec_diff) + 1e-8))

    if T >= 2:
        flux = np.mean(np.sqrt(np.sum(np.diff(mel_linear, axis=0) ** 2, axis=1)))
    else:
        flux = 0.0

    if T >= 4:
        am_depth = np.std(frame_energy) / (np.mean(frame_energy) + 1e-8)
    else:
        am_depth = 0.0

    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10)) / (np.log(n_mels) + 1e-8)

    # LIGHT properties
    light = {
        'L_color': float(centroid),
        'L_brightness': float(rms),
        'L_saturation': float(1.0 - bandwidth),
        'L_texture': float(flatness),
        'L_warmth': float(low_ratio),
    }

    # TOUCH properties
    if T >= 4:
        onset_slope = np.max(np.diff(frame_energy[:T//4])) / (np.mean(frame_energy) + 1e-8)
        hardness = np.clip(onset_slope, 0, 10)
    else:
        hardness = 0.0
    touch = {
        'T_roughness': float(flatness),
        'T_hardness': float(hardness),
        'T_weight': float(low_ratio),
        'T_temperature': float(1.0 - centroid),
        'T_vibration': float(np.clip(am_depth, 0, 5)),
    }

    # GEOMETRY properties
    if n_mels >= 3:
        d2 = np.diff(mean_spectrum, n=2)
        curvature = np.mean(np.abs(d2)) / (np.mean(mean_spectrum) + 1e-8)
    else:
        curvature = 0.0
    pr = (spec_norm.sum() ** 2) / (np.sum(spec_norm ** 2) + 1e-10)
    compactness = 1.0 - (pr / n_mels)
    geometry = {
        'G_curvature': float(curvature),
        'G_compactness': float(compactness),
        'G_dimensionality': float(entropy),
        'G_smoothness': float(smoothness),
        'G_size': float(low_ratio),
    }

    # EMOTION properties
    mid_e = np.sum(mean_spectrum[third:2*third])
    balance = mid_e / (np.sum(mean_spectrum) + 1e-8)
    valence = 0.5 * centroid + 0.5 * balance
    arousal_raw = 0.4 * np.clip(rms / 1000, 0, 1) + 0.3 * np.clip(flux / 100, 0, 1) + 0.3 * am_depth
    loudness = np.log1p(np.mean(frame_energy))
    dominance = 0.4 * np.clip(loudness / 20, 0, 1) + 0.3 * low_ratio + 0.3 * flatness
    emotion = {
        'E_valence': float(valence),
        'E_arousal': float(arousal_raw),
        'E_dominance': float(dominance),
        'E_beauty': float(0.5 * smoothness + 0.5 * max(0, centroid)),
    }

    return {**light, **touch, **geometry, **emotion}


def load_encoder(checkpoint_path, device='cpu'):
    try:
        from mamba_ssm import Mamba2
        use_real = True
    except ImportError:
        use_real = False
    if use_real:
        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(40, 128)
                self.layers = nn.ModuleList([Mamba2(d_model=128, d_state=16, d_conv=4, expand=2) for _ in range(6)])
                self.norm = nn.LayerNorm(128)
            def forward(self, x):
                x = self.input_proj(x)
                for l in self.layers: x = x + l(x)
                return self.norm(x)
    else:
        class CB(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(128, 256, 4, padding=2)
                self.act = nn.SiLU()
                self.proj = nn.Linear(256, 128)
            def forward(self, x):
                y = self.conv(x.transpose(1,2)).transpose(1,2)[:,:x.size(1),:]
                return self.proj(self.act(y))
        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(40, 128)
                self.layers = nn.ModuleList([CB() for _ in range(6)])
                self.norm = nn.LayerNorm(128)
            def forward(self, x):
                x = self.input_proj(x)
                for l in self.layers: x = x + l(x)
                return self.norm(x)
    enc = Enc()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    es = ckpt['encoder'] if 'encoder' in ckpt else {}
    loaded = enc.load_state_dict(es, strict=False)
    n = len(es) - len(loaded.missing_keys)
    print(f"Encoder: {n}/{len(list(enc.state_dict().keys()))} params")
    enc.to(device)
    enc.train(False)
    return enc


def run_experiment(data_dir='data/training/mel/esc50', checkpoint_path='checkpoints/clap_distill_spread_a.pt'):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    mel_dir = Path(data_dir)
    meta = json.load(open('data/clap_meta.json'))
    categories = meta['categories']

    mel_files = sorted(mel_dir.glob('*.npy'), key=lambda p: int(p.stem))
    all_mels, all_props, valid_files = [], [], []
    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        if categories[idx] in ('ambient', 'mixed', 'music'): continue
        props = compute_all_properties(mel)
        all_mels.append(mel)
        all_props.append(props)
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    # Build property matrices per modality
    light_keys = [k for k in all_props[0] if k.startswith('L_')]
    touch_keys = [k for k in all_props[0] if k.startswith('T_')]
    geom_keys = [k for k in all_props[0] if k.startswith('G_')]
    emo_keys = [k for k in all_props[0] if k.startswith('E_')]
    all_keys = light_keys + touch_keys + geom_keys + emo_keys

    V_all = np.array([[p[k] for k in all_keys] for p in all_props])
    V_light = np.array([[p[k] for k in light_keys] for p in all_props])
    V_touch = np.array([[p[k] for k in touch_keys] for p in all_props])
    V_geom = np.array([[p[k] for k in geom_keys] for p in all_props])
    V_emo = np.array([[p[k] for k in emo_keys] for p in all_props])

    # Get embeddings
    encoder = load_encoder(checkpoint_path, device)
    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            emb = encoder(x).mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
    E = np.array(embeddings)

    results = {}

    # ═══ TEST 1: Shared subspace dimensionality ═══
    print("\n" + "="*60)
    print("TEST 1: How many dimensions span ALL modalities?")
    print("="*60)

    # Standardize all properties
    scaler_all = StandardScaler()
    V_std = scaler_all.fit_transform(V_all)

    # Replace NaN/inf with 0 for PCA
    V_std = np.nan_to_num(V_std, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA()
    pca.fit(V_std)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print(f"  Total cross-modal properties: {len(all_keys)}")
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_dims = np.searchsorted(cumvar, threshold) + 1
        print(f"  {threshold:.0%} variance explained by: {n_dims} dimensions")

    # KEY QUESTION: if modalities were independent, we'd need ~19 dims for 95%.
    # If unified, we need far fewer.
    n95 = np.searchsorted(cumvar, 0.95) + 1
    compression = len(all_keys) / n95
    print(f"\n  Compression ratio: {len(all_keys)}/{n95} = {compression:.1f}x")
    if compression > 3:
        print(f"  STRONG UNIFICATION: {compression:.1f}x compression means modalities share structure")
    elif compression > 2:
        print(f"  MODERATE UNIFICATION: some shared structure")
    else:
        print(f"  WEAK UNIFICATION: modalities are mostly independent")

    results['test1_pca'] = {
        'n_properties': len(all_keys),
        'dims_80': int(np.searchsorted(cumvar, 0.80) + 1),
        'dims_90': int(np.searchsorted(cumvar, 0.90) + 1),
        'dims_95': int(n95),
        'compression': float(compression),
        'explained_variance': [float(v) for v in pca.explained_variance_ratio_[:10]],
    }

    # ═══ TEST 2: Cross-modality prediction ═══
    print("\n" + "="*60)
    print("TEST 2: Can one modality predict another?")
    print("="*60)

    modalities = {
        'light': V_light, 'touch': V_touch, 'geometry': V_geom, 'emotion': V_emo
    }

    test2 = {}
    for src_name, src_V in modalities.items():
        src_std = StandardScaler().fit_transform(np.nan_to_num(src_V))
        for tgt_name, tgt_V in modalities.items():
            if src_name == tgt_name: continue
            tgt_std = StandardScaler().fit_transform(np.nan_to_num(tgt_V))
            # Predict each target column from source modality
            r2_scores = []
            for j in range(tgt_std.shape[1]):
                y = tgt_std[:, j]
                if np.std(y) < 1e-10: continue
                scores = cross_val_score(Ridge(alpha=1.0), src_std, y, cv=5, scoring='r2')
                r2_scores.append(scores.mean())
            mean_r2 = np.mean(r2_scores) if r2_scores else 0.0
            key = f"{src_name}->{tgt_name}"
            test2[key] = float(mean_r2)
            marker = "***" if mean_r2 > 0.5 else "**" if mean_r2 > 0.3 else "*" if mean_r2 > 0.1 else ""
            print(f"  {key:22s}: mean R-sq = {mean_r2:.3f} {marker}")

    results['test2_cross_predict'] = test2

    # ═══ TEST 3: Canonical correlations between modalities ═══
    print("\n" + "="*60)
    print("TEST 3: Canonical correlations (shared dimensions)")
    print("="*60)

    test3 = {}
    for i, (n1, v1) in enumerate(modalities.items()):
        for n2, v2 in list(modalities.items())[i+1:]:
            v1_clean = np.nan_to_num(StandardScaler().fit_transform(v1))
            v2_clean = np.nan_to_num(StandardScaler().fit_transform(v2))
            n_components = min(v1_clean.shape[1], v2_clean.shape[1])
            cca = CCA(n_components=n_components)
            try:
                X_c, Y_c = cca.fit_transform(v1_clean, v2_clean)
                # Canonical correlations
                cc = [float(np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1]) for k in range(n_components)]
                test3[f"{n1}<->{n2}"] = cc
                sig_dims = sum(1 for c in cc if abs(c) > 0.5)
                print(f"  {n1:8s} <-> {n2:8s}: {cc[0]:.3f}, {cc[1]:.3f}, ... ({sig_dims} dims with |r|>0.5)")
            except Exception as e:
                print(f"  {n1:8s} <-> {n2:8s}: CCA failed ({e})")
    results['test3_cca'] = test3

    # ═══ TEST 4: Embedding predicts unified space ═══
    print("\n" + "="*60)
    print("TEST 4: Does the encoder capture the unified space?")
    print("="*60)

    E_scaled = StandardScaler().fit_transform(E)
    # Project all properties into PCA space, predict from embeddings
    V_pca = pca.transform(V_std)

    test4 = {}
    for pc in range(min(10, V_pca.shape[1])):
        y = V_pca[:, pc]
        if np.std(y) < 1e-10: continue
        scores = cross_val_score(Ridge(alpha=1.0), E_scaled, y, cv=5, scoring='r2')
        r2 = scores.mean()
        var_explained = pca.explained_variance_ratio_[pc]
        test4[f"PC{pc+1}"] = {'r2': float(r2), 'var_pct': float(var_explained * 100)}
        marker = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  PC{pc+1:2d} ({var_explained*100:5.1f}% var): R-sq = {r2:.3f} {marker}")

    # How much of the unified cross-modal space does the encoder capture?
    total_captured = sum(
        test4[f"PC{i+1}"]['r2'] * test4[f"PC{i+1}"]['var_pct'] / 100
        for i in range(min(10, len(test4)))
        if test4[f"PC{i+1}"]['r2'] > 0
    )
    print(f"\n  Total cross-modal variance captured by encoder: {total_captured*100:.1f}%")
    results['test4_pca_prediction'] = test4
    results['test4_total_captured'] = float(total_captured)

    # ═══ TEST 5: The frequency basis ═══
    print("\n" + "="*60)
    print("TEST 5: What frequency features span the unified space?")
    print("="*60)

    # PCA loadings — which properties load on which components
    print("  Top loadings on PC1 (largest shared dimension):")
    pc1_loadings = list(zip(all_keys, pca.components_[0]))
    pc1_sorted = sorted(pc1_loadings, key=lambda x: abs(x[1]), reverse=True)
    for name, loading in pc1_sorted[:7]:
        modality = name[0]
        prop = name[2:]
        print(f"    {modality}:{prop:16s} = {loading:+.3f}")

    if V_pca.shape[1] >= 2:
        print("\n  Top loadings on PC2:")
        pc2_sorted = sorted(zip(all_keys, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)
        for name, loading in pc2_sorted[:7]:
            modality = name[0]
            prop = name[2:]
            print(f"    {modality}:{prop:16s} = {loading:+.3f}")

    results['test5_loadings'] = {
        'PC1': {n: float(l) for n, l in pc1_sorted[:7]},
        'PC2': {n: float(l) for n, l in (sorted(zip(all_keys, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:7] if V_pca.shape[1] >= 2 else [])}
    }

    # ═══ VERDICT ═══
    print(f"\n{'='*60}\nVERDICT: Is frequency a unified substrate?\n{'='*60}")

    # Evidence for unification
    evidence = []
    if compression > 3:
        evidence.append(f"PCA compression {compression:.1f}x (19 props -> {n95} dims)")
    cross_pred_strong = sum(1 for v in test2.values() if v > 0.3)
    if cross_pred_strong > 4:
        evidence.append(f"{cross_pred_strong}/12 cross-predictions R-sq > 0.3")
    if total_captured > 0.4:
        evidence.append(f"Encoder captures {total_captured*100:.0f}% of unified variance")

    if len(evidence) >= 2:
        verdict = "CONFIRMED - frequency IS a unified cross-modal substrate"
    elif len(evidence) >= 1:
        verdict = "PARTIAL - some unification, some modality-specific structure"
    else:
        verdict = "NOT CONFIRMED - modalities appear independent in this encoding"

    for e in evidence:
        print(f"  + {e}")
    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict

    with open('data/wc_unified_crossmodal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_unified_crossmodal_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
