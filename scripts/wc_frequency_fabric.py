#!/usr/bin/env python3
"""
Wildcard: Frequency as Fabric
===============================
The most fundamental test: if frequency is the fabric of reality, then
RAW spectral features alone (no derived properties) should predict ALL
cross-modal properties simultaneously.

We test: can 40 mel-frequency bands directly predict light, touch, geometry,
and emotion? No hand-crafted feature engineering — just the spectrum.

Additionally: do the SAME spectral bands matter for ALL modalities?
If yes, frequency is truly unified. If different bands matter for different
modalities, then the unification is at a higher level of abstraction.

Also tests scale invariance: do low/mid/high frequency bands carry the
same cross-modal information structure?
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


def compute_all_targets(mel_frames):
    """Compute target properties across all modalities."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    frame_energy = np.sum(mel_linear ** 2, axis=1)
    centroid = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)
    geo_mean = np.exp(np.mean(np.log(mean_spectrum + 1e-10)))
    flatness = geo_mean / (np.mean(mean_spectrum) + 1e-8)
    rms = np.sqrt(np.mean(mel_linear ** 2))
    third = n_mels // 3
    low_ratio = np.sum(mean_spectrum[:third]) / (np.sum(mean_spectrum) + 1e-8)
    bw = np.sqrt(max(np.sum(((freq_bins - centroid) ** 2) * mean_spectrum) / np.sum(mean_spectrum), 0))
    spec_diff = np.abs(np.diff(mean_spectrum))
    smoothness = 1.0 - (np.mean(spec_diff) / (np.max(spec_diff) + 1e-8))

    if T >= 2:
        flux = np.mean(np.sqrt(np.sum(np.diff(mel_linear, axis=0) ** 2, axis=1)))
    else:
        flux = 0.0
    if T >= 4:
        am = np.std(frame_energy) / (np.mean(frame_energy) + 1e-8)
        onset = np.max(np.diff(frame_energy[:T//4])) / (np.mean(frame_energy) + 1e-8)
    else:
        am, onset = 0.0, 0.0

    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10)) / (np.log(n_mels) + 1e-8)
    if n_mels >= 3:
        curv = np.mean(np.abs(np.diff(mean_spectrum, n=2))) / (np.mean(mean_spectrum) + 1e-8)
    else:
        curv = 0.0
    pr = (spec_norm.sum()**2) / (np.sum(spec_norm**2) + 1e-10)
    compact = 1.0 - (pr / n_mels)
    mid_e = np.sum(mean_spectrum[third:2*third])
    balance = mid_e / (np.sum(mean_spectrum) + 1e-8)
    loudness = np.log1p(np.mean(frame_energy))

    return {
        # Light
        'color': float(centroid), 'brightness': float(rms),
        'saturation': float(1-bw), 'warmth': float(low_ratio),
        # Touch
        'roughness': float(flatness), 'hardness': float(np.clip(onset,0,10)),
        'weight': float(low_ratio), 'temperature': float(1-centroid),
        'vibration': float(np.clip(am,0,5)),
        # Geometry
        'curvature': float(curv), 'compactness': float(compact),
        'dimensionality': float(entropy), 'shape_smooth': float(smoothness),
        # Emotion
        'valence': float(0.5*centroid + 0.5*balance),
        'arousal': float(0.4*np.clip(rms/1000,0,1) + 0.3*np.clip(flux/100,0,1) + 0.3*am),
        'dominance': float(0.4*np.clip(loudness/20,0,1) + 0.3*low_ratio + 0.3*flatness),
        'beauty': float(0.5*smoothness + 0.5*max(0,centroid)),
    }


def run_experiment(data_dir='data/training/mel/esc50', checkpoint_path='checkpoints/clap_distill_spread_a.pt'):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    mel_dir = Path(data_dir)
    meta = json.load(open('data/clap_meta.json'))
    categories = meta['categories']

    mel_files = sorted(mel_dir.glob('*.npy'), key=lambda p: int(p.stem))
    all_raw_spectra = []  # raw mean mel spectra
    all_targets = []
    all_mels = []

    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        if categories[idx] in ('ambient', 'mixed', 'music'): continue

        mel_linear = np.exp(mel)
        raw_spectrum = mel_linear.mean(axis=0)  # 40-dim raw frequency features
        targets = compute_all_targets(mel)

        all_raw_spectra.append(raw_spectrum)
        all_targets.append(targets)
        all_mels.append(mel)

    print(f"Valid samples: {len(all_mels)}")

    S = np.array(all_raw_spectra)  # (N, 40) raw spectra
    target_names = list(all_targets[0].keys())
    Y = np.array([[t[k] for k in target_names] for t in all_targets])

    # Modality grouping
    modality_map = {
        'light': ['color', 'brightness', 'saturation', 'warmth'],
        'touch': ['roughness', 'hardness', 'weight', 'temperature', 'vibration'],
        'geometry': ['curvature', 'compactness', 'dimensionality', 'shape_smooth'],
        'emotion': ['valence', 'arousal', 'dominance', 'beauty'],
    }

    results = {}

    # ═══ TEST 1: Raw spectrum predicts all modalities ═══
    print("\n" + "="*60)
    print("TEST 1: Can 40 raw mel bands predict ALL cross-modal properties?")
    print("="*60)

    S_scaled = StandardScaler().fit_transform(S)
    test1 = {}
    for mod_name, props in modality_map.items():
        print(f"\n  [{mod_name.upper()}]")
        mod_r2s = []
        for prop in props:
            idx = target_names.index(prop)
            y = Y[:, idx]
            if np.isnan(y).any() or np.std(y) < 1e-10:
                print(f"    {prop:16s}: SKIPPED")
                continue
            scores = cross_val_score(Ridge(alpha=1.0), S_scaled, y, cv=5, scoring='r2')
            r2 = scores.mean()
            mod_r2s.append(r2)
            test1[prop] = float(r2)
            marker = "***" if r2 > 0.8 else "**" if r2 > 0.5 else "*" if r2 > 0.3 else ""
            print(f"    {prop:16s}: R-sq = {r2:.3f} {marker}")
        if mod_r2s:
            print(f"    mean: {np.mean(mod_r2s):.3f}")

    results['test1_raw_spectrum_r2'] = test1
    all_r2 = [v for v in test1.values()]
    strong = sum(1 for v in all_r2 if v > 0.3)
    print(f"\n  Summary: {strong}/{len(all_r2)} properties with R-sq > 0.3 from raw spectrum alone")

    # ═══ TEST 2: Which bands matter for which modalities? ═══
    print("\n" + "="*60)
    print("TEST 2: Do the same bands matter for all modalities?")
    print("="*60)

    # Fit Ridge for each modality, extract coefficient magnitudes
    band_importance = {}
    for mod_name, props in modality_map.items():
        mod_coefs = np.zeros(40)
        n_props = 0
        for prop in props:
            idx = target_names.index(prop)
            y = Y[:, idx]
            if np.isnan(y).any() or np.std(y) < 1e-10: continue
            ridge = Ridge(alpha=1.0).fit(S_scaled, y)
            mod_coefs += np.abs(ridge.coef_)
            n_props += 1
        if n_props > 0:
            mod_coefs /= n_props
        band_importance[mod_name] = mod_coefs

    # Compare band importance profiles across modalities
    print("\n  Band importance correlation between modalities:")
    mod_names = list(band_importance.keys())
    test2_corr = {}
    for i in range(len(mod_names)):
        for j in range(i+1, len(mod_names)):
            n1, n2 = mod_names[i], mod_names[j]
            r = np.corrcoef(band_importance[n1], band_importance[n2])[0, 1]
            test2_corr[f"{n1}<->{n2}"] = float(r)
            print(f"    {n1:8s} <-> {n2:8s}: r = {r:.3f}")

    mean_band_corr = np.mean(list(test2_corr.values()))
    print(f"\n  Mean band importance correlation: {mean_band_corr:.3f}")
    if mean_band_corr > 0.7:
        print("  SAME BANDS matter for all modalities — frequency is truly unified")
    elif mean_band_corr > 0.4:
        print("  OVERLAPPING bands — partial unity, some modality-specific")
    else:
        print("  DIFFERENT bands — modalities use different frequency information")

    results['test2_band_correlations'] = test2_corr
    results['test2_mean_corr'] = float(mean_band_corr)

    # Top bands per modality
    print("\n  Top 5 bands per modality:")
    for mod_name, coefs in band_importance.items():
        top5 = np.argsort(coefs)[-5:][::-1]
        print(f"    {mod_name:8s}: bands {list(top5)} (importance: {coefs[top5[0]]:.3f}..{coefs[top5[-1]]:.3f})")
    results['test2_top_bands'] = {n: list(map(int, np.argsort(c)[-5:][::-1])) for n, c in band_importance.items()}

    # ═══ TEST 3: Scale invariance ═══
    print("\n" + "="*60)
    print("TEST 3: Scale invariance — same structure at different octaves?")
    print("="*60)

    # Split spectrum into 3 octave-like bands
    bands = {
        'low (0-13)': S_scaled[:, :14],
        'mid (14-26)': S_scaled[:, 14:27],
        'high (27-39)': S_scaled[:, 27:],
    }

    test3 = {}
    for band_name, band_features in bands.items():
        band_r2s = []
        for prop in target_names:
            idx = target_names.index(prop)
            y = Y[:, idx]
            if np.isnan(y).any() or np.std(y) < 1e-10: continue
            scores = cross_val_score(Ridge(alpha=1.0), band_features, y, cv=5, scoring='r2')
            band_r2s.append(scores.mean())

        mean_r2 = np.mean(band_r2s) if band_r2s else 0
        strong_ct = sum(1 for v in band_r2s if v > 0.3)
        test3[band_name] = {'mean_r2': float(mean_r2), 'strong_count': strong_ct}
        print(f"  {band_name}: mean R-sq = {mean_r2:.3f}, {strong_ct}/{len(band_r2s)} > 0.3")

    results['test3_scale_invariance'] = test3

    # Check if all bands carry similar info (scale invariant) or if one dominates
    r2_values = [v['mean_r2'] for v in test3.values()]
    r2_range = max(r2_values) - min(r2_values)
    if r2_range < 0.1:
        print("\n  SCALE INVARIANT: all octaves carry similar cross-modal information")
    elif r2_range < 0.2:
        print("\n  PARTIALLY INVARIANT: some octave-dependence")
    else:
        print("\n  SCALE DEPENDENT: different octaves carry different information")
    results['test3_scale_range'] = float(r2_range)

    # ═══ TEST 4: Raw spectrum vs encoder embeddings ═══
    print("\n" + "="*60)
    print("TEST 4: Raw spectrum vs encoder — what does learning add?")
    print("="*60)

    # Load encoder embeddings
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
    enc.load_state_dict(es, strict=False)
    enc.to(device)
    enc.train(False)

    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            emb = enc(x).mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
    E = np.array(embeddings)
    E_scaled = StandardScaler().fit_transform(E)

    test4 = {}
    print(f"\n  {'Property':16s} {'Raw Spectrum':>14s} {'Encoder':>10s} {'Delta':>8s}")
    print(f"  {'-'*50}")
    for prop in target_names:
        idx = target_names.index(prop)
        y = Y[:, idx]
        if np.isnan(y).any() or np.std(y) < 1e-10: continue

        # Raw spectrum
        raw_scores = cross_val_score(Ridge(alpha=1.0), S_scaled, y, cv=5, scoring='r2')
        # Encoder
        enc_scores = cross_val_score(Ridge(alpha=1.0), E_scaled, y, cv=5, scoring='r2')

        raw_r2 = raw_scores.mean()
        enc_r2 = enc_scores.mean()
        delta = enc_r2 - raw_r2
        test4[prop] = {'raw': float(raw_r2), 'encoder': float(enc_r2), 'delta': float(delta)}
        arrow = "+" if delta > 0.01 else "-" if delta < -0.01 else "="
        print(f"  {prop:16s} {raw_r2:14.3f} {enc_r2:10.3f} {arrow}{abs(delta):7.3f}")

    # Summary
    deltas = [v['delta'] for v in test4.values()]
    mean_delta = np.mean(deltas)
    enc_wins = sum(1 for d in deltas if d > 0.01)
    raw_wins = sum(1 for d in deltas if d < -0.01)
    print(f"\n  Mean delta: {mean_delta:+.3f}")
    print(f"  Encoder wins: {enc_wins}, Raw wins: {raw_wins}, Ties: {len(deltas)-enc_wins-raw_wins}")

    if mean_delta > 0.05:
        print("  Encoder adds substantial cross-modal information beyond raw spectrum")
    elif mean_delta > 0:
        print("  Encoder adds marginal information — most cross-modal structure is in raw frequency")
    else:
        print("  Raw spectrum is sufficient — cross-modal structure IS frequency structure")

    results['test4_raw_vs_encoder'] = test4

    # ═══ GRAND VERDICT ═══
    print(f"\n{'='*60}")
    print("GRAND VERDICT: Is frequency the fabric of reality?")
    print("="*60)

    evidence = []
    raw_strong = sum(1 for v in test1.values() if v > 0.3)
    if raw_strong >= 12:
        evidence.append(f"Raw spectrum predicts {raw_strong}/{len(test1)} properties (R-sq>0.3)")
    if mean_band_corr > 0.5:
        evidence.append(f"Same bands matter across modalities (mean r={mean_band_corr:.2f})")
    if r2_range < 0.15:
        evidence.append(f"Scale invariant: all octaves carry cross-modal info (range={r2_range:.3f})")
    if mean_delta < 0.02:
        evidence.append("Cross-modal structure is IN the frequency, not learned on top of it")

    for e in evidence:
        print(f"  + {e}")

    if len(evidence) >= 3:
        verdict = "YES - frequency IS the unified substrate of cross-modal perception"
    elif len(evidence) >= 2:
        verdict = "MOSTLY - frequency carries most cross-modal structure"
    elif len(evidence) >= 1:
        verdict = "PARTIALLY - some support, needs more investigation"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  GRAND VERDICT: {verdict}")
    results['grand_verdict'] = verdict
    results['evidence_count'] = len(evidence)

    with open('data/wc_frequency_fabric_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_frequency_fabric_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
