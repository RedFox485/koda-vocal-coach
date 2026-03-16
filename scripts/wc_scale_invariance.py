#!/usr/bin/env python3
"""
Wildcard: Scale Invariance of Cross-Modal Mapping
===================================================
The deepest test of "frequency is everything": if the cross-modal mappings
are universal, they should be SELF-SIMILAR across frequency scales.

This means: the relationship between spectral features and perceived
properties should be the SAME whether you're looking at 100Hz-1kHz,
1kHz-10kHz, or any other band. This is the acoustic equivalent of
fractal self-similarity — the same physics at every scale.

Tests:
1. Do cross-modal correlations hold within each octave independently?
2. Is the mapping function (spectral shape -> perceived property) invariant?
3. Harmonic ratio invariance: do musical intervals map to the same
   cross-modal properties regardless of absolute pitch?
4. Information content per octave: is cross-modal information uniformly
   distributed or concentrated at specific scales?
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


def compute_band_properties(spectrum_band, freq_bins):
    """Compute normalized cross-modal properties for an arbitrary frequency band."""
    s = spectrum_band + 1e-8
    norm = s / (s.sum() + 1e-8)
    n = len(s)

    # Centroid (color/temperature analog)
    centroid = np.sum(freq_bins * s) / np.sum(s)

    # Flatness (roughness/texture analog)
    geo = np.exp(np.mean(np.log(s + 1e-10)))
    flatness = geo / (np.mean(s) + 1e-8)

    # Bandwidth (saturation analog)
    bw = np.sqrt(max(np.sum(((freq_bins - centroid)**2) * s) / np.sum(s), 0))

    # Energy concentration (compactness analog)
    pr = (norm.sum()**2) / (np.sum(norm**2) + 1e-10)
    concentration = 1.0 - (pr / max(n, 1))

    # Smoothness (beauty/roundness analog)
    if n >= 2:
        d = np.abs(np.diff(s))
        smoothness = 1.0 - (np.mean(d) / (np.max(d) + 1e-8))
    else:
        smoothness = 1.0

    # Entropy (dimensionality analog)
    entropy = -np.sum(norm * np.log(norm + 1e-10)) / (np.log(max(n, 2)) + 1e-8)

    # Peak count (topology analog)
    peaks = 0
    if n >= 3:
        thresh = np.mean(s) * 0.3
        for i in range(1, n-1):
            if s[i] > s[i-1] and s[i] > s[i+1] and s[i] > thresh:
                peaks += 1
    topology = peaks / max(n/4, 1)

    return {
        'centroid': float(centroid),
        'flatness': float(flatness),
        'bandwidth': float(bw),
        'concentration': float(concentration),
        'smoothness': float(smoothness),
        'entropy': float(entropy),
        'topology': float(topology),
    }


def run_experiment(data_dir='data/training/mel/esc50'):
    print("Scale Invariance Test")
    print("="*60)

    mel_dir = Path(data_dir)
    meta = json.load(open('data/clap_meta.json'))
    categories = meta['categories']

    mel_files = sorted(mel_dir.glob('*.npy'), key=lambda p: int(p.stem))
    all_spectra = []
    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        if categories[idx] in ('ambient', 'mixed', 'music'): continue
        mel_linear = np.exp(mel)
        all_spectra.append(mel_linear.mean(axis=0))

    S = np.array(all_spectra)
    N, n_mels = S.shape
    print(f"Samples: {N}, Mel bands: {n_mels}")

    # Define octave-like bands (each ~13 bands for 40-band mel)
    band_defs = {
        'low': (0, 13),      # ~0-1kHz
        'mid': (13, 27),     # ~1-4kHz
        'high': (27, 40),    # ~4-22kHz
    }

    results = {}

    # ═══ TEST 1: Same properties emerge at each scale ═══
    print("\n" + "="*60)
    print("TEST 1: Do the same cross-modal properties emerge at each octave?")
    print("="*60)

    band_properties = {}
    for band_name, (lo, hi) in band_defs.items():
        band_data = S[:, lo:hi]
        freq_ax = np.linspace(0, 1, hi - lo)
        props_list = [compute_band_properties(band_data[i], freq_ax) for i in range(N)]
        prop_names = list(props_list[0].keys())
        band_properties[band_name] = np.array([[p[k] for k in prop_names] for p in props_list])

    # Compare property distributions across bands
    print("\n  Property distribution similarity (KL divergence):")
    test1 = {}
    for pi, prop in enumerate(prop_names):
        band_vals = {}
        for band_name in band_defs:
            v = band_properties[band_name][:, pi]
            band_vals[band_name] = v

        # Correlation between bands for this property
        corrs = []
        for b1 in band_defs:
            for b2 in band_defs:
                if b1 >= b2: continue
                r = np.corrcoef(band_vals[b1], band_vals[b2])[0, 1]
                corrs.append(r)
        mean_corr = np.mean(corrs) if corrs else 0
        test1[prop] = float(mean_corr)
        marker = "***" if mean_corr > 0.7 else "**" if mean_corr > 0.4 else "*" if mean_corr > 0.1 else ""
        print(f"  {prop:16s}: cross-band r = {mean_corr:.3f} {marker}")

    invariant_props = sum(1 for v in test1.values() if v > 0.5)
    print(f"\n  {invariant_props}/{len(prop_names)} properties are scale-invariant (r > 0.5)")
    results['test1_cross_band_corr'] = test1

    # ═══ TEST 2: Structure preservation across scales ═══
    print("\n" + "="*60)
    print("TEST 2: Is the inter-property structure preserved across octaves?")
    print("="*60)

    # Compute correlation matrix WITHIN each band
    band_corr_matrices = {}
    for band_name in band_defs:
        V = band_properties[band_name]
        V_clean = np.nan_to_num(V)
        band_corr_matrices[band_name] = np.corrcoef(V_clean.T)

    # Compare correlation structures
    test2 = {}
    band_list = list(band_defs.keys())
    for i in range(len(band_list)):
        for j in range(i+1, len(band_list)):
            b1, b2 = band_list[i], band_list[j]
            m1 = band_corr_matrices[b1]
            m2 = band_corr_matrices[b2]
            # Flatten upper triangle
            triu_idx = np.triu_indices(len(prop_names), k=1)
            v1 = m1[triu_idx]
            v2 = m2[triu_idx]
            r = np.corrcoef(v1, v2)[0, 1]
            test2[f"{b1}<->{b2}"] = float(r)
            print(f"  {b1:4s} <-> {b2:4s} structure similarity: r = {r:.3f}")

    mean_struct = np.mean(list(test2.values()))
    print(f"\n  Mean structure similarity: {mean_struct:.3f}")
    if mean_struct > 0.7:
        print("  STRONGLY SELF-SIMILAR: same relationships at every scale")
    elif mean_struct > 0.4:
        print("  MODERATELY SELF-SIMILAR: core relationships preserved")
    else:
        print("  SCALE-DEPENDENT: different relationships at different scales")
    results['test2_structure_similarity'] = test2

    # ═══ TEST 3: Information content per octave ═══
    print("\n" + "="*60)
    print("TEST 3: Information distribution across octaves")
    print("="*60)

    # Compute full-band properties as targets
    full_freq = np.linspace(0, 1, n_mels)
    full_props = [compute_band_properties(S[i], full_freq) for i in range(N)]
    Y_full = np.array([[p[k] for k in prop_names] for p in full_props])

    test3 = {}
    for band_name, (lo, hi) in band_defs.items():
        band_features = StandardScaler().fit_transform(S[:, lo:hi])
        band_r2s = []
        for pi, prop in enumerate(prop_names):
            y = Y_full[:, pi]
            if np.std(y) < 1e-10: continue
            scores = cross_val_score(Ridge(alpha=1.0), band_features, y, cv=5, scoring='r2')
            band_r2s.append(scores.mean())
        mean_r2 = np.mean(band_r2s)
        test3[band_name] = {'mean_r2': float(mean_r2), 'n_strong': sum(1 for v in band_r2s if v > 0.3)}
        print(f"  {band_name:4s} ({lo:2d}-{hi:2d}): mean R-sq = {mean_r2:.3f}, "
              f"{test3[band_name]['n_strong']}/{len(band_r2s)} > 0.3")

    # Is information uniform?
    r2s = [v['mean_r2'] for v in test3.values()]
    info_range = max(r2s) - min(r2s)
    print(f"\n  Information range across octaves: {info_range:.3f}")
    if info_range < 0.1:
        print("  UNIFORMLY DISTRIBUTED: every octave carries full cross-modal info")
    elif info_range < 0.2:
        print("  MOSTLY UNIFORM: slight concentration at one scale")
    else:
        print("  CONCENTRATED: some octaves carry much more cross-modal info")
    results['test3_info_distribution'] = test3

    # ═══ TEST 4: Ratio invariance (musical intervals) ═══
    print("\n" + "="*60)
    print("TEST 4: Harmonic ratio invariance")
    print("="*60)
    print("Do frequency RATIOS (not absolutes) determine cross-modal mapping?")

    # For each sample, compute ratio features: each pair of bands
    # If ratios matter more than absolutes, ratio features should predict better
    ratio_features = []
    for i in range(N):
        s = S[i] + 1e-8
        ratios = []
        for b in range(0, n_mels - 1, 2):
            if b + 1 < n_mels:
                ratios.append(s[b+1] / s[b])  # adjacent band ratio
            if b + 2 < n_mels:
                ratios.append(s[b+2] / s[b])  # octave-like ratio
        ratio_features.append(ratios)

    min_len = min(len(r) for r in ratio_features)
    R = np.array([r[:min_len] for r in ratio_features])
    R_scaled = StandardScaler().fit_transform(R)

    # Compare: absolute spectrum vs ratio features
    abs_r2s = []
    ratio_r2s = []
    S_scaled = StandardScaler().fit_transform(S)

    test4 = {}
    for pi, prop in enumerate(prop_names):
        y = Y_full[:, pi]
        if np.std(y) < 1e-10: continue
        abs_scores = cross_val_score(Ridge(alpha=1.0), S_scaled, y, cv=5, scoring='r2')
        ratio_scores = cross_val_score(Ridge(alpha=1.0), R_scaled, y, cv=5, scoring='r2')
        abs_r2s.append(abs_scores.mean())
        ratio_r2s.append(ratio_scores.mean())
        test4[prop] = {'absolute': float(abs_scores.mean()), 'ratio': float(ratio_scores.mean())}

    mean_abs = np.mean(abs_r2s)
    mean_ratio = np.mean(ratio_r2s)
    print(f"\n  Absolute features:  mean R-sq = {mean_abs:.3f}")
    print(f"  Ratio features:     mean R-sq = {mean_ratio:.3f}")
    print(f"  Delta:              {mean_ratio - mean_abs:+.3f}")

    if mean_ratio > mean_abs + 0.02:
        print("\n  RATIOS WIN: frequency relationships matter more than absolute frequency")
        print("  This supports 'frequency as fabric' — it's the PATTERN, not the position")
    elif abs(mean_ratio - mean_abs) < 0.02:
        print("\n  TIE: both carry similar information")
    else:
        print("\n  ABSOLUTES WIN: position in spectrum matters more than relationships")
    results['test4_ratio_invariance'] = test4
    results['test4_summary'] = {'mean_absolute': float(mean_abs), 'mean_ratio': float(mean_ratio)}

    # ═══ GRAND VERDICT ═══
    print(f"\n{'='*60}")
    print("GRAND VERDICT: Is cross-modal mapping scale-invariant?")
    print("="*60)

    evidence = []
    if invariant_props >= 4:
        evidence.append(f"{invariant_props}/7 properties are scale-invariant (r>0.5)")
    if mean_struct > 0.5:
        evidence.append(f"Inter-property structure preserved across octaves (r={mean_struct:.2f})")
    if info_range < 0.15:
        evidence.append(f"Information uniformly distributed (range={info_range:.3f})")
    if mean_ratio >= mean_abs - 0.02:
        evidence.append("Ratios carry at least as much info as absolutes")

    for e in evidence:
        print(f"  + {e}")

    if len(evidence) >= 3:
        verdict = "YES - cross-modal mapping is scale-invariant (universal frequency principle)"
    elif len(evidence) >= 2:
        verdict = "MOSTLY - core structure is scale-invariant with some scale-specific features"
    else:
        verdict = "NO - significant scale dependence exists"

    print(f"\n  VERDICT: {verdict}")
    results['grand_verdict'] = verdict

    with open('data/wc_scale_invariance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_scale_invariance_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    args = p.parse_args()
    run_experiment(args.data_dir)
