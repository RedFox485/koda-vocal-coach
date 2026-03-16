#!/usr/bin/env python3
"""
Wildcard Experiment: Sound as Touch
====================================
Can audio embeddings predict tactile-analog properties?

If sound-as-light works (5/7 R-squared > 0.3), does sound-as-touch also work?
Sound and touch share mechanical wave physics - both are pressure phenomena.
We might expect STRONGER cross-modal mapping than light.

Tactile-analog properties computed from mel spectrograms:
  - roughness: spectral noisiness (high spectral flatness = rough texture)
  - hardness: attack sharpness (fast onset = hard surface)
  - weight: low-frequency dominance (heavy = bass-heavy)
  - temperature: spectral centroid (high freq = cool, low freq = warm)
  - elasticity: decay oscillation (bouncy = ringing decay)
  - stickiness: spectral persistence (sticky = sustained energy)
  - vibration: amplitude modulation depth (vibrating = tremolo)

Expected cross-modal correlations (from haptic research):
  - roughness and hardness: positive (rough things feel hard)
  - weight and temperature: negative (heavy = warm, light = cool, via thermal mass)
  - elasticity and hardness: negative (elastic = soft, rigid = hard)
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


def compute_tactile_properties(mel_frames):
    """Compute tactile-analog properties from log-mel spectrogram."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # 1. ROUGHNESS - spectral flatness (noisy = rough, tonal = smooth)
    geo_mean = np.exp(np.mean(np.log(mean_spectrum + 1e-10)))
    arith_mean = np.mean(mean_spectrum)
    roughness = geo_mean / (arith_mean + 1e-8)

    # 2. HARDNESS - attack sharpness (energy rise rate in first frames)
    onset_len = max(T // 4, 2)  # need at least 2 frames for diff
    if T >= 4 and onset_len >= 2:
        diffs = np.diff(frame_energy[:onset_len])
        onset_slope = np.max(diffs) / (np.mean(frame_energy) + 1e-8) if len(diffs) > 0 else 0.0
        hardness = np.clip(onset_slope, 0, 10)
    else:
        hardness = 0.0

    # 3. WEIGHT - low-frequency dominance ratio
    low_cut = n_mels // 3
    weight = np.sum(mean_spectrum[:low_cut]) / (np.sum(mean_spectrum) + 1e-8)

    # 4. TEMPERATURE - inverse spectral centroid (low freq = warm, high = cool)
    centroid = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)
    temperature = 1.0 - centroid  # high = warm, low = cool

    # 5. ELASTICITY - decay oscillation (autocorrelation of energy envelope)
    if T >= 10:
        energy_centered = frame_energy - frame_energy.mean()
        norm = np.sum(energy_centered ** 2)
        if norm > 1e-10:
            autocorr = np.correlate(energy_centered, energy_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (norm + 1e-8)
            crossings = np.where(np.diff(np.sign(autocorr)))[0]
            elasticity = len(crossings) / T
        else:
            elasticity = 0.0
    else:
        elasticity = 0.0

    # 6. STICKINESS - spectral persistence (how long energy sustains)
    if T >= 4:
        peak_idx = np.argmax(frame_energy)
        if peak_idx < T - 1:
            sustain = frame_energy[peak_idx:].sum() / (frame_energy.sum() + 1e-8)
        else:
            sustain = 0.5
        stickiness = sustain
    else:
        stickiness = 0.5

    # 7. VIBRATION - amplitude modulation depth
    if T >= 4:
        am_depth = np.std(frame_energy) / (np.mean(frame_energy) + 1e-8)
        vibration = np.clip(am_depth, 0, 5)
    else:
        vibration = 0.0

    return {
        'roughness': float(roughness),
        'hardness': float(hardness),
        'weight': float(weight),
        'temperature': float(temperature),
        'elasticity': float(elasticity),
        'stickiness': float(stickiness),
        'vibration': float(vibration),
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

    all_mels, all_tactile, all_cats, valid_files = [], [], [], []
    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        cn = categories[idx]
        if cn in ('ambient', 'mixed', 'music'): continue
        tac = compute_tactile_properties(mel)
        all_mels.append(mel)
        all_tactile.append(tac)
        all_cats.append(cat_to_id[cn])
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    prop_names = ['roughness', 'hardness', 'weight', 'temperature', 'elasticity', 'stickiness', 'vibration']
    V = np.array([[t[p] for p in prop_names] for t in all_tactile])

    print(f"\nTactile property stats:")
    for i, name in enumerate(prop_names):
        print(f"  {name:12s}: mean={V[:,i].mean():.3f}, std={V[:,i].std():.3f}")

    # Get encoder embeddings
    encoder = load_encoder(checkpoint_path, device)
    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            emb = encoder(x).mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
    E = np.array(embeddings)

    # CLAP embeddings
    clap_embs = None
    clap_path = Path('data/clap_embeddings.npz')
    if clap_path.exists():
        clap_data = np.load(clap_path)
        clap_E = clap_data['embeddings']
        matched = []
        for f in valid_files:
            idx = int(f.replace('.npy',''))
            if idx < len(clap_E):
                matched.append(clap_E[idx])
        if len(matched) == len(valid_files):
            clap_embs = np.array(matched)

    results = {}

    # TEST 1: Linear prediction
    print("\n" + "="*60)
    print("TEST 1: Can embeddings linearly predict tactile properties?")
    print("="*60)

    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)
    test1 = {}
    for i, name in enumerate(prop_names):
        y = V[:, i]
        if np.isnan(y).any() or np.std(y) < 1e-10:
            test1[name] = 0.0
            print(f"  {name:12s}: SKIPPED")
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
        test1_clap = {}
        for i, name in enumerate(prop_names):
            y = V[:, i]
            if np.isnan(y).any() or np.std(y) < 1e-10: continue
            scores = cross_val_score(Ridge(alpha=1.0), clap_scaled, y, cv=5, scoring='r2')
            test1_clap[name] = float(scores.mean())
            print(f"  {name:12s}: R-sq = {scores.mean():.3f}")
        results['test1_clap_r2'] = test1_clap

    # TEST 2: Binary tactile classification
    print("\n" + "="*60)
    print("TEST 2: Binary tactile classification")
    print("="*60)
    binary_tasks = {
        'smooth_vs_rough': ('roughness', lambda x: x > np.median(x)),
        'soft_vs_hard': ('hardness', lambda x: x > np.median(x)),
        'light_vs_heavy': ('weight', lambda x: x > np.median(x)),
        'cool_vs_warm': ('temperature', lambda x: x > np.median(x)),
        'rigid_vs_bouncy': ('elasticity', lambda x: x > np.median(x)),
    }
    test2 = {}
    for task, (prop, binarize) in binary_tasks.items():
        idx = prop_names.index(prop)
        y = binarize(V[:, idx]).astype(int)
        scores = cross_val_score(LogisticRegression(max_iter=1000, C=1.0), E_scaled, y, cv=5, scoring='accuracy')
        test2[task] = float(scores.mean())
        marker = "***" if scores.mean() > 0.75 else "**" if scores.mean() > 0.65 else ""
        print(f"  {task:20s}: {scores.mean():.1%} {marker}")
    results['test2_binary'] = test2

    # TEST 3: Cross-modal correlation structure
    print("\n" + "="*60)
    print("TEST 3: Tactile correlation structure")
    print("="*60)
    expected = {
        ('roughness', 'hardness'): 'positive',
        ('weight', 'temperature'): 'positive',
        ('elasticity', 'hardness'): 'negative',
        ('stickiness', 'roughness'): 'positive',
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
    print(f"\n  Tactile structure score: {matches}/{len(expected)}")
    results['test3_correlations'] = test3

    # TEST 4: Per-category tactile profiles
    print("\n" + "="*60)
    print("TEST 4: What does each sound 'feel' like?")
    print("="*60)
    cats_arr = np.array(all_cats)
    test4 = {}
    for i, name in enumerate(prop_names):
        vals = {cat_names[c]: V[cats_arr == c, i].mean() for c in sorted(set(all_cats))}
        s = sorted(vals.items(), key=lambda x: x[1])
        test4[name] = {'lowest': s[0][0], 'highest': s[-1][0]}
        print(f"  {name:12s}: lowest={s[0][0]:20s} ({s[0][1]:.3f})  highest={s[-1][0]:20s} ({s[-1][1]:.3f})")
    results['test4_profiles'] = test4

    # VERDICT
    strong_r2 = sum(1 for v in test1.values() if v > 0.3)
    good_class = sum(1 for v in test2.values() if v > 0.65)
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  Linear prediction: {strong_r2}/7 with R-sq > 0.3")
    print(f"  Classification:    {good_class}/5 > 65%")
    print(f"  Correlations:      {matches}/{len(expected)} match haptic research")

    if strong_r2 >= 4 and matches >= 3:
        verdict = "STRONG - sound encodes tactile dimensions"
    elif strong_r2 >= 2 or matches >= 2:
        verdict = "PARTIAL - some tactile dimensions present"
    else:
        verdict = "WEAK - tactile mapping is weak"
    print(f"  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['summary'] = {'strong_r2': strong_r2, 'good_class': good_class, 'correlations': f"{matches}/{len(expected)}"}

    with open('data/wc_sound_as_touch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_sound_as_touch_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
