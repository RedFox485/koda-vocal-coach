#!/usr/bin/env python3
"""
Wildcard Experiment: Light as Sound
====================================
Cross-modal dimensional mapping — can audio embeddings predict visual-analog properties?

Hypothesis: If sound and light share dimensional structure (both are wave phenomena),
then an encoder trained purely on audio should produce embeddings that linearly
predict visual-analog properties computed from spectrograms.

Visual-analog properties computed from mel spectrograms:
  - color (spectral centroid mapped to hue)
  - brightness (RMS energy)
  - saturation (spectral bandwidth / spread)
  - texture (spectral flatness — tonal vs noisy)
  - flicker (temporal modulation rate)
  - warmth (low-frequency energy ratio)
  - glow (onset energy / sustain energy)

If R² > 0.5 for linear prediction, these dimensions are already IN the embeddings.
If cross-modal correlations match real-world light correlations, the mapping is
not arbitrary — it reflects shared physical structure.

Philosophy: Sound couples to everything (photoacoustic, sonoluminescence,
thermoacoustic). This experiment asks whether a purely auditory encoder
discovers cross-modal structure without ever being trained on it.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ─── Visual property extraction from mel spectrograms ───

def compute_visual_properties(mel_frames):
    """
    Compute visual-analog properties from mel spectrogram frames.

    Args:
        mel_frames: numpy array (T, n_mels) — log-mel spectrogram (can be negative)

    Returns:
        dict of property_name -> scalar value
    """
    # Convert log-mel to linear power (positive values)
    mel_linear = np.exp(mel_frames)

    n_mels = mel_linear.shape[1]
    freq_bins = np.linspace(0, 1, n_mels)  # normalized frequency axis

    # Time-average spectrum
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8

    # 1. COLOR — spectral centroid (where is the "center of mass"?)
    #    Low centroid = red/warm, high centroid = blue/cool
    color = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)

    # 2. BRIGHTNESS — RMS energy (louder = brighter)
    brightness = np.sqrt(np.mean(mel_linear ** 2))

    # 3. SATURATION — spectral bandwidth (narrow = saturated, wide = pastel)
    bw_sq = np.sum(((freq_bins - color) ** 2) * mean_spectrum) / np.sum(mean_spectrum)
    saturation = 1.0 - np.sqrt(max(bw_sq, 0))
    saturation = np.clip(saturation, 0, 1)

    # 4. TEXTURE — spectral flatness (tonal = smooth, noisy = rough)
    log_spec = np.log(mean_spectrum + 1e-10)
    geo_mean = np.exp(np.mean(log_spec))
    arith_mean = np.mean(mean_spectrum)
    texture = geo_mean / (arith_mean + 1e-8)  # 0=tonal/smooth, 1=noisy/rough

    # 5. FLICKER — temporal modulation rate (how fast does energy change?)
    frame_energy = np.sum(mel_linear ** 2, axis=1)
    if len(frame_energy) > 2:
        flicker = np.mean(np.abs(np.diff(frame_energy))) / (np.mean(frame_energy) + 1e-8)
    else:
        flicker = 0.0

    # 6. WARMTH — low-frequency energy ratio
    low_cutoff = n_mels // 3
    warmth = np.sum(mean_spectrum[:low_cutoff]) / (np.sum(mean_spectrum) + 1e-8)

    # 7. GLOW — onset energy vs sustain (flash = high glow, steady = low)
    if mel_linear.shape[0] >= 4:
        onset_energy = np.mean(mel_linear[:mel_linear.shape[0]//4] ** 2)
        sustain_energy = np.mean(mel_linear[mel_linear.shape[0]//4:] ** 2) + 1e-8
        glow = onset_energy / sustain_energy
    else:
        glow = 1.0

    return {
        'color': float(color),
        'brightness': float(brightness),
        'saturation': float(saturation),
        'texture': float(texture),
        'flicker': float(flicker),
        'warmth': float(warmth),
        'glow': float(glow),
    }


# ─── Encoder loading ───

def load_encoder(checkpoint_path, device='cpu'):
    """Load encoder from checkpoint, with conv_stub fallback for MPS."""
    try:
        from mamba_ssm import Mamba2
        use_real_mamba = True
        print("Using real Mamba2")
    except ImportError:
        use_real_mamba = False
        print("Using conv_stub fallback (MPS/CPU)")

    if use_real_mamba:
        class MambaEncoder(nn.Module):
            def __init__(self, input_dim=40, d_model=128, n_layers=6):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.layers = nn.ModuleList([
                    Mamba2(d_model=d_model, d_state=16, d_conv=4, expand=2)
                    for _ in range(n_layers)
                ])
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                x = self.input_proj(x)
                for layer in self.layers:
                    x = x + layer(x)
                return self.norm(x)
    else:
        class ConvBlock(nn.Module):
            def __init__(self, d_model=128):
                super().__init__()
                self.conv = nn.Conv1d(d_model, d_model * 2, kernel_size=4, padding=2, groups=1)
                self.act = nn.SiLU()
                self.proj = nn.Linear(d_model * 2, d_model)

            def forward(self, x):
                y = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, :x.size(1), :]
                return self.proj(self.act(y))

        class MambaEncoder(nn.Module):
            def __init__(self, input_dim=40, d_model=128, n_layers=6):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.layers = nn.ModuleList([ConvBlock(d_model) for _ in range(n_layers)])
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                x = self.input_proj(x)
                for layer in self.layers:
                    x = x + layer(x)
                return self.norm(x)

    encoder = MambaEncoder()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load encoder weights — checkpoint stores encoder as nested dict
    if 'encoder' in ckpt:
        enc_state = ckpt['encoder']
    else:
        enc_state = {k.replace('encoder.', ''): v for k, v in ckpt.items() if k.startswith('encoder.')}

    loaded = encoder.load_state_dict(enc_state, strict=False)
    n_loaded = len(enc_state) - len(loaded.missing_keys)
    n_total = len(list(encoder.state_dict().keys()))
    print(f"Encoder loaded: {n_loaded}/{n_total} params")
    if n_loaded == 0 and not use_real_mamba:
        print("  WARNING: Conv stub can't load Mamba2 weights. Encoder embeddings will be random.")
        print("  CLAP comparison will still be valid.")

    encoder.to(device)
    encoder.train(False)
    return encoder


# ─── Main experiment ───

def run_experiment(data_dir='data/training/mel/esc50', checkpoint_path='checkpoints/clap_distill_spread_a.pt'):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    mel_dir = Path(data_dir)
    if not mel_dir.exists():
        print(f"ERROR: {mel_dir} not found.")
        return

    # Load category labels from clap_meta.json (numeric index -> category name)
    meta_path = Path('data/clap_meta.json')
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        return

    meta = json.load(open(meta_path))
    categories = meta['categories']  # list indexed by file number

    # Build category name -> ID mapping
    unique_cats_list = sorted(set(categories))
    cat_to_id = {c: i for i, c in enumerate(unique_cats_list)}
    cat_names = {i: c for c, i in cat_to_id.items()}

    # Load all mel spectrograms and compute visual properties
    mel_files = sorted(mel_dir.glob('*.npy'), key=lambda p: int(p.stem))
    print(f"Found {len(mel_files)} mel files")

    all_mels = []
    all_visual = []
    all_cats = []
    valid_files = []

    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4:
            continue

        idx = int(mf.stem)
        if idx >= len(categories):
            continue
        cat_name = categories[idx]
        if cat_name in ('ambient', 'mixed', 'music'):
            continue  # skip non-ESC50 categories
        cat = cat_to_id[cat_name]

        vis = compute_visual_properties(mel)
        all_mels.append(mel)
        all_visual.append(vis)
        all_cats.append(cat)
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    # Compute visual property matrix
    prop_names = ['color', 'brightness', 'saturation', 'texture', 'flicker', 'warmth', 'glow']
    V = np.array([[v[p] for p in prop_names] for v in all_visual])
    print(f"\nVisual property stats:")
    for i, name in enumerate(prop_names):
        print(f"  {name:12s}: mean={V[:, i].mean():.3f}, std={V[:, i].std():.3f}, "
              f"range=[{V[:, i].min():.3f}, {V[:, i].max():.3f}]")

    # Get encoder embeddings
    print(f"\nLoading encoder from {checkpoint_path}...")
    encoder = load_encoder(checkpoint_path, device)

    embeddings = []
    with torch.no_grad():
        for mel in all_mels:
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
            h = encoder(x)  # (1, T, 128)
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()  # (128,)
            embeddings.append(emb)

    E = np.array(embeddings)
    print(f"Embeddings shape: {E.shape}")

    # Also get CLAP embeddings for comparison
    clap_path = Path('data/clap_embeddings.npz')
    clap_embs = None
    if clap_path.exists():
        clap_data = np.load(clap_path)
        clap_E = clap_data['embeddings']
        # Match by index — CLAP embeddings indexed same as mel files
        matched_clap = []
        for vf in valid_files:
            idx = int(vf.replace('.npy', ''))
            if idx < len(clap_E):
                matched_clap.append(clap_E[idx])
            else:
                matched_clap.append(None)

        if all(m is not None for m in matched_clap):
            clap_embs = np.array(matched_clap)
            print(f"CLAP embeddings matched: {clap_embs.shape}")

    results = {}

    # ─── TEST 1: Linear prediction of visual properties from embeddings ───
    print("\n" + "="*60)
    print("TEST 1: Can embeddings linearly predict visual properties?")
    print("="*60)

    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E)

    test1_results = {}
    for i, name in enumerate(prop_names):
        y = V[:, i]
        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, E_scaled, y, cv=5, scoring='r2')
        r2 = scores.mean()
        test1_results[name] = float(r2)
        marker = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:12s}: R² = {r2:.3f} {marker}")

    results['test1_encoder_r2'] = test1_results

    # Compare with CLAP
    if clap_embs is not None:
        print("\n  CLAP comparison:")
        clap_scaled = StandardScaler().fit_transform(clap_embs)
        test1_clap = {}
        for i, name in enumerate(prop_names):
            y = V[:, i]
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, clap_scaled, y, cv=5, scoring='r2')
            r2 = scores.mean()
            test1_clap[name] = float(r2)
            print(f"  {name:12s}: R² = {r2:.3f}")
        results['test1_clap_r2'] = test1_clap

    # ─── TEST 2: Binary visual classification ───
    print("\n" + "="*60)
    print("TEST 2: Binary visual classification from embeddings")
    print("="*60)

    binary_tasks = {
        'warm_vs_cool': ('warmth', lambda x: x > np.median(x)),
        'dim_vs_bright': ('brightness', lambda x: x > np.median(x)),
        'smooth_vs_rough': ('texture', lambda x: x > np.median(x)),
        'steady_vs_flicker': ('flicker', lambda x: x > np.median(x)),
    }

    test2_results = {}
    for task_name, (prop, binarize) in binary_tasks.items():
        idx = prop_names.index(prop)
        y = binarize(V[:, idx]).astype(int)

        clf = LogisticRegression(max_iter=500, solver='saga', C=1.0)
        scores = cross_val_score(clf, E_scaled, y, cv=5, scoring='accuracy')
        acc = scores.mean()
        test2_results[task_name] = float(acc)
        marker = "***" if acc > 0.75 else "**" if acc > 0.65 else "*" if acc > 0.55 else ""
        print(f"  {task_name:20s}: {acc:.1%} {marker}")

    results['test2_binary_classification'] = test2_results

    # ─── TEST 3: Cross-modal correlation structure ───
    print("\n" + "="*60)
    print("TEST 3: Cross-modal correlation structure")
    print("="*60)
    print("Do visual properties correlate like real light properties?")

    # Expected correlations from real-world light:
    # - brightness and warmth: slightly positive (warm light tends to be perceived as brighter)
    # - color and warmth: strong negative (low freq = warm = "red", high freq = cool = "blue")
    # - texture and saturation: negative (noise = low saturation)

    corr_matrix = np.corrcoef(V.T)

    expected = {
        ('color', 'warmth'): 'negative',      # red=warm, blue=cool
        ('brightness', 'warmth'): 'positive',  # warm light perceived brighter
        ('texture', 'saturation'): 'negative', # noise = desaturated
        ('glow', 'flicker'): 'positive',       # flashes flicker
    }

    test3_results = {}
    matches = 0
    total = 0
    for (p1, p2), expected_sign in expected.items():
        i, j = prop_names.index(p1), prop_names.index(p2)
        r = corr_matrix[i, j]
        actual_sign = 'positive' if r > 0 else 'negative'
        match = actual_sign == expected_sign
        matches += match
        total += 1
        test3_results[f"{p1}_vs_{p2}"] = {
            'correlation': float(r),
            'expected': expected_sign,
            'actual': actual_sign,
            'match': match
        }
        print(f"  {p1:12s} vs {p2:12s}: r={r:+.3f} (expected {expected_sign}, "
              f"{'MATCH' if match else 'MISMATCH'})")

    print(f"\n  Cross-modal structure score: {matches}/{total} predictions match real-world light")
    results['test3_cross_modal'] = test3_results
    results['test3_structure_score'] = f"{matches}/{total}"

    # ─── TEST 4: Per-category visual profiles ───
    print("\n" + "="*60)
    print("TEST 4: What does each sound 'look like'?")
    print("="*60)

    cats_array = np.array(all_cats)
    unique_cats = sorted(set(all_cats))

    # Compute mean visual profile per category
    profiles = {}
    for cat in unique_cats:
        mask = cats_array == cat
        mean_vis = V[mask].mean(axis=0)
        profiles[cat_names[cat]] = {prop_names[i]: float(mean_vis[i]) for i in range(len(prop_names))}

    # Find extremes for each property
    print("\n  Visual extremes by sound category:")
    test4_results = {}
    for i, name in enumerate(prop_names):
        vals = {cat_names[c]: V[cats_array == c, i].mean() for c in unique_cats}
        sorted_cats = sorted(vals.items(), key=lambda x: x[1])
        lowest = sorted_cats[0]
        highest = sorted_cats[-1]
        test4_results[name] = {
            'lowest': {'category': lowest[0], 'value': float(lowest[1])},
            'highest': {'category': highest[0], 'value': float(highest[1])},
        }
        print(f"  {name:12s}: lowest={lowest[0]:20s} ({lowest[1]:.3f})  "
              f"highest={highest[0]:20s} ({highest[1]:.3f})")

    results['test4_category_profiles'] = test4_results
    results['test4_all_profiles'] = profiles

    # ─── VERDICT ───
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Count strong predictions (R² > 0.3)
    strong_r2 = sum(1 for v in test1_results.values() if v > 0.3)
    # Count good classifications (> 65%)
    good_class = sum(1 for v in test2_results.values() if v > 0.65)
    # Cross-modal match rate
    cm_matches = matches

    print(f"  Linear prediction:  {strong_r2}/7 properties with R² > 0.3")
    print(f"  Classification:     {good_class}/4 tasks > 65%")
    print(f"  Cross-modal match:  {cm_matches}/{total} correlations match real light")

    if strong_r2 >= 4 and cm_matches >= 3:
        verdict = "STRONG — sound and light share dimensional structure"
    elif strong_r2 >= 2 or cm_matches >= 2:
        verdict = "PARTIAL — some cross-modal dimensions present"
    else:
        verdict = "WEAK — mapping is mostly arbitrary"

    print(f"\n  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['summary'] = {
        'strong_r2_count': strong_r2,
        'good_class_count': good_class,
        'cross_modal_matches': f"{cm_matches}/{total}",
    }

    # Save results
    out_path = Path('data/wc_light_as_sound_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Light as Sound — cross-modal dimensional mapping')
    parser.add_argument('--data-dir', default='data/training/mel/esc50')
    parser.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = parser.parse_args()

    run_experiment(args.data_dir, args.checkpoint)
