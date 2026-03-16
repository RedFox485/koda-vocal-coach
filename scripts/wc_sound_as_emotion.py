#!/usr/bin/env python3
"""
Wildcard Experiment: Sound as Emotion
======================================
Can audio embeddings predict dimensional affect without emotion training?

The dimensional model of emotion (Russell 1980) maps feelings to:
  - valence: positive vs negative
  - arousal: calm vs excited
  - dominance: submissive vs dominant

Psychoacoustic research shows reliable mappings:
  - High pitch + fast tempo = high arousal
  - Major mode + bright timbre = positive valence
  - Loud + low + harsh = dominant

We add two more dimensions from aesthetics research:
  - tension: predictable vs unpredictable (information-theoretic)
  - beauty: harmonic richness + spectral balance (consonance)

The question: does an encoder trained only on next-frame prediction + CLAP
distillation discover these affect dimensions without any emotion labels?
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


def compute_emotion_properties(mel_frames):
    """Compute emotion-analog properties from log-mel spectrogram."""
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # Spectral centroid
    centroid = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)

    # 1. VALENCE - brightness + harmonic balance (bright + balanced = positive)
    # High spectral centroid = bright. Even harmonic distribution = pleasant.
    brightness = centroid
    # Spectral balance: ratio of mid-band to extremes
    third = n_mels // 3
    low_e = np.sum(mean_spectrum[:third])
    mid_e = np.sum(mean_spectrum[third:2*third])
    high_e = np.sum(mean_spectrum[2*third:])
    total = low_e + mid_e + high_e + 1e-8
    balance = mid_e / total  # more mid = more balanced = more pleasant
    valence = 0.5 * brightness + 0.5 * balance

    # 2. AROUSAL - energy + tempo + spectral flux
    rms = np.sqrt(np.mean(mel_linear ** 2))
    if T >= 2:
        flux = np.mean(np.sqrt(np.sum(np.diff(mel_linear, axis=0) ** 2, axis=1)))
    else:
        flux = 0.0
    # Tempo proxy: zero-crossing rate of energy envelope
    if T >= 4:
        e_centered = frame_energy - frame_energy.mean()
        zcr = np.sum(np.abs(np.diff(np.sign(e_centered)))) / (2 * T)
    else:
        zcr = 0.0
    arousal = 0.4 * np.clip(rms / 1000, 0, 1) + 0.3 * np.clip(flux / 100, 0, 1) + 0.3 * zcr

    # 3. DOMINANCE - loudness + low frequency + harshness
    loudness = np.log1p(np.mean(frame_energy))
    low_ratio = np.sum(mean_spectrum[:third]) / total
    # Harshness: spectral flatness (noise-like = harsh)
    geo_mean = np.exp(np.mean(np.log(mean_spectrum + 1e-10)))
    harshness = geo_mean / (np.mean(mean_spectrum) + 1e-8)
    dominance = 0.4 * np.clip(loudness / 20, 0, 1) + 0.3 * low_ratio + 0.3 * harshness

    # 4. TENSION - unpredictability (entropy of spectral change)
    if T >= 3:
        diffs = np.diff(mel_linear, axis=0)
        frame_variances = np.var(diffs, axis=1)
        if np.std(frame_variances) > 1e-10:
            # Normalize to probability-like
            fv_norm = frame_variances / (frame_variances.sum() + 1e-8)
            tension = -np.sum(fv_norm * np.log(fv_norm + 1e-10)) / np.log(len(fv_norm) + 1e-10)
        else:
            tension = 0.0
    else:
        tension = 0.0

    # 5. BEAUTY - harmonic richness + spectral smoothness
    # Smooth spectrum = consonant = beautiful
    spec_diff = np.abs(np.diff(mean_spectrum))
    smoothness = 1.0 - (np.mean(spec_diff) / (np.max(spec_diff) + 1e-8))
    # Harmonic richness: autocorrelation strength in spectrum
    spec_centered = mean_spectrum - mean_spectrum.mean()
    spec_norm = np.sum(spec_centered ** 2)
    if spec_norm > 1e-10:
        spec_ac = np.correlate(spec_centered, spec_centered, mode='full')
        spec_ac = spec_ac[len(spec_ac)//2:]
        spec_ac = spec_ac / (spec_norm + 1e-8)
        harmonic_strength = np.max(spec_ac[1:]) if len(spec_ac) > 1 else 0.0
    else:
        harmonic_strength = 0.0
    beauty = 0.5 * smoothness + 0.5 * max(harmonic_strength, 0)

    return {
        'valence': float(valence),
        'arousal': float(arousal),
        'dominance': float(dominance),
        'tension': float(tension),
        'beauty': float(beauty),
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
    all_mels, all_emo, all_cats, valid_files = [], [], [], []
    for mf in mel_files:
        mel = np.load(mf)
        if mel.shape[0] < 4: continue
        idx = int(mf.stem)
        if idx >= len(categories): continue
        cn = categories[idx]
        if cn in ('ambient', 'mixed', 'music'): continue
        emo = compute_emotion_properties(mel)
        all_mels.append(mel)
        all_emo.append(emo)
        all_cats.append(cat_to_id[cn])
        valid_files.append(mf.name)

    print(f"Valid samples: {len(all_mels)}")

    prop_names = ['valence', 'arousal', 'dominance', 'tension', 'beauty']
    V = np.array([[e[p] for p in prop_names] for e in all_emo])

    print(f"\nEmotion property stats:")
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
    print("TEST 1: Can embeddings predict emotional dimensions?")
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

    # TEST 2: Binary emotion classification
    print("\n" + "="*60)
    print("TEST 2: Binary emotion classification")
    print("="*60)
    binary_tasks = {
        'negative_vs_positive': ('valence', lambda x: x > np.median(x)),
        'calm_vs_excited': ('arousal', lambda x: x > np.median(x)),
        'submissive_vs_dominant': ('dominance', lambda x: x > np.median(x)),
        'relaxed_vs_tense': ('tension', lambda x: x > np.median(x)),
        'ugly_vs_beautiful': ('beauty', lambda x: x > np.median(x)),
    }
    test2 = {}
    for task, (prop, binarize) in binary_tasks.items():
        idx = prop_names.index(prop)
        y = binarize(V[:, idx]).astype(int)
        scores = cross_val_score(LogisticRegression(max_iter=1000, C=1.0), E_scaled, y, cv=5, scoring='accuracy')
        test2[task] = float(scores.mean())
        marker = "***" if scores.mean() > 0.75 else "**" if scores.mean() > 0.65 else ""
        print(f"  {task:25s}: {scores.mean():.1%} {marker}")
    results['test2_binary'] = test2

    # TEST 3: Emotion correlation structure
    print("\n" + "="*60)
    print("TEST 3: Emotion correlation structure")
    print("="*60)
    print("(Russell's circumplex predicts specific correlations)")
    expected = {
        ('valence', 'arousal'): 'independent',  # orthogonal in circumplex
        ('valence', 'beauty'): 'positive',       # pleasant = beautiful
        ('arousal', 'tension'): 'positive',      # excited = tense
        ('dominance', 'arousal'): 'positive',    # dominant = energetic
    }
    corr = np.corrcoef(V.T)
    test3 = {}
    matches = 0
    total = len(expected)
    for (p1, p2), exp in expected.items():
        i, j = prop_names.index(p1), prop_names.index(p2)
        r = corr[i, j]
        if exp == 'independent':
            match = abs(r) < 0.3  # near-zero = independent
            label = f"|r|<0.3={'YES' if match else 'NO'}"
        elif exp == 'positive':
            match = r > 0.1
            label = f"r>0={'YES' if match else 'NO'}"
        else:
            match = r < -0.1
            label = f"r<0={'YES' if match else 'NO'}"
        matches += match
        test3[f"{p1}_vs_{p2}"] = {'r': float(r), 'expected': exp, 'match': bool(match)}
        print(f"  {p1:12s} vs {p2:12s}: r={r:+.3f} (expected {exp}, {label})")
    print(f"\n  Emotion structure score: {matches}/{total}")
    results['test3_correlations'] = test3

    # TEST 4: Emotional profiles per category
    print("\n" + "="*60)
    print("TEST 4: Emotional profile of each sound")
    print("="*60)
    cats_arr = np.array(all_cats)

    # Most emotionally extreme categories
    test4 = {}
    for i, name in enumerate(prop_names):
        vals = {cat_names[c]: V[cats_arr == c, i].mean() for c in sorted(set(all_cats))}
        s = sorted(vals.items(), key=lambda x: x[1])
        test4[name] = {'lowest': s[0][0], 'highest': s[-1][0]}
        print(f"  {name:12s}: lowest={s[0][0]:20s} ({s[0][1]:.3f})  highest={s[-1][0]:20s} ({s[-1][1]:.3f})")

    # Emotional quadrant analysis
    print("\n  Emotional quadrants:")
    valence_idx = prop_names.index('valence')
    arousal_idx = prop_names.index('arousal')
    v_med = np.median(V[:, valence_idx])
    a_med = np.median(V[:, arousal_idx])

    quadrants = {
        'happy (high V, high A)': [],
        'calm (high V, low A)': [],
        'angry (low V, high A)': [],
        'sad (low V, low A)': [],
    }
    for c in sorted(set(all_cats)):
        mask = cats_arr == c
        mv = V[mask, valence_idx].mean()
        ma = V[mask, arousal_idx].mean()
        name = cat_names[c]
        if mv >= v_med and ma >= a_med:
            quadrants['happy (high V, high A)'].append(name)
        elif mv >= v_med:
            quadrants['calm (high V, low A)'].append(name)
        elif ma >= a_med:
            quadrants['angry (low V, high A)'].append(name)
        else:
            quadrants['sad (low V, low A)'].append(name)

    for quad, cats in quadrants.items():
        print(f"  {quad}: {', '.join(cats[:5])}{'...' if len(cats) > 5 else ''}")
    results['test4_profiles'] = test4
    results['test4_quadrants'] = quadrants

    # VERDICT
    strong_r2 = sum(1 for v in test1.values() if v > 0.3)
    good_class = sum(1 for v in test2.values() if v > 0.65)
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  Linear prediction: {strong_r2}/5 with R-sq > 0.3")
    print(f"  Classification:    {good_class}/5 > 65%")
    print(f"  Correlations:      {matches}/{total} match Russell's circumplex")

    if strong_r2 >= 3 and matches >= 3:
        verdict = "STRONG - sound encodes emotional dimensions"
    elif strong_r2 >= 2 or matches >= 2:
        verdict = "PARTIAL - some emotional dimensions present"
    else:
        verdict = "WEAK - emotion mapping is weak"
    print(f"  VERDICT: {verdict}")
    results['verdict'] = verdict

    with open('data/wc_sound_as_emotion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/wc_sound_as_emotion_results.json")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    args = p.parse_args()
    run_experiment(args.data_dir, args.checkpoint)
