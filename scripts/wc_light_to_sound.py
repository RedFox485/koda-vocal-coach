#!/usr/bin/env python3
"""
Wildcard: Light to Sound (Reverse Direction)
=============================================
We showed sound encodes light-analog properties. Now flip it:
Can we PREDICT acoustic properties from visual-analog features?

This is fundamentally an EXPANSION problem:
- Light: ~1 octave (380-780nm), 3 channels (RGB), slow changes
- Sound: ~10 octaves (20-20kHz), 40+ mel bands, fast changes (ms)

Converting light → sound requires DIMENSIONAL EXPANSION:
each visual feature must unfold into multiple acoustic dimensions.

This experiment tests:
1. Do light-analog features predict acoustic properties? (reverse regression)
2. What is the expansion ratio? (how many acoustic dims per visual dim?)
3. Is the mapping symmetric? (light→sound vs sound→light)
4. What information is LOST going light→sound? (the expansion gap)

If light→sound works nearly as well as sound→light, the mapping
is truly bidirectional and the dimensional structure is real.
If it works WORSE, that tells us sound has MORE structure than light
can capture — sound is the richer representation.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def compute_visual_properties(mel_frames):
    """Visual-analog properties from spectrograms (same as wc_light_as_sound)."""
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    fb = np.linspace(0, 1, n)
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)
    centroid = np.sum(fb * ms) / np.sum(ms)
    rms = np.sqrt(np.mean(ml ** 2))
    third = n // 3
    low_r = np.sum(ms[:third]) / (np.sum(ms) + 1e-8)
    high_r = np.sum(ms[2*third:]) / (np.sum(ms) + 1e-8)
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    bw = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)

    color = centroid
    brightness = np.log1p(rms)
    saturation = 1.0 - flatness
    warmth = low_r
    coolness = high_r

    if T >= 2:
        flux = np.mean(np.abs(np.diff(fe))) / (np.mean(fe) + 1e-8)
        flicker = flux
    else:
        flicker = 0.0

    sd = np.abs(np.diff(ms))
    texture = np.mean(sd) / (np.max(sd) + 1e-8)

    return {
        'color': float(color),
        'brightness': float(brightness),
        'saturation': float(saturation),
        'warmth': float(warmth),
        'coolness': float(coolness),
        'flicker': float(flicker),
        'texture': float(texture),
    }


def compute_acoustic_properties(mel_frames):
    """Rich acoustic properties — the TARGET for light→sound prediction."""
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    fb = np.linspace(0, 1, n)
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)
    centroid = np.sum(fb * ms) / np.sum(ms)
    rms = np.sqrt(np.mean(ml ** 2))

    # Basic spectral
    spectral_centroid = centroid
    spectral_bandwidth = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))
    spectral_rolloff = fb[np.searchsorted(np.cumsum(ms) / np.sum(ms), 0.85)] if np.sum(ms) > 0 else 0.5
    spectral_flatness = np.exp(np.mean(np.log(ms + 1e-10))) / (np.mean(ms) + 1e-8)
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)

    # Energy
    energy_mean = float(np.log1p(rms))
    if T >= 4:
        energy_var = float(np.std(fe) / (np.mean(fe) + 1e-8))
    else:
        energy_var = 0.0

    # Temporal
    if T >= 2:
        spectral_flux = float(np.mean(np.sqrt(np.sum(np.diff(ml, axis=0)**2, axis=1))) / (rms + 1e-8))
    else:
        spectral_flux = 0.0

    if T >= 4:
        energy_slope = float(np.polyfit(np.arange(T), fe, 1)[0] / (np.mean(fe) + 1e-8))
    else:
        energy_slope = 0.0

    # Harmonic structure
    spec_ac = np.correlate(ms - ms.mean(), ms - ms.mean(), 'full')
    spec_ac = spec_ac[len(spec_ac)//2:]
    harmonicity = float(max(spec_ac[1:].max() / (spec_ac[0] + 1e-8), 0)) if len(spec_ac) > 1 else 0.0

    # Temporal periodicity
    if T >= 10:
        ec = fe - fe.mean(); norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            periodicity = float(max(ac[1:].max(), 0)) if len(ac) > 1 else 0.0
        else:
            periodicity = 0.0
    else:
        periodicity = 0.0

    # Band energies (low/mid/high)
    third = n // 3
    low_energy = float(np.sum(ms[:third]) / (np.sum(ms) + 1e-8))
    mid_energy = float(np.sum(ms[third:2*third]) / (np.sum(ms) + 1e-8))
    high_energy = float(np.sum(ms[2*third:]) / (np.sum(ms) + 1e-8))

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flatness': spectral_flatness,
        'entropy': entropy,
        'energy_mean': energy_mean,
        'energy_var': energy_var,
        'spectral_flux': spectral_flux,
        'energy_slope': energy_slope,
        'harmonicity': harmonicity,
        'periodicity': periodicity,
        'low_energy': low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy,
    }


def load_encoder(cp, dev='cpu'):
    try:
        from mamba_ssm import Mamba2; real=True
    except: real=False
    if real:
        class E(nn.Module):
            def __init__(self):
                super().__init__(); self.input_proj=nn.Linear(40,128)
                self.layers=nn.ModuleList([Mamba2(d_model=128,d_state=16,d_conv=4,expand=2) for _ in range(6)])
                self.norm=nn.LayerNorm(128)
            def forward(self,x):
                x=self.input_proj(x)
                for l in self.layers: x=x+l(x)
                return self.norm(x)
    else:
        class CB(nn.Module):
            def __init__(self):
                super().__init__(); self.conv=nn.Conv1d(128,256,4,padding=2); self.act=nn.SiLU(); self.proj=nn.Linear(256,128)
            def forward(self,x):
                y=self.conv(x.transpose(1,2)).transpose(1,2)[:,:x.size(1),:]; return self.proj(self.act(y))
        class E(nn.Module):
            def __init__(self):
                super().__init__(); self.input_proj=nn.Linear(40,128)
                self.layers=nn.ModuleList([CB() for _ in range(6)]); self.norm=nn.LayerNorm(128)
            def forward(self,x):
                x=self.input_proj(x)
                for l in self.layers: x=x+l(x)
                return self.norm(x)
    e=E(); ck=torch.load(cp,map_location=dev,weights_only=True)
    es=ck['encoder'] if 'encoder' in ck else {}
    ld=e.load_state_dict(es,strict=False)
    print(f"Encoder: {len(es)-len(ld.missing_keys)}/{len(list(e.state_dict().keys()))} params")
    e.to(dev); e.train(False); return e


def run(data_dir='data/training/mel/esc50', cp='checkpoints/clap_distill_spread_a.pt'):
    dev='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")
    meta=json.load(open('data/clap_meta.json')); cats=meta['categories']
    uc=sorted(set(c for c in cats if c not in ('ambient','mixed','music')))
    c2i={c:i for i,c in enumerate(uc)}

    mels, visuals, acoustics = [], [], []
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m)
        visuals.append(compute_visual_properties(m))
        acoustics.append(compute_acoustic_properties(m))
    print(f"Samples: {len(mels)}")

    vpn = ['color','brightness','saturation','warmth','coolness','flicker','texture']
    apn = ['spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_flatness',
           'entropy','energy_mean','energy_var','spectral_flux','energy_slope',
           'harmonicity','periodicity','low_energy','mid_energy','high_energy']

    Vvis = np.array([[v[p] for p in vpn] for v in visuals])
    Vac = np.array([[a[p] for p in apn] for a in acoustics])
    Vs_vis = StandardScaler().fit_transform(Vvis)
    Vs_ac = StandardScaler().fit_transform(Vac)

    # Load encoder
    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    Es=StandardScaler().fit_transform(np.array(embs))

    results = {}

    # ================================================================
    # TEST 1: Light → Sound (can visual features predict acoustic properties?)
    # ================================================================
    print("\n"+"="*65)
    print("TEST 1: LIGHT → SOUND (visual features predicting acoustic)")
    print("="*65)
    t1_l2s = {}
    for i, name in enumerate(apn):
        y = Vac[:, i]
        if np.std(y) < 1e-10: continue
        sc = cross_val_score(Ridge(alpha=1.0), Vs_vis, y, cv=5, scoring='r2')
        r2 = sc.mean(); t1_l2s[name] = float(r2)
        mk = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:22s}: R-sq = {r2:.3f} {mk}")
    results['light_to_sound'] = t1_l2s

    # ================================================================
    # TEST 2: Sound → Light (reverse — for symmetry comparison)
    # ================================================================
    print("\n"+"="*65)
    print("TEST 2: SOUND → LIGHT (acoustic features predicting visual)")
    print("="*65)
    t2_s2l = {}
    for i, name in enumerate(vpn):
        y = Vvis[:, i]
        if np.std(y) < 1e-10: continue
        sc = cross_val_score(Ridge(alpha=1.0), Vs_ac, y, cv=5, scoring='r2')
        r2 = sc.mean(); t2_s2l[name] = float(r2)
        mk = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:22s}: R-sq = {r2:.3f} {mk}")
    results['sound_to_light'] = t2_s2l

    # ================================================================
    # TEST 3: Symmetry analysis
    # ================================================================
    print("\n"+"="*65)
    print("TEST 3: Symmetry — is the mapping bidirectional?")
    print("="*65)

    l2s_mean = np.mean(list(t1_l2s.values()))
    s2l_mean = np.mean(list(t2_s2l.values()))
    print(f"  Light→Sound mean R²:  {l2s_mean:.3f}")
    print(f"  Sound→Light mean R²:  {s2l_mean:.3f}")
    print(f"  Asymmetry ratio:      {l2s_mean / (s2l_mean + 1e-8):.2f}")

    l2s_above = sum(1 for v in t1_l2s.values() if v > 0.3)
    s2l_above = sum(1 for v in t2_s2l.values() if v > 0.3)
    print(f"  Light→Sound R²>0.3:   {l2s_above}/{len(t1_l2s)}")
    print(f"  Sound→Light R²>0.3:   {s2l_above}/{len(t2_s2l)}")

    if l2s_mean < s2l_mean * 0.7:
        print("\n  → Sound→Light works BETTER. Sound is the richer representation.")
        print("    Light can only capture a subset of acoustic structure.")
        print("    A dimensional EXPANDER would be needed for light→sound.")
    elif l2s_mean > s2l_mean * 1.3:
        print("\n  → Light→Sound works BETTER. Surprising — visual features are")
        print("    sufficient to predict acoustic properties.")
    else:
        print("\n  → Roughly SYMMETRIC. The cross-modal mapping is bidirectional.")
    results['symmetry'] = {'l2s_mean': float(l2s_mean), 's2l_mean': float(s2l_mean)}

    # ================================================================
    # TEST 4: Expansion ratio — how many acoustic dims per visual dim?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 4: Dimensional expansion ratio")
    print("="*65)

    # How much of acoustic variance can 7 visual dims explain?
    from sklearn.cross_decomposition import CCA
    n_components = min(7, len(apn), len(vpn))
    cca = CCA(n_components=n_components)
    cca.fit(Vs_vis, Vs_ac)
    X_c, Y_c = cca.transform(Vs_vis, Vs_ac)

    # Canonical correlations
    can_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    print(f"  Canonical correlations: {[f'{c:.3f}' for c in can_corrs]}")

    significant = sum(1 for c in can_corrs if c > 0.3)
    print(f"  Significant shared dims (r>0.3): {significant}/{n_components}")
    print(f"  Effective expansion ratio: {len(apn)}/{significant} = {len(apn)/max(significant,1):.1f}x")
    print(f"  → Each shared visual dimension maps to ~{len(apn)/max(significant,1):.0f} acoustic dimensions")

    results['expansion'] = {
        'canonical_correlations': [float(c) for c in can_corrs],
        'shared_dims': significant,
        'expansion_ratio': float(len(apn) / max(significant, 1))
    }

    # ================================================================
    # TEST 5: What's LOST going light→sound? (the expansion gap)
    # ================================================================
    print("\n"+"="*65)
    print("TEST 5: The expansion gap — what light can't capture")
    print("="*65)

    # For each acoustic property, compare light-prediction vs encoder-prediction
    gaps = {}
    for i, name in enumerate(apn):
        y = Vac[:, i]
        if np.std(y) < 1e-10: continue
        sc_light = cross_val_score(Ridge(alpha=1.0), Vs_vis, y, cv=5, scoring='r2').mean()
        sc_enc = cross_val_score(Ridge(alpha=1.0), Es, y, cv=5, scoring='r2').mean()
        gap = sc_enc - sc_light
        gaps[name] = {'light': float(sc_light), 'encoder': float(sc_enc), 'gap': float(gap)}
        marker = " ← EXPANSION NEEDED" if gap > 0.2 else ""
        print(f"  {name:22s}: light={sc_light:.3f}  encoder={sc_enc:.3f}  gap={gap:+.3f}{marker}")

    results['expansion_gaps'] = gaps

    mean_gap = np.mean([g['gap'] for g in gaps.values()])
    print(f"\n  Mean expansion gap: {mean_gap:+.3f}")
    if mean_gap > 0.1:
        print("  → Encoder captures significantly more than light-analog features.")
        print("  → A light dimensional expander would need to recover this gap.")
        print(f"  → The expander must generate ~{mean_gap*len(apn):.0f} extra dimensions of information.")
    results['mean_expansion_gap'] = float(mean_gap)

    # ================================================================
    # TEST 6: PCA — shared vs unique variance
    # ================================================================
    print("\n"+"="*65)
    print("TEST 6: Shared vs unique dimensional structure")
    print("="*65)

    combined = np.hstack([Vs_vis, Vs_ac])
    pca = PCA()
    pca.fit(combined)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1

    pca_vis = PCA(); pca_vis.fit(Vs_vis)
    pca_ac = PCA(); pca_ac.fit(Vs_ac)
    n_90_vis = int(np.searchsorted(np.cumsum(pca_vis.explained_variance_ratio_), 0.90)) + 1
    n_90_ac = int(np.searchsorted(np.cumsum(pca_ac.explained_variance_ratio_), 0.90)) + 1

    print(f"  Visual dims for 90%:     {n_90_vis}/{len(vpn)}")
    print(f"  Acoustic dims for 90%:   {n_90_ac}/{len(apn)}")
    print(f"  Combined dims for 90%:   {n_90}/{len(vpn)+len(apn)}")
    print(f"  Expected if independent: {n_90_vis + n_90_ac}")
    shared = (n_90_vis + n_90_ac) - n_90
    print(f"  Shared dimensions:       ~{shared}")
    print(f"  Unique acoustic dims:    ~{n_90_ac - shared}")
    print(f"  → A light→sound expander must synthesize ~{n_90_ac - shared} dimensions")

    results['dim_analysis'] = {
        'visual_90': n_90_vis, 'acoustic_90': n_90_ac,
        'combined_90': n_90, 'shared': shared,
        'unique_acoustic': n_90_ac - shared
    }

    # Verdict
    l2s_good = l2s_mean > 0.3
    symmetric = abs(l2s_mean - s2l_mean) < 0.15
    expansion_real = mean_gap > 0.05

    if l2s_good and expansion_real:
        verdict = "CONFIRMED"
        print(f"\nVERDICT: {verdict} — Light→Sound works but requires dimensional expansion")
    elif l2s_good:
        verdict = "SYMMETRIC"
        print(f"\nVERDICT: {verdict} — Bidirectional mapping, minimal expansion needed")
    else:
        verdict = "WEAK"
        print(f"\nVERDICT: {verdict} — Light→Sound mapping is weak")

    results['verdict'] = verdict
    json.dump(results, open('data/wc_light_to_sound_results.json', 'w'), indent=2)
    print("Saved")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    a = p.parse_args(); run(a.data_dir, a.checkpoint)
