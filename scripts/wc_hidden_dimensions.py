#!/usr/bin/env python3
"""
Wildcard: Hidden Dimensions — Cross-Linguistic Universal Word Pairs
=====================================================================
Words that mean the same TWO things across unrelated languages aren't
metaphors — they're evidence of shared frequency dimensions that humans
can perceive but never named as separate senses.

Each universal dual-meaning word points at a hidden cross-modal dimension.
This experiment maps 15 such dimensions and tests whether our encoder
captures them.

Universal word pairs (same dual meaning across 3+ language families):
  1. bright/smart    — EN, FR(brillant), ES(brillante), JP(kagayaku), ZH(congming/liang)
  2. heavy/serious   — EN, FR(grave), DE(schwer), ES(grave/pesado), RU(tyazhelyy)
  3. sharp/clever    — EN, JP(surudoi), FR(vif), DE(scharf), ZH(jianrui)
  4. warm/friendly   — nearly all languages
  5. dark/sad        — EN, FR(sombre), ES(oscuro), DE(dunkel/traurig)
  6. sweet/kind      — EN, FR(doux), ES(dulce), IT(dolce), AR(hilw)
  7. deep/profound   — EN, FR(profond), ES(profundo), DE(tief), ZH(shen)
  8. hard/difficult  — EN, FR(dur), ES(duro), DE(hart/schwer), JP(katai/muzukashii)
  9. clear/obvious   — EN, FR(clair), ES(claro), IT(chiaro), PT(claro)
 10. rough/difficult — EN, FR(rude), ES(aspero/dificil), JP(arai)
 11. smooth/easy     — EN, ES(suave), IT(liscio), multiple Asian languages
 12. high/happy      — EN(elated), FR(haut/gai), ZH(gao xing = high + mood)
 13. vibrant/alive   — EN, FR(vibrant/vivant), ES(vibrante/vivo), IT(vibrante)
 14. harmony/peace   — EN, FR(harmonie/paix), JP(wa = harmony AND peace), ZH(hexie)
 15. resonance/truth — EN(rings true), FR(resonner), DE(widerhallen), Sanskrit(dhvani)

Each of these suggests a HIDDEN DIMENSION that connects two seemingly
unrelated domains through shared frequency structure.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_hidden_dimensions(mel_frames):
    """
    Compute 15 hidden dimensions suggested by cross-linguistic universals.
    Each is a frequency property that bridges two conceptual domains.
    """
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
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)
    sd = np.abs(np.diff(ms))
    smoothness = 1.0 - (np.mean(sd) / (np.max(sd) + 1e-8))
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    bw = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))
    if T >= 2:
        flux = np.mean(np.sqrt(np.sum(np.diff(ml, axis=0)**2, axis=1)))
    else:
        flux = 0.0
    if T >= 4:
        am = np.std(fe) / (np.mean(fe) + 1e-8)
    else:
        am = 0.0

    # Autocorrelation of energy
    if T >= 8:
        ec = fe - fe.mean(); norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            periodicity = max(ac[1:].max(), 0) if len(ac) > 1 else 0
        else:
            periodicity = 0.0
    else:
        periodicity = 0.0

    # Spectral autocorrelation (harmonicity)
    sc = ms - ms.mean()
    snorm = np.sum(sc**2)
    if snorm > 1e-10:
        sac = np.correlate(sc, sc, 'full')[len(sc)-1:]
        sac = sac / (snorm + 1e-8)
        harmonicity = max(sac[1:].max(), 0) if len(sac) > 1 else 0
    else:
        harmonicity = 0.0

    # Onset characteristics
    if T >= 4:
        onset_rate = np.max(np.diff(fe[:T//4])) / (np.mean(fe) + 1e-8)
    else:
        onset_rate = 0.0

    # Spectral concentration
    pr = (sn.sum()**2) / (np.sum(sn**2) + 1e-10)
    concentration = 1.0 - (pr / n)

    # Energy trend
    if T >= 4:
        slope = np.polyfit(np.arange(T), fe, 1)[0]
        trend = slope / (np.mean(fe) + 1e-8)
    else:
        trend = 0.0

    # 2nd derivative (curvature)
    if n >= 3:
        curv = np.mean(np.abs(np.diff(ms, n=2))) / (np.mean(ms) + 1e-8)
    else:
        curv = 0.0

    # === THE 15 HIDDEN DIMENSIONS ===

    # 1. BRILLIANCE (bright/smart): high energy + high frequency + clarity
    #    Light = luminance. Intelligence = processing speed. Both = high-freq energy.
    brilliance = 0.4 * centroid + 0.3 * np.clip(rms/1000, 0, 1) + 0.3 * (1-flatness)

    # 2. GRAVITY (heavy/serious): low frequency + sustained + slow change
    #    Physical weight = low resonance. Emotional weight = sustained impact.
    gravity = 0.4 * low_r + 0.3 * (1-am) + 0.3 * (1 - np.clip(flux/500, 0, 1))

    # 3. ACUITY (sharp/clever): fast onset + narrow bandwidth + high contrast
    #    Physical sharpness = narrow point. Mental sharpness = precise discrimination.
    acuity = 0.35 * np.clip(onset_rate/5, 0, 1) + 0.35 * concentration + 0.3 * (1-bw)

    # 4. WARMTH (warm/friendly): low centroid + smooth + moderate energy
    #    Thermal warmth = low-freq radiation. Social warmth = non-threatening presence.
    warmth = 0.4 * low_r + 0.3 * smoothness + 0.3 * (1 - np.clip(onset_rate/5, 0, 1))

    # 5. SHADOW (dark/sad): low brightness + low centroid + declining energy
    #    Physical dark = absence of light. Emotional dark = absence of vitality.
    shadow = 0.35 * (1-centroid) + 0.35 * (1 - np.clip(rms/1000, 0, 1)) + 0.3 * max(-trend, 0)

    # 6. SWEETNESS (sweet/kind): smooth + harmonic + mid-frequency + consonant
    #    Gustatory = pleasant chemistry. Social = pleasant interaction. Same pleasant pattern.
    sweetness = 0.3 * smoothness + 0.3 * harmonicity + 0.2 * (1-flatness) + 0.2 * (1-am)

    # 7. PROFUNDITY (deep/profound): low frequency + high entropy + long sustain
    #    Physical depth = low resonance. Intellectual depth = information density.
    profundity = 0.35 * low_r + 0.35 * entropy + 0.3 * (1 - np.clip(flux/500, 0, 1))

    # 8. RESISTANCE (hard/difficult): high onset + high energy + spectral rigidity
    #    Physical hardness = high impedance. Cognitive difficulty = high resistance.
    resistance = 0.35 * np.clip(onset_rate/5, 0, 1) + 0.35 * np.clip(rms/1000, 0, 1) + 0.3 * (1-smoothness)

    # 9. TRANSPARENCY (clear/obvious): low noise + high harmonic content + narrow
    #    Optical clarity = low scatter. Cognitive clarity = low ambiguity.
    transparency = 0.35 * (1-flatness) + 0.35 * harmonicity + 0.3 * concentration

    # 10. FRICTION (rough/difficult): high flatness + high spectral variation + noisy
    #     Surface roughness = micro-obstacles. Cognitive friction = processing obstacles.
    friction = 0.35 * flatness + 0.35 * curv + 0.3 * am

    # 11. FLUENCY (smooth/easy): low spectral variation + predictable + flowing
    #     Surface smoothness = no obstacles. Cognitive ease = no friction.
    if T >= 4:
        predictability = 1.0 - np.std(np.diff(fe)) / (np.mean(np.abs(np.diff(fe))) + 1e-8)
        predictability = max(predictability, 0)
    else:
        predictability = 0.5
    fluency = 0.35 * smoothness + 0.35 * predictability + 0.3 * (1-flatness)

    # 12. ELEVATION (high/happy): high centroid + rising energy + bright
    #     Physical height = high frequency. Emotional height = positive arousal.
    elevation = 0.35 * centroid + 0.35 * max(trend, 0) + 0.3 * high_r

    # 13. VIVACITY (vibrant/alive): high modulation + rich spectrum + dynamic
    #     Vibration = physical oscillation. Vitality = biological oscillation.
    vivacity = 0.35 * am + 0.35 * entropy + 0.3 * np.clip(flux/500, 0, 1)

    # 14. CONSONANCE (harmony/peace): periodic + harmonic + balanced
    #     Musical harmony = frequency ratios. Social harmony = balanced relations.
    consonance = 0.35 * harmonicity + 0.35 * periodicity + 0.3 * smoothness

    # 15. VERITY (resonance/truth): strong resonance + stable + self-reinforcing
    #     Physical resonance = energy amplification. "Rings true" = self-consistent.
    #     Resonant systems are STABLE — they resist perturbation. Truth persists.
    verity = 0.35 * harmonicity + 0.35 * (1-am) + 0.3 * periodicity

    return {
        'brilliance': float(brilliance),
        'gravity': float(gravity),
        'acuity': float(acuity),
        'warmth': float(warmth),
        'shadow': float(shadow),
        'sweetness': float(sweetness),
        'profundity': float(profundity),
        'resistance': float(resistance),
        'transparency': float(transparency),
        'friction': float(friction),
        'fluency': float(fluency),
        'elevation': float(elevation),
        'vivacity': float(vivacity),
        'consonance': float(consonance),
        'verity': float(verity),
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
    c2i={c:i for i,c in enumerate(uc)}; cn={i:c for c,i in c2i.items()}

    mels, dims, catids = [], [], []
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); dims.append(compute_hidden_dimensions(m)); catids.append(c2i[cats[idx]])
    print(f"Samples: {len(mels)}")

    dim_names = list(dims[0].keys())
    V = np.array([[d[k] for k in dim_names] for d in dims])

    # Language origins for each dimension
    origins = {
        'brilliance': 'bright/smart — EN,FR,ES,JP,ZH',
        'gravity': 'heavy/serious — EN,FR,DE,ES,RU',
        'acuity': 'sharp/clever — EN,JP,FR,DE,ZH',
        'warmth': 'warm/friendly — universal',
        'shadow': 'dark/sad — EN,FR,ES,DE',
        'sweetness': 'sweet/kind — EN,FR,ES,IT,AR',
        'profundity': 'deep/profound — EN,FR,ES,DE,ZH',
        'resistance': 'hard/difficult — EN,FR,ES,DE,JP',
        'transparency': 'clear/obvious — EN,FR,ES,IT,PT',
        'friction': 'rough/difficult — EN,FR,ES,JP',
        'fluency': 'smooth/easy — EN,ES,IT,+Asian',
        'elevation': 'high/happy — EN,FR,ZH',
        'vivacity': 'vibrant/alive — EN,FR,ES,IT',
        'consonance': 'harmony/peace — EN,FR,JP,ZH',
        'verity': 'resonance/truth — EN,FR,DE,Sanskrit',
    }

    print(f"\n{'='*70}")
    print("THE 15 HIDDEN DIMENSIONS OF FREQUENCY")
    print(f"{'='*70}")
    for i, name in enumerate(dim_names):
        print(f"  {i+1:2d}. {name:14s}: {origins[name]}")
        print(f"      mean={V[:,i].mean():.3f}, std={V[:,i].std():.3f}")

    # Get embeddings
    enc = load_encoder(cp, dev)
    embs = []
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E = np.array(embs)
    Es = StandardScaler().fit_transform(E)

    results = {}

    # TEST 1: Which hidden dimensions does the encoder capture?
    print(f"\n{'='*70}")
    print("TEST 1: Can the encoder predict each hidden dimension?")
    print(f"{'='*70}")
    t1 = {}
    for i, name in enumerate(dim_names):
        y = V[:, i]
        if np.std(y) < 1e-10: t1[name] = 0.0; continue
        sc = cross_val_score(Ridge(alpha=1.0), Es, y, cv=5, scoring='r2')
        r2 = sc.mean(); t1[name] = float(r2)
        mk = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        lang = origins[name].split(' — ')[1]
        print(f"  {name:14s}: R-sq = {r2:.3f} {mk:3s}  [{lang}]")
    results['test1_r2'] = t1

    strong = sum(1 for v in t1.values() if v > 0.3)
    print(f"\n  {strong}/15 hidden dimensions captured (R-sq > 0.3)")

    # TEST 2: Binary classification for each dimension
    print(f"\n{'='*70}")
    print("TEST 2: Binary classification of hidden dimensions")
    print(f"{'='*70}")
    t2 = {}
    for name in dim_names:
        idx = dim_names.index(name)
        y = (V[:, idx] > np.median(V[:, idx])).astype(int)
        sc = cross_val_score(LogisticRegression(max_iter=1000, C=1.0), Es, y, cv=5, scoring='accuracy')
        t2[name] = float(sc.mean())
        mk = "***" if sc.mean() > 0.75 else "**" if sc.mean() > 0.65 else ""
        print(f"  {name:14s}: {sc.mean():.1%} {mk}")
    results['test2_binary'] = t2

    # TEST 3: How many truly independent hidden dimensions exist?
    print(f"\n{'='*70}")
    print("TEST 3: How many INDEPENDENT hidden dimensions?")
    print(f"{'='*70}")
    from sklearn.decomposition import PCA
    Vs = StandardScaler().fit_transform(V)
    Vs = np.nan_to_num(Vs)
    pca = PCA()
    pca.fit(Vs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for thresh in [0.80, 0.90, 0.95, 0.99]:
        nd = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh:.0%} variance: {nd} dimensions")
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"\n  15 linguistic dimensions compress to {n95} at 95%")
    print(f"  This means {15 - n95} dimensions are redundant (shared across word pairs)")
    print(f"  And {n95} are truly independent frequency dimensions")
    results['test3_independent_dims'] = n95

    # TEST 4: The correlation structure — which dimensions are the same?
    print(f"\n{'='*70}")
    print("TEST 4: Which hidden dimensions are actually the same?")
    print(f"{'='*70}")
    corr = np.corrcoef(V.T)
    # Find highly correlated pairs
    pairs = []
    for i in range(len(dim_names)):
        for j in range(i+1, len(dim_names)):
            r = corr[i, j]
            if abs(r) > 0.5:
                pairs.append((dim_names[i], dim_names[j], float(r)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("  Strongly correlated dimension pairs (|r| > 0.5):")
    t4_pairs = []
    for n1, n2, r in pairs[:15]:
        sign = "=" if r > 0 else "opposite of"
        print(f"    {n1:14s} {sign} {n2:14s}: r={r:+.3f}")
        t4_pairs.append({'dim1': n1, 'dim2': n2, 'r': r})
    results['test4_correlated_pairs'] = t4_pairs

    # Find the LEAST correlated pairs — these are the truly independent dimensions
    independent = []
    for i in range(len(dim_names)):
        max_corr = max(abs(corr[i, j]) for j in range(len(dim_names)) if j != i)
        independent.append((dim_names[i], max_corr))
    independent.sort(key=lambda x: x[1])

    print(f"\n  Most independent dimensions (lowest max correlation with others):")
    for name, mc in independent[:5]:
        print(f"    {name:14s}: max |r| with any other = {mc:.3f}")

    # TEST 5: Per-category dimension profiles — what do sounds look like in hidden space?
    print(f"\n{'='*70}")
    print("TEST 5: Sound profiles in hidden dimension space")
    print(f"{'='*70}")
    ca = np.array(catids)

    # Find the most extreme sound for each dimension
    t5 = {}
    for i, name in enumerate(dim_names):
        vals = {cn[c]: V[ca==c, i].mean() for c in sorted(set(catids))}
        s = sorted(vals.items(), key=lambda x: x[1])
        t5[name] = {'lowest': s[0][0], 'highest': s[-1][0]}
        word_pair = origins[name].split(' — ')[0]
        print(f"  {name:14s} ({word_pair}):")
        print(f"    least: {s[0][0]:18s} ({s[0][1]:.3f})")
        print(f"    most:  {s[-1][0]:18s} ({s[-1][1]:.3f})")
    results['test5_profiles'] = t5

    # GRAND SYNTHESIS
    print(f"\n{'='*70}")
    print("GRAND SYNTHESIS")
    print(f"{'='*70}")
    print(f"  Hidden dimensions captured by encoder: {strong}/15")
    good_binary = sum(1 for v in t2.values() if v > 0.65)
    print(f"  Binary classification > 65%:            {good_binary}/15")
    print(f"  Independent dimensions (PCA 95%):       {n95}/15")
    print(f"  Redundant (shared across word pairs):   {15 - n95}/15")

    print(f"\n  INTERPRETATION:")
    print(f"  Humans discovered {n95} independent frequency dimensions")
    print(f"  and named them {15} different ways across languages.")
    print(f"  Each language pair is a 2D projection of higher-D frequency space.")

    if strong >= 10:
        verdict = "CONFIRMED — language universals map to real frequency dimensions"
    elif strong >= 7:
        verdict = "STRONG — most linguistic universals have frequency bases"
    elif strong >= 4:
        verdict = "PARTIAL — some universals map to frequency, others may need other senses"
    else:
        verdict = "WEAK"
    print(f"  VERDICT: {verdict}")
    results['verdict'] = verdict
    results['summary'] = {
        'captured': strong, 'binary_good': good_binary,
        'independent': n95, 'redundant': 15 - n95
    }

    json.dump(results, open('data/wc_hidden_dimensions_results.json', 'w'), indent=2)
    print(f"\nSaved to data/wc_hidden_dimensions_results.json")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    a = p.parse_args(); run(a.data_dir, a.checkpoint)
