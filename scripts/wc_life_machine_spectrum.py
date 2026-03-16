#!/usr/bin/env python3
"""
Wildcard: Life-Machine Spectrum
================================
Life and machine aren't binary — they're a continuous spectrum.
What if we're viewing it from the wrong axis?

Hypothesis: There are AT LEAST two independent axes:
  Axis 1: Information substrate (digital ↔ biological)
    - 1s and 0s on one extreme, DNA on the other
    - Measured by: temporal regularity, spectral rigidity, pattern repetition

  Axis 2: Adaptive complexity (simple ↔ complex)
    - Simple reflexes vs complex adaptive behavior
    - Measured by: information content, multi-scale structure, context-sensitivity

This gives us a 2D space where:
  - Pure machine (clock_tick): high regularity, low complexity → bottom-left
  - Complex machine (engine): moderate regularity, moderate complexity → middle-left
  - Insects: moderate regularity, moderate complexity → MIDDLE (between machine and life!)
  - Plants (crackling_fire as proxy for organic-but-not-sentient): low regularity, moderate complexity
  - Animals (dog, cat, rooster): low regularity, high complexity → top-right
  - Humans (speech, laughing): lowest regularity, highest complexity → far top-right

The key insight: bugs ARE between mechanical and life. They run on
biological hardware but with relatively fixed programs (like firmware).
Plants are biological but non-behavioral (different axis entirely).

If this 2D model works, the 62% alive/mechanical accuracy makes PERFECT SENSE —
we were forcing a 2D space onto a 1D binary label.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Conceptual placement on the life-machine spectrum
# Format: category -> (substrate_score, complexity_score)
# substrate: 0 = pure digital/mechanical, 1 = pure biological
# complexity: 0 = simple/fixed, 1 = complex/adaptive
SPECTRUM_PLACEMENT = {
    # Pure mechanical (digital substrate, simple)
    'clock_tick': (0.05, 0.05),
    'clock_alarm': (0.05, 0.10),
    'mouse_click': (0.05, 0.05),
    'keyboard_typing': (0.10, 0.15),

    # Complex mechanical (digital substrate, moderate complexity)
    'engine': (0.10, 0.35),
    'train': (0.10, 0.30),
    'helicopter': (0.10, 0.40),
    'airplane': (0.10, 0.35),
    'chainsaw': (0.10, 0.25),
    'vacuum_cleaner': (0.10, 0.20),
    'washing_machine': (0.10, 0.20),
    'siren': (0.15, 0.30),
    'hand_saw': (0.15, 0.20),

    # Nature sounds (non-living but organic-adjacent)
    'rain': (0.40, 0.30),         # natural but not alive
    'thunderstorm': (0.40, 0.45), # complex natural system
    'wind': (0.35, 0.25),
    'sea_waves': (0.40, 0.35),
    'crackling_fire': (0.35, 0.40),  # chaotic, organic-feeling
    'water_drops': (0.35, 0.20),
    'pouring_water': (0.35, 0.25),

    # Insects (biological hardware, firmware behavior — THE MIDDLE)
    'insects': (0.65, 0.45),      # biological but repetitive, like firmware

    # Simple animals (biological, moderate-high complexity)
    'frog': (0.75, 0.55),
    'crow': (0.80, 0.60),
    'hen': (0.75, 0.50),
    'rooster': (0.80, 0.55),
    'chirping_birds': (0.80, 0.60),

    # Complex animals (biological, high complexity)
    'dog': (0.85, 0.75),
    'cat': (0.85, 0.70),
    'pig': (0.80, 0.60),
    'cow': (0.80, 0.55),
    'sheep': (0.75, 0.50),

    # Human (most biological, most complex)
    'crying_baby': (0.95, 0.70),
    'laughing': (0.95, 0.85),
    'sneezing': (0.90, 0.40),   # reflex — complex substrate, simple behavior
    'coughing': (0.90, 0.40),   # reflex
    'breathing': (0.90, 0.35),  # autonomic
    'snoring': (0.90, 0.30),    # involuntary
    'drinking_sipping': (0.90, 0.50),

    # Musical / human-created (interesting: biological creator, mechanical medium)
    'door_knock': (0.50, 0.15),
    'door_wood_creaks': (0.40, 0.20),
    'can_opening': (0.50, 0.15),
    'glass_breaking': (0.30, 0.25),
    'footsteps': (0.70, 0.35),
    'clapping': (0.80, 0.30),
    'brushing_teeth': (0.60, 0.20),
    'toilet_flush': (0.20, 0.20),
    'church_bells': (0.20, 0.35),
    'fireworks': (0.20, 0.40),
}


def compute_spectrum_properties(mel_frames):
    """Compute properties that place sounds on the life-machine spectrum."""
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)

    # === AXIS 1: Information Substrate (mechanical ↔ biological) ===

    # 1a. Temporal regularity (machines are periodic, life is quasi-periodic)
    if T >= 10:
        ec = fe - fe.mean()
        norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            # Perfect periodicity → ac peak near 1.0. Life → peak 0.3-0.7
            peak = ac[1:].max() if len(ac) > 1 else 0
            regularity = max(peak, 0)
        else:
            regularity = 0.5
    else:
        regularity = 0.5

    # 1b. Spectral rigidity (machines have fixed spectra, life varies)
    if T >= 4:
        # Frame-to-frame spectral correlation
        corrs = []
        for t in range(min(T-1, 50)):
            r = np.corrcoef(ml[t], ml[t+1])[0, 1]
            if not np.isnan(r): corrs.append(r)
        rigidity = np.mean(corrs) if corrs else 0.5
    else:
        rigidity = 0.5

    # 1c. Pattern repetition (exact repetition = mechanical)
    if T >= 8:
        # Split into segments and measure similarity
        seg_len = T // 4
        segs = [fe[i*seg_len:(i+1)*seg_len] for i in range(4)]
        seg_corrs = []
        for i in range(3):
            if len(segs[i]) == len(segs[i+1]) and len(segs[i]) > 1:
                r = np.corrcoef(segs[i], segs[i+1])[0, 1]
                if not np.isnan(r): seg_corrs.append(r)
        repetition = np.mean(seg_corrs) if seg_corrs else 0.5
    else:
        repetition = 0.5

    # 1d. Harmonic purity (machines = pure harmonics, life = noisy harmonics)
    spec_ac = np.correlate(ms - ms.mean(), ms - ms.mean(), 'full')
    spec_ac = spec_ac[len(spec_ac)//2:]
    harmonic_purity = max(spec_ac[1:].max() / (spec_ac[0] + 1e-8), 0) if len(spec_ac) > 1 else 0

    # Substrate score: low regularity + low rigidity + low repetition = biological
    substrate = 1.0 - (0.3 * regularity + 0.3 * rigidity + 0.25 * repetition + 0.15 * harmonic_purity)

    # === AXIS 2: Adaptive Complexity (simple ↔ complex) ===

    # 2a. Information content (entropy)
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)

    # 2b. Multi-scale structure (variance at different timescales)
    if T >= 8:
        var_1 = np.var(fe) + 1e-10
        var_2 = np.var(fe[::2]) + 1e-10
        var_4 = np.var(fe[::4]) + 1e-10
        # Ratio of variances across scales — complex signals have structure at all scales
        multi_scale = 1.0 - abs(np.log(var_2/var_1) - np.log(var_4/var_2)) / 3.0
        multi_scale = max(min(multi_scale, 1), 0)
    else:
        multi_scale = 0.5

    # 2c. Spectral diversity over time (adaptive systems change their spectrum)
    if T >= 4:
        quarter = max(T // 4, 1)
        specs = [ml[i*quarter:(i+1)*quarter].mean(axis=0) for i in range(min(4, T//quarter))]
        if len(specs) >= 2:
            spec_vars = np.var(np.array(specs), axis=0).mean()
            spec_diversity = np.log1p(spec_vars) / 5.0
            spec_diversity = min(spec_diversity, 1.0)
        else:
            spec_diversity = 0.0
    else:
        spec_diversity = 0.0

    # 2d. Temporal surprise (complex signals are locally unpredictable)
    if T >= 4:
        diffs = np.abs(np.diff(fe))
        if np.mean(diffs) > 1e-10:
            surprise = np.std(diffs) / (np.mean(diffs) + 1e-8)
        else:
            surprise = 0.0
    else:
        surprise = 0.0

    # 2e. Bandwidth utilization (complex signals use more frequency range)
    active_bins = np.sum(ms > ms.mean() * 0.1) / n

    complexity = 0.25 * entropy + 0.2 * multi_scale + 0.2 * spec_diversity + 0.2 * min(surprise, 1) + 0.15 * active_bins

    # === DERIVED AXES ===

    # Sentience proxy: high substrate + high complexity
    # (biological AND complex → more likely sentient)
    sentience = substrate * complexity * 2  # scale up

    # Firmware index: biological substrate but low complexity
    # (insects, breathing, reflexes)
    firmware = substrate * (1 - complexity)

    return {
        'substrate': float(substrate),
        'complexity': float(complexity),
        'regularity': float(regularity),
        'rigidity': float(rigidity),
        'repetition': float(repetition),
        'entropy': float(entropy),
        'multi_scale': float(multi_scale),
        'surprise': float(min(surprise, 2)),
        'sentience_proxy': float(sentience),
        'firmware_index': float(firmware),
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
    mels,props,catids,catnames=[],[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); props.append(compute_spectrum_properties(m))
        catids.append(c2i[cats[idx]]); catnames.append(cats[idx])
    print(f"Samples: {len(mels)}")

    all_props = ['substrate','complexity','regularity','rigidity','repetition',
                 'entropy','multi_scale','surprise','sentience_proxy','firmware_index']
    V = np.array([[p[k] for k in all_props] for p in props])

    # Load encoder
    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results = {}

    # ================================================================
    # TEST 1: Can embeddings predict the 2D spectrum position?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 1: Can embeddings predict spectrum properties?")
    print("="*65)
    t1 = {}
    for i, name in enumerate(all_props):
        y = V[:, i]
        if np.std(y) < 1e-10: t1[name] = 0.0; continue
        sc = cross_val_score(Ridge(alpha=1.0), Es, y, cv=5, scoring='r2')
        r2 = sc.mean(); t1[name] = float(r2)
        mk = "***" if r2 > 0.5 else "**" if r2 > 0.3 else "*" if r2 > 0.1 else ""
        print(f"  {name:18s}: R-sq = {r2:.3f} {mk}")
    results['test1'] = t1

    # ================================================================
    # TEST 2: Ground truth spectrum placement — do categories land where expected?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 2: Category placement on the spectrum")
    print("="*65)

    ca = np.array(catids)
    cat_positions = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        mask = ca == c
        if mask.sum() == 0: continue
        sub = V[mask, 0].mean()  # substrate
        cpx = V[mask, 1].mean()  # complexity
        cat_positions[nm] = (float(sub), float(cpx))

    # Sort by substrate score
    sorted_cats = sorted(cat_positions.items(), key=lambda x: x[1][0])
    print("\n  MECHANICAL ←————————————————————————→ BIOLOGICAL")
    print(f"  {'Category':20s} {'Substrate':>10s} {'Complexity':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for nm, (sub, cpx) in sorted_cats:
        bar_pos = int(sub * 40)
        bar = "." * bar_pos + "|" + "." * (40 - bar_pos)
        marker = ""
        if nm in SPECTRUM_PLACEMENT:
            expected_sub = SPECTRUM_PLACEMENT[nm][0]
            delta = sub - expected_sub
            marker = f" (expected {expected_sub:.2f}, delta {delta:+.2f})"
        print(f"  {nm:20s} {sub:10.3f} {cpx:10.3f}{marker}")

    results['cat_positions'] = {k: {'substrate': v[0], 'complexity': v[1]} for k, v in cat_positions.items()}

    # ================================================================
    # TEST 3: Correlation with ground truth placement
    # ================================================================
    print("\n"+"="*65)
    print("TEST 3: Correlation with expected placement")
    print("="*65)

    gt_sub, gt_cpx, meas_sub, meas_cpx = [], [], [], []
    for nm, (ms, mc) in cat_positions.items():
        if nm in SPECTRUM_PLACEMENT:
            es, ec = SPECTRUM_PLACEMENT[nm]
            gt_sub.append(es); gt_cpx.append(ec)
            meas_sub.append(ms); meas_cpx.append(mc)

    gt_sub, gt_cpx = np.array(gt_sub), np.array(gt_cpx)
    meas_sub, meas_cpx = np.array(meas_sub), np.array(meas_cpx)

    r_sub = np.corrcoef(gt_sub, meas_sub)[0, 1]
    r_cpx = np.corrcoef(gt_cpx, meas_cpx)[0, 1]
    print(f"  Substrate correlation:  r = {r_sub:.3f} ({'STRONG' if abs(r_sub) > 0.5 else 'WEAK'})")
    print(f"  Complexity correlation: r = {r_cpx:.3f} ({'STRONG' if abs(r_cpx) > 0.5 else 'WEAK'})")
    results['gt_correlations'] = {'substrate': float(r_sub), 'complexity': float(r_cpx)}

    # ================================================================
    # TEST 4: Are substrate and complexity INDEPENDENT axes?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 4: Are substrate and complexity independent?")
    print("="*65)

    r_axes = np.corrcoef(V[:, 0], V[:, 1])[0, 1]
    print(f"  Substrate-Complexity correlation: r = {r_axes:.3f}")
    print(f"  Independence: {'YES (|r| < 0.3)' if abs(r_axes) < 0.3 else 'PARTIAL (0.3 < |r| < 0.6)' if abs(r_axes) < 0.6 else 'NO (|r| > 0.6)'}")
    results['axis_independence'] = float(r_axes)

    # PCA on all properties — how many real axes?
    Vs = StandardScaler().fit_transform(V)
    pca = PCA()
    pca.fit(Vs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"  PCA: {n_90} dims for 90%, {n_95} dims for 95%")
    print(f"  Top eigenvalues: {pca.explained_variance_ratio_[:5].round(3)}")
    results['pca'] = {'dims_90': n_90, 'dims_95': n_95,
                      'top_5': [float(x) for x in pca.explained_variance_ratio_[:5]]}

    # ================================================================
    # TEST 5: The Firmware Zone — are insects really in between?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 5: The Firmware Zone (biological substrate + simple behavior)")
    print("="*65)

    # Quadrant analysis
    sub_med = np.median(V[:, 0])
    cpx_med = np.median(V[:, 1])
    quadrants = {
        'mechanical_simple': [],    # low sub, low cpx (clock, keyboard)
        'mechanical_complex': [],   # low sub, high cpx (helicopter, siren)
        'biological_simple': [],    # high sub, low cpx (breathing, snoring) — FIRMWARE ZONE
        'biological_complex': [],   # high sub, high cpx (speech, dog) — SENTIENT ZONE
    }
    for i, nm in enumerate(catnames):
        s, c = V[i, 0], V[i, 1]
        if s < sub_med and c < cpx_med: quadrants['mechanical_simple'].append(nm)
        elif s < sub_med and c >= cpx_med: quadrants['mechanical_complex'].append(nm)
        elif s >= sub_med and c < cpx_med: quadrants['biological_simple'].append(nm)
        else: quadrants['biological_complex'].append(nm)

    for qname, items in quadrants.items():
        from collections import Counter
        counts = Counter(items).most_common(5)
        top = ", ".join(f"{c}({n})" for c, n in counts)
        print(f"  {qname:25s}: {len(items):4d} samples — top: {top}")

    # Where do insects specifically land?
    if 'insects' in c2i:
        ins_mask = ca == c2i['insects']
        if ins_mask.sum() > 0:
            ins_sub = V[ins_mask, 0].mean()
            ins_cpx = V[ins_mask, 1].mean()
            ins_fw = V[ins_mask, 9].mean()  # firmware_index
            print(f"\n  INSECTS: substrate={ins_sub:.3f}, complexity={ins_cpx:.3f}, firmware={ins_fw:.3f}")

            # Compare with pure mechanical and pure biological
            mech_cats = {'clock_tick', 'engine', 'vacuum_cleaner'}
            bio_cats = {'dog', 'cat', 'rooster', 'laughing'}

            mech_mask = np.array([nm in mech_cats for nm in catnames])
            bio_mask = np.array([nm in bio_cats for nm in catnames])

            if mech_mask.sum() > 0 and bio_mask.sum() > 0:
                mech_sub = V[mech_mask, 0].mean()
                bio_sub = V[bio_mask, 0].mean()
                # Where do insects fall as percentage between mechanical and biological?
                if bio_sub != mech_sub:
                    pct = (ins_sub - mech_sub) / (bio_sub - mech_sub) * 100
                    print(f"  Insects are {pct:.0f}% of the way from mechanical to biological")
                    results['insects_position'] = float(pct)

    # ================================================================
    # TEST 6: Does 2D spectrum explain more than 1D alive/dead?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 6: 2D spectrum vs 1D binary — which fits better?")
    print("="*65)

    from sklearn.linear_model import LogisticRegression
    from scripts.wc_sound_as_life import ALIVE_CATS, MECHANICAL_CATS

    alive_labels = np.array([1 if catnames[i] in ALIVE_CATS else
                             0 if catnames[i] in MECHANICAL_CATS else -1
                             for i in range(len(catnames))])
    mask = alive_labels >= 0

    if mask.sum() >= 50:
        # 1D: just substrate
        sc_1d = cross_val_score(LogisticRegression(max_iter=1000),
                                V[mask, 0:1], alive_labels[mask], cv=5, scoring='accuracy')
        # 2D: substrate + complexity
        sc_2d = cross_val_score(LogisticRegression(max_iter=1000),
                                V[mask, :2], alive_labels[mask], cv=5, scoring='accuracy')
        # All properties
        sc_all = cross_val_score(LogisticRegression(max_iter=1000),
                                 Vs[mask], alive_labels[mask], cv=5, scoring='accuracy')
        # Embeddings (from original life experiment)
        sc_emb = cross_val_score(LogisticRegression(max_iter=1000),
                                 Es[mask], alive_labels[mask], cv=5, scoring='accuracy')

        print(f"  1D (substrate only):  {sc_1d.mean():.1%}")
        print(f"  2D (sub + complexity): {sc_2d.mean():.1%}")
        print(f"  All 10 properties:    {sc_all.mean():.1%}")
        print(f"  Encoder embeddings:   {sc_emb.mean():.1%}")

        results['classification'] = {
            '1d_substrate': float(sc_1d.mean()),
            '2d_spectrum': float(sc_2d.mean()),
            'all_props': float(sc_all.mean()),
            'embeddings': float(sc_emb.mean()),
        }

        improvement = sc_2d.mean() - sc_1d.mean()
        print(f"\n  2D vs 1D improvement: {improvement:+.1%}")
        if improvement > 0.02:
            print("  → The second axis MATTERS. Life isn't 1D.")
        elif improvement > -0.02:
            print("  → Second axis is marginal. Substrate alone captures most of it.")
        else:
            print("  → Substrate alone is actually better. Complexity confuses the classifier.")

    # ================================================================
    # TEST 7: The sentience gradient
    # ================================================================
    print("\n"+"="*65)
    print("TEST 7: Sentience gradient (substrate * complexity)")
    print("="*65)

    sentience_by_cat = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        mask_c = ca == c
        if mask_c.sum() == 0: continue
        sentience_by_cat[nm] = float(V[mask_c, 8].mean())  # sentience_proxy

    sorted_sent = sorted(sentience_by_cat.items(), key=lambda x: x[1])
    print("\n  SENTIENCE GRADIENT (low → high):")
    for nm, s in sorted_sent[:5]:
        print(f"    {nm:20s}: {s:.3f}")
    print(f"    {'...':20s}")
    for nm, s in sorted_sent[-5:]:
        print(f"    {nm:20s}: {s:.3f}")

    results['sentience_gradient'] = sentience_by_cat

    # Verdict
    sr = sum(1 for v in t1.values() if v > 0.3)
    axis_indep = abs(r_axes) < 0.5
    gt_match = abs(r_sub) > 0.3 and abs(r_cpx) > 0.3

    if sr >= 5 and axis_indep and gt_match:
        verdict = "STRONG"
    elif sr >= 3 or (axis_indep and gt_match):
        verdict = "PARTIAL"
    else:
        verdict = "WEAK"

    print(f"\nVERDICT: {verdict} (R-sq>0.3: {sr}/10, axes independent: {axis_indep}, GT match: {gt_match})")
    results['verdict'] = verdict

    json.dump(results, open('data/wc_life_machine_spectrum_results.json', 'w'), indent=2,
              default=lambda x: float(x) if hasattr(x, 'item') else x)
    print("Saved")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    a = p.parse_args(); run(a.data_dir, a.checkpoint)
