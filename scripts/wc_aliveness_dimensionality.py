#!/usr/bin/env python3
"""
Wildcard: Aliveness as Dimensionality
======================================
Thesis: Alive sounds don't have MORE variation — they have variation
in MORE DIMENSIONS simultaneously.

A washing machine varies in one dimension (periodic energy).
A dog bark varies in many dimensions at once (pitch, energy, timbre,
timing, all changing independently).

Test: For each ESC-50 category, measure the EFFECTIVE DIMENSIONALITY
of temporal variation. Correlate with alive/mechanical labels.
If alive = more temporal dimensions, this should dramatically
improve classification beyond the 63% we got before.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ALIVE_CATS = {
    'dog', 'cat', 'rooster', 'pig', 'cow', 'frog', 'hen', 'insects',
    'sheep', 'crow', 'chirping_birds', 'crying_baby', 'sneezing',
    'coughing', 'laughing', 'breathing', 'snoring', 'drinking_sipping',
}
MECHANICAL_CATS = {
    'helicopter', 'chainsaw', 'engine', 'train', 'airplane',
    'vacuum_cleaner', 'washing_machine', 'clock_tick', 'clock_alarm',
    'keyboard_typing', 'mouse_click', 'hand_saw', 'siren'
}


def compute_temporal_dimensionality(mel_frames):
    """Measure the effective dimensionality of how this sound varies over time."""
    ml = np.exp(mel_frames)
    T, n = ml.shape
    if T < 5:
        return {}

    # PCA on frames: how many dimensions does this sound MOVE in?
    pca = PCA()
    pca.fit(ml)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Participation ratio = effective number of dimensions
    ev = pca.explained_variance_ratio_ + 1e-10
    participation_ratio = 1.0 / np.sum(ev**2)

    # Dims for 90% and 95% variance
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # Spectral entropy of eigenvalues (high = many equal dims, low = one dominant)
    ev_entropy = -np.sum(ev * np.log(ev + 1e-10)) / np.log(len(ev))

    # Frame-to-frame: are changes happening in one direction or many?
    if T >= 4:
        diffs = np.diff(ml, axis=0)  # (T-1, n)
        diff_pca = PCA()
        diff_pca.fit(diffs)
        diff_ev = diff_pca.explained_variance_ratio_ + 1e-10
        diff_participation = 1.0 / np.sum(diff_ev**2)
        diff_n90 = int(np.searchsorted(np.cumsum(diff_pca.explained_variance_ratio_), 0.90)) + 1
    else:
        diff_participation = 1.0
        diff_n90 = 1

    # Independence of changes across frequency bands
    if T >= 6:
        # Correlation matrix of mel band changes over time
        changes = np.diff(ml, axis=0)
        corr = np.corrcoef(changes.T)
        corr = np.nan_to_num(corr, nan=0.0)
        # Mean absolute off-diagonal correlation (low = independent changes)
        mask = ~np.eye(n, dtype=bool)
        mean_cross_corr = np.mean(np.abs(corr[mask]))
        independence = 1.0 - mean_cross_corr
    else:
        independence = 0.5

    # Multi-scale dimensionality (alive things vary at MULTIPLE timescales)
    if T >= 16:
        # Fast changes (frame-to-frame)
        fast_var = np.var(np.diff(ml, axis=0), axis=0)
        # Slow changes (4-frame averages)
        slow = np.array([ml[i:i+4].mean(axis=0) for i in range(0, T-3, 4)])
        slow_var = np.var(slow, axis=0) if len(slow) > 1 else np.zeros(n)
        # Ratio: do fast and slow changes use different dimensions?
        fast_dim = fast_var / (fast_var.sum() + 1e-8)
        slow_dim = slow_var / (slow_var.sum() + 1e-8)
        # KL divergence between fast and slow spectral profiles
        multi_scale_diversity = np.sum(fast_dim * np.log((fast_dim + 1e-10) / (slow_dim + 1e-10)))
        multi_scale_diversity = max(0, float(multi_scale_diversity))
    else:
        multi_scale_diversity = 0.0

    return {
        'participation_ratio': float(participation_ratio),
        'dims_90': int(n_90),
        'dims_95': int(n_95),
        'ev_entropy': float(ev_entropy),
        'diff_participation': float(diff_participation),
        'diff_dims_90': int(diff_n90),
        'independence': float(independence),
        'multi_scale_diversity': float(multi_scale_diversity),
        'top_eigenvalue_ratio': float(pca.explained_variance_ratio_[0]),
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

    mels, dims, catids, catnames = [], [], [], []
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<5: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        d = compute_temporal_dimensionality(m)
        if not d: continue
        mels.append(m); dims.append(d)
        catids.append(c2i[cats[idx]]); catnames.append(cats[idx])
    print(f"Samples: {len(mels)}")

    prop_names = list(dims[0].keys())
    V = np.array([[d[p] for p in prop_names] for d in dims])

    print("\nDimensionality stats:")
    for i, name in enumerate(prop_names):
        print(f"  {name:25s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    # Labels
    labels = np.array([1 if catnames[i] in ALIVE_CATS else
                       0 if catnames[i] in MECHANICAL_CATS else -1
                       for i in range(len(catnames))])
    mask = labels >= 0

    results = {}

    # ================================================================
    # TEST 1: Do alive sounds have more temporal dimensions?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 1: Temporal dimensionality — alive vs mechanical")
    print("="*65)

    alive_mask = labels == 1
    mech_mask = labels == 0

    print(f"\n  {'Property':25s} {'Alive':>8s} {'Mech':>8s} {'Delta':>8s} {'p-value':>8s}")
    t1 = {}
    for i, name in enumerate(prop_names):
        alive_vals = V[alive_mask, i]
        mech_vals = V[mech_mask, i]
        delta = alive_vals.mean() - mech_vals.mean()

        # Simple t-test approximation
        se = np.sqrt(alive_vals.var()/len(alive_vals) + mech_vals.var()/len(mech_vals))
        t_stat = delta / (se + 1e-8)
        # Approximate p-value (two-tailed, normal approx)
        from scipy import stats
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {name:25s} {alive_vals.mean():8.3f} {mech_vals.mean():8.3f} {delta:+8.3f} {p_val:8.4f} {stars}")
        t1[name] = {'alive': float(alive_vals.mean()), 'mech': float(mech_vals.mean()),
                    'delta': float(delta), 'p': float(p_val)}
    results['test1'] = t1

    # ================================================================
    # TEST 2: Classification using dimensionality features
    # ================================================================
    print("\n"+"="*65)
    print("TEST 2: Alive/mechanical classification")
    print("="*65)

    Vs = StandardScaler().fit_transform(V)

    # Dimensionality features only
    sc_dim = cross_val_score(LogisticRegression(max_iter=1000),
                              Vs[mask], labels[mask], cv=5, scoring='accuracy')
    print(f"  Dimensionality features: {sc_dim.mean():.1%}")

    # Encoder embeddings
    enc = load_encoder(cp, dev); embs = []
    with torch.no_grad():
        for m in mels:
            x = torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    Es = StandardScaler().fit_transform(np.array(embs))

    sc_enc = cross_val_score(LogisticRegression(max_iter=1000),
                              Es[mask], labels[mask], cv=5, scoring='accuracy')
    print(f"  Encoder embeddings:      {sc_enc.mean():.1%}")

    # Combined: encoder + dimensionality
    combined = np.hstack([Es, Vs])
    sc_combo = cross_val_score(LogisticRegression(max_iter=1000),
                                combined[mask], labels[mask], cv=5, scoring='accuracy')
    print(f"  Combined (enc + dims):   {sc_combo.mean():.1%}")

    improvement = sc_combo.mean() - sc_enc.mean()
    print(f"\n  Improvement from adding dimensionality: {improvement:+.1%}")
    if improvement > 0.03:
        print("  → Dimensionality adds NEW information the encoder doesn't capture!")
    elif improvement > 0:
        print("  → Small improvement — some complementary signal.")
    else:
        print("  → No improvement — encoder already captures dimensionality.")

    results['test2'] = {
        'dim_only': float(sc_dim.mean()),
        'encoder': float(sc_enc.mean()),
        'combined': float(sc_combo.mean()),
        'improvement': float(improvement),
    }

    # ================================================================
    # TEST 3: Per-category dimensionality profile
    # ================================================================
    print("\n"+"="*65)
    print("TEST 3: Dimensionality profile per category")
    print("="*65)

    ca = np.array(catids)
    cat_dims = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        part_ratio = V[cmask, prop_names.index('participation_ratio')].mean()
        diff_part = V[cmask, prop_names.index('diff_participation')].mean()
        indep = V[cmask, prop_names.index('independence')].mean()
        cat_dims[nm] = (float(part_ratio), float(diff_part), float(indep))

    # Sort by participation ratio
    sorted_dims = sorted(cat_dims.items(), key=lambda x: x[1][0])
    print(f"\n  {'Category':20s} {'Part.Ratio':>10s} {'Diff.Part':>10s} {'Indep':>8s} {'Group':>12s}")
    for nm, (pr, dp, ind) in sorted_dims:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"  {nm:20s} {pr:10.2f} {dp:10.2f} {ind:8.3f} {group:>12s}")

    results['test3'] = {nm: {'participation': v[0], 'diff_part': v[1], 'independence': v[2]}
                        for nm, v in cat_dims.items()}

    # ================================================================
    # TEST 4: The expansion ratio — alive sounds expand into HOW MANY more dims?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 4: The expansion ratio")
    print("="*65)

    alive_part = V[alive_mask, prop_names.index('participation_ratio')].mean()
    mech_part = V[mech_mask, prop_names.index('participation_ratio')].mean()
    expansion_ratio = alive_part / (mech_part + 1e-8)

    alive_diff = V[alive_mask, prop_names.index('diff_participation')].mean()
    mech_diff = V[mech_mask, prop_names.index('diff_participation')].mean()
    diff_ratio = alive_diff / (mech_diff + 1e-8)

    print(f"  Alive participation ratio:      {alive_part:.2f}")
    print(f"  Mechanical participation ratio:  {mech_part:.2f}")
    print(f"  EXPANSION RATIO:                 {expansion_ratio:.2f}x")
    print(f"\n  Alive change dimensions:         {alive_diff:.2f}")
    print(f"  Mechanical change dimensions:    {mech_diff:.2f}")
    print(f"  Change expansion ratio:          {diff_ratio:.2f}x")

    results['test4'] = {
        'alive_participation': float(alive_part),
        'mech_participation': float(mech_part),
        'expansion_ratio': float(expansion_ratio),
        'diff_expansion_ratio': float(diff_ratio),
    }

    # Verdict
    dim_acc = sc_dim.mean()
    combo_acc = sc_combo.mean()
    sig_props = sum(1 for v in t1.values() if v['p'] < 0.05)

    if combo_acc > 0.72 and sig_props >= 5:
        verdict = "STRONG"
    elif combo_acc > 0.65 or sig_props >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "WEAK"

    print(f"\nVERDICT: {verdict}")
    print(f"  Classification: {combo_acc:.1%} (dim only: {dim_acc:.1%})")
    print(f"  Significant properties: {sig_props}/{len(prop_names)}")
    print(f"  Expansion ratio: {expansion_ratio:.2f}x")
    results['verdict'] = verdict

    json.dump(results, open('data/wc_aliveness_dimensionality_results.json', 'w'), indent=2)
    print("Saved")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    a = p.parse_args(); run(a.data_dir, a.checkpoint)
