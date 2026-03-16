#!/usr/bin/env python3
"""
Wildcard: Finding the "Alive" Direction in Latent Space
========================================================
Instead of hand-crafting life/machine features, ask the encoder directly:
What direction in 128-dim embedding space separates alive from mechanical?

This uses:
1. LDA to find the optimal separating hyperplane
2. PCA of the alive-mechanical difference vector
3. Probing: what acoustic properties correlate with the "alive direction"?
4. The full spectrum: project ALL categories onto the alive axis

If the encoder learned a meaningful alive/mechanical distinction through
CLAP distillation (which was trained on audio-text pairs like "dog barking"
vs "engine running"), the alive direction should emerge naturally.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


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
# The in-between categories — neither clearly alive nor mechanical
AMBIGUOUS_CATS = {
    'rain', 'thunderstorm', 'wind', 'sea_waves', 'crackling_fire',
    'water_drops', 'pouring_water', 'fireworks', 'church_bells',
    'door_wood_creaks', 'door_wood_knock', 'glass_breaking',
    'footsteps', 'clapping', 'can_opening', 'toilet_flush',
    'brushing_teeth', 'car_horn',
}


def compute_spectral_features(mel_frames):
    """Compute interpretable spectral features for probing."""
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
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)
    bw = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))

    features = {'centroid': centroid, 'rms': np.log1p(rms), 'low_ratio': low_r,
                'entropy': entropy, 'flatness': flatness, 'bandwidth': bw}

    if T >= 4:
        # Temporal features — these might be what separates alive from mechanical
        flux = np.mean(np.abs(np.diff(fe))) / (np.mean(fe) + 1e-8)
        energy_var = np.std(fe) / (np.mean(fe) + 1e-8)
        # Regularity of energy changes (mechanical = regular, alive = irregular)
        diffs = np.abs(np.diff(fe))
        if np.mean(diffs) > 1e-10:
            irregularity = np.std(diffs) / (np.mean(diffs) + 1e-8)
        else:
            irregularity = 0.0
        features['flux'] = flux
        features['energy_var'] = energy_var
        features['irregularity'] = irregularity

        # Spectral stability (how much does the spectrum change?)
        corrs = []
        for t in range(min(T-1, 30)):
            r = np.corrcoef(ml[t], ml[t+1])[0, 1]
            if not np.isnan(r): corrs.append(r)
        features['spectral_stability'] = np.mean(corrs) if corrs else 0.5

        # Onset sharpness (biological onsets are sharper, more varied)
        onset_diffs = np.diff(fe[:T//3]) if T >= 6 else np.diff(fe)
        features['onset_sharpness'] = np.max(onset_diffs) / (np.mean(fe) + 1e-8) if len(onset_diffs) > 0 else 0

        # Temporal autocorrelation (periodicity)
        ec = fe - fe.mean(); norm = np.sum(ec**2)
        if norm > 1e-10 and T >= 10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            features['periodicity'] = max(ac[1:].max(), 0) if len(ac) > 1 else 0
        else:
            features['periodicity'] = 0
    else:
        features.update({'flux': 0, 'energy_var': 0, 'irregularity': 0,
                        'spectral_stability': 0.5, 'onset_sharpness': 0, 'periodicity': 0})

    return features


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

    mels, feats, catids, catnames = [], [], [], []
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); feats.append(compute_spectral_features(m))
        catids.append(c2i[cats[idx]]); catnames.append(cats[idx])
    print(f"Samples: {len(mels)}")

    # Get embeddings
    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    # Labels
    labels = np.array([1 if catnames[i] in ALIVE_CATS else
                       0 if catnames[i] in MECHANICAL_CATS else -1
                       for i in range(len(catnames))])
    mask = labels >= 0
    print(f"Alive: {(labels==1).sum()}, Mechanical: {(labels==0).sum()}, Ambiguous: {(labels==-1).sum()}")

    results = {}

    # ================================================================
    # TEST 1: Encoder embeddings classify alive/mechanical
    # ================================================================
    print("\n"+"="*65)
    print("TEST 1: Encoder-based alive/mechanical classification")
    print("="*65)

    sc = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                         Es[mask], labels[mask], cv=5, scoring='accuracy')
    print(f"  Logistic Regression: {sc.mean():.1%} (+/- {sc.std():.1%})")

    # LDA — find the SINGLE direction that best separates alive/mechanical
    lda = LinearDiscriminantAnalysis()
    lda.fit(Es[mask], labels[mask])
    lda_acc = lda.score(Es[mask], labels[mask])
    print(f"  LDA (train):         {lda_acc:.1%}")
    sc_lda = cross_val_score(LinearDiscriminantAnalysis(), Es[mask], labels[mask], cv=5, scoring='accuracy')
    print(f"  LDA (5-fold CV):     {sc_lda.mean():.1%}")

    results['classification'] = {
        'logistic': float(sc.mean()),
        'lda_train': float(lda_acc),
        'lda_cv': float(sc_lda.mean())
    }

    # ================================================================
    # TEST 2: The alive direction — what IS it?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 2: The alive direction in embedding space")
    print("="*65)

    # LDA gives us the direction
    alive_direction = lda.coef_[0]  # 128-dim vector
    alive_direction = alive_direction / (np.linalg.norm(alive_direction) + 1e-8)

    # Project all samples onto the alive direction
    projections = Es @ alive_direction

    # Show mean projection per category
    ca = np.array(catids)
    cat_proj = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        cat_proj[nm] = float(projections[cmask].mean())

    sorted_proj = sorted(cat_proj.items(), key=lambda x: x[1])

    print("\n  MECHANICAL ←————————————————————————→ ALIVE")
    print(f"  {'Category':20s} {'Projection':>10s}  {'Group':>12s}")
    for nm, p in sorted_proj:
        group = "ALIVE" if nm in ALIVE_CATS else "MECHANICAL" if nm in MECHANICAL_CATS else "ambiguous"
        bar_pos = int((p - sorted_proj[0][1]) / (sorted_proj[-1][1] - sorted_proj[0][1] + 1e-8) * 30)
        bar = "." * bar_pos + "|"
        print(f"  {nm:20s} {p:+10.3f}  {group:>12s}  {bar}")

    results['alive_direction'] = cat_proj

    # How well does the alive direction separate?
    alive_projs = projections[labels == 1]
    mech_projs = projections[labels == 0]
    separation = (alive_projs.mean() - mech_projs.mean()) / np.sqrt(alive_projs.var() + mech_projs.var() + 1e-8)
    print(f"\n  Alive mean:     {alive_projs.mean():+.3f}")
    print(f"  Mechanical mean: {mech_projs.mean():+.3f}")
    print(f"  Separation (d'): {separation:.3f}")
    results['separation'] = float(separation)

    # ================================================================
    # TEST 3: What spectral features correlate with the alive direction?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 3: What IS the alive direction? (spectral correlates)")
    print("="*65)

    feat_names = list(feats[0].keys())
    F = np.array([[f[k] for k in feat_names] for f in feats])

    correlations = {}
    for i, fn in enumerate(feat_names):
        r = np.corrcoef(projections, F[:, i])[0, 1]
        if np.isnan(r): r = 0.0
        correlations[fn] = float(r)

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\n  Feature correlations with the alive direction:")
    for fn, r in sorted_corr:
        direction = "→ MORE alive" if r > 0 else "→ MORE mechanical"
        stars = "***" if abs(r) > 0.3 else "**" if abs(r) > 0.2 else "*" if abs(r) > 0.1 else ""
        print(f"    {fn:22s}: r = {r:+.3f} {stars:3s} {direction}")

    results['spectral_correlates'] = correlations

    # ================================================================
    # TEST 4: The ambiguous zone — where do in-between sounds land?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 4: The ambiguous zone")
    print("="*65)

    ambig_mask = labels == -1
    if ambig_mask.sum() > 0:
        ambig_projs = projections[ambig_mask]
        print(f"  Ambiguous samples: {ambig_mask.sum()}")
        print(f"  Mean projection:   {ambig_projs.mean():+.3f}")
        print(f"  Std:               {ambig_projs.std():.3f}")
        print(f"  Range:             [{ambig_projs.min():+.3f}, {ambig_projs.max():+.3f}]")

        # Which ambiguous categories lean alive vs mechanical?
        print("\n  Ambiguous categories on the alive axis:")
        ambig_cats = {}
        for nm in sorted(AMBIGUOUS_CATS):
            if nm in c2i:
                cmask = ca == c2i[nm]
                if cmask.sum() > 0:
                    p = projections[cmask].mean()
                    ambig_cats[nm] = float(p)
        for nm, p in sorted(ambig_cats.items(), key=lambda x: x[1]):
            lean = "← mechanical" if p < 0 else "→ alive"
            print(f"    {nm:20s}: {p:+.3f} {lean}")

        results['ambiguous_zone'] = ambig_cats

    # ================================================================
    # TEST 5: Is the alive direction orthogonal to other dimensions?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 5: Alive direction vs other cross-modal dimensions")
    print("="*65)

    # Load other property computations and check correlation with alive direction
    other_dims = {}

    # Size (from geometry) — correlate with centroid
    size_proxy = -F[:, feat_names.index('centroid')]  # low centroid = large
    r = np.corrcoef(projections, size_proxy)[0, 1]
    other_dims['size'] = float(r) if not np.isnan(r) else 0.0

    # Energy
    r = np.corrcoef(projections, F[:, feat_names.index('rms')])[0, 1]
    other_dims['energy'] = float(r) if not np.isnan(r) else 0.0

    # Complexity (entropy)
    r = np.corrcoef(projections, F[:, feat_names.index('entropy')])[0, 1]
    other_dims['complexity'] = float(r) if not np.isnan(r) else 0.0

    # Temporal variability
    r = np.corrcoef(projections, F[:, feat_names.index('energy_var')])[0, 1]
    other_dims['temporal_variability'] = float(r) if not np.isnan(r) else 0.0

    print("  Correlation of alive direction with other dimensions:")
    for dim, r in sorted(other_dims.items(), key=lambda x: abs(x[1]), reverse=True):
        orth = "ORTHOGONAL" if abs(r) < 0.1 else "weak overlap" if abs(r) < 0.3 else "OVERLAPPING"
        print(f"    {dim:25s}: r = {r:+.3f} ({orth})")

    results['orthogonality'] = other_dims

    # ================================================================
    # TEST 6: Multi-dimensional alive — is one direction enough?
    # ================================================================
    print("\n"+"="*65)
    print("TEST 6: Is one direction enough? (PCA of alive/mechanical centroids)")
    print("="*65)

    # Get centroid for each category
    centroids_alive = []
    centroids_mech = []
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        centroid_vec = Es[cmask].mean(axis=0)
        if nm in ALIVE_CATS:
            centroids_alive.append(centroid_vec)
        elif nm in MECHANICAL_CATS:
            centroids_mech.append(centroid_vec)

    all_centroids = np.array(centroids_alive + centroids_mech)
    cent_labels = np.array([1]*len(centroids_alive) + [0]*len(centroids_mech))

    # PCA of centroids
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(all_centroids)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1

    print(f"  Category centroids: {len(centroids_alive)} alive, {len(centroids_mech)} mechanical")
    print(f"  PCA dims for 90% variance: {n_90}/{len(all_centroids)}")
    print(f"  Top 5 eigenvalues: {pca.explained_variance_ratio_[:5].round(3)}")

    # How well does PC1 alone classify?
    pc1_proj = pca.transform(all_centroids)[:, 0]
    from sklearn.metrics import accuracy_score
    threshold = np.median(pc1_proj)
    pc1_pred = (pc1_proj > threshold).astype(int)
    pc1_acc = accuracy_score(cent_labels, pc1_pred)
    # Try flipping
    pc1_acc = max(pc1_acc, 1 - pc1_acc)
    print(f"  PC1 alone classification: {pc1_acc:.1%}")

    # PC1+PC2?
    if all_centroids.shape[0] > 3:
        sc_pca = cross_val_score(LogisticRegression(max_iter=1000),
                                  pca.transform(all_centroids)[:, :2], cent_labels,
                                  cv=min(3, len(all_centroids)//2), scoring='accuracy')
        print(f"  PC1+PC2 classification:   {sc_pca.mean():.1%}")
        results['centroid_pca'] = {'dims_90': n_90, 'pc1_acc': float(pc1_acc),
                                    'pc12_acc': float(sc_pca.mean())}

    # Verdict
    best_acc = max(sc.mean(), sc_lda.mean())
    if best_acc > 0.80:
        verdict = "STRONG"
    elif best_acc > 0.65:
        verdict = "PARTIAL"
    else:
        verdict = "WEAK"

    print(f"\nVERDICT: {verdict}")
    print(f"  Best classification: {best_acc:.1%}")
    print(f"  Separation (d'):     {separation:.3f}")
    print(f"  Top correlate:       {sorted_corr[0][0]} (r={sorted_corr[0][1]:+.3f})")
    results['verdict'] = verdict

    json.dump(results, open('data/wc_alive_direction_results.json', 'w'), indent=2)
    print("Saved")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    p.add_argument('--checkpoint', default='checkpoints/clap_distill_spread_a.pt')
    a = p.parse_args(); run(a.data_dir, a.checkpoint)
