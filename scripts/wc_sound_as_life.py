#!/usr/bin/env python3
"""
Wildcard: Sound as Life
========================
The wildest question: can sound tell alive from dead?

Living systems have signatures:
  - organic: spectral irregularity + quasi-periodicity (heartbeat, breathing)
  - vitality: energy modulation depth (alive things MOVE, dead things are static)
  - growth: spectral evolution (growing = changing over time)
  - complexity: information content (life = organized complexity)
  - metabolism: energy throughput rate (fast metabolism = high energy flux)
  - homeostasis: stability under perturbation (returning to baseline)
  - reproduction: self-similarity at different time scales (fractal structure)

ESC-50 has both organic (dog, cat, rooster, frog, insects, birds) and
mechanical (engine, chainsaw, vacuum, helicopter) sounds. Can the encoder
tell them apart on life-analog dimensions?
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# Ground truth: which ESC-50 categories are "alive"
ALIVE_CATS = {
    'dog', 'cat', 'rooster', 'pig', 'cow', 'frog', 'hen', 'insects',
    'sheep', 'crow', 'chirping_birds', 'crying_baby', 'sneezing',
    'coughing', 'laughing', 'breathing', 'snoring', 'drinking_sipping',
    'speech'  # human = alive
}
MECHANICAL_CATS = {
    'helicopter', 'chainsaw', 'engine', 'train', 'airplane',
    'vacuum_cleaner', 'washing_machine', 'clock_tick', 'clock_alarm',
    'keyboard_typing', 'mouse_click', 'hand_saw', 'siren'
}


def compute_life_properties(mel_frames):
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)

    # 1. ORGANIC: spectral irregularity (not perfectly periodic or flat)
    if n >= 3:
        d2 = np.abs(np.diff(ms, n=2))
        irregularity = np.std(d2) / (np.mean(d2) + 1e-8)
    else:
        irregularity = 0.0
    # Quasi-periodicity: autocorrelation has peaks but not perfect
    if T >= 10:
        ec = fe - fe.mean(); norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            peaks = ac[1:].max() if len(ac) > 1 else 0
            quasi_periodic = peaks * (1 - peaks)  # max at 0.5 (partially periodic)
        else:
            quasi_periodic = 0.0
    else:
        quasi_periodic = 0.0
    organic = 0.5 * irregularity + 0.5 * quasi_periodic * 4  # scale to ~0-1

    # 2. VITALITY: energy modulation depth (alive = moving = modulated)
    if T >= 4:
        vitality = np.std(fe) / (np.mean(fe) + 1e-8)
    else:
        vitality = 0.0

    # 3. GROWTH: spectral change over time (growing = evolving)
    if T >= 4:
        first_half = ml[:T//2].mean(axis=0)
        second_half = ml[T//2:].mean(axis=0)
        growth = np.sqrt(np.sum((second_half - first_half)**2)) / (np.sqrt(np.sum(first_half**2)) + 1e-8)
    else:
        growth = 0.0

    # 4. COMPLEXITY: spectral entropy (organized complexity)
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    complexity = entropy

    # 5. METABOLISM: energy throughput (total spectral flux)
    if T >= 2:
        metabolism = np.mean(np.sqrt(np.sum(np.diff(ml, axis=0)**2, axis=1)))
    else:
        metabolism = 0.0

    # 6. HOMEOSTASIS: tendency to return to mean (negative autocorrelation of deviations)
    if T >= 6:
        deviations = fe - np.mean(fe)
        if len(deviations) >= 4:
            lag1_corr = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]
            homeostasis = max(-lag1_corr, 0)  # negative autocorr = mean-reverting
        else:
            homeostasis = 0.0
    else:
        homeostasis = 0.0

    # 7. REPRODUCTION: self-similarity across time scales
    if T >= 8:
        # Compare variance at different scales
        var_1 = np.var(fe)
        var_2 = np.var(fe[::2])  # downsampled 2x
        var_4 = np.var(fe[::4])  # downsampled 4x
        if var_1 > 1e-10 and var_2 > 1e-10:
            ratio_1 = var_2 / var_1
            ratio_2 = var_4 / (var_2 + 1e-10)
            reproduction = 1.0 - abs(ratio_1 - ratio_2)  # self-similar if ratios match
            reproduction = max(reproduction, 0)
        else:
            reproduction = 0.0
    else:
        reproduction = 0.0

    return {
        'organic': float(organic),
        'vitality': float(np.clip(vitality, 0, 5)),
        'growth': float(np.clip(growth, 0, 5)),
        'complexity': float(complexity),
        'metabolism': float(metabolism),
        'homeostasis': float(homeostasis),
        'reproduction': float(reproduction),
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
    mels,lifes,catids,catnames_list=[],[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); lifes.append(compute_life_properties(m))
        catids.append(c2i[cats[idx]]); catnames_list.append(cats[idx])
    print(f"Samples: {len(mels)}")
    pn=['organic','vitality','growth','complexity','metabolism','homeostasis','reproduction']
    V=np.array([[l[p] for p in pn] for l in lifes])
    print("\nLife stats:")
    for i,name in enumerate(pn): print(f"  {name:14s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results={}
    print("\n"+"="*60+"\nTEST 1: Can embeddings predict life properties?\n"+"="*60)
    t1={}
    for i,name in enumerate(pn):
        y=V[:,i]
        if np.std(y)<1e-10: t1[name]=0.0; continue
        sc=cross_val_score(Ridge(alpha=1.0),Es,y,cv=5,scoring='r2')
        r2=sc.mean(); t1[name]=float(r2)
        mk="***" if r2>0.5 else "**" if r2>0.3 else "*" if r2>0.1 else ""
        print(f"  {name:14s}: R-sq = {r2:.3f} {mk}")
    results['test1']=t1

    # THE BIG TEST: Can embeddings tell alive from mechanical?
    print("\n"+"="*60+"\nTEST 2: ALIVE vs MECHANICAL (the real test)\n"+"="*60)
    alive_labels = np.array([1 if catnames_list[i] in ALIVE_CATS else
                             0 if catnames_list[i] in MECHANICAL_CATS else -1
                             for i in range(len(catnames_list))])
    mask = alive_labels >= 0
    print(f"  Alive samples: {(alive_labels==1).sum()}, Mechanical: {(alive_labels==0).sum()}, "
          f"Ambiguous: {(alive_labels==-1).sum()}")

    if mask.sum() >= 100:
        # From embeddings
        sc_emb = cross_val_score(LogisticRegression(max_iter=1000,C=1.0),
                                  Es[mask], alive_labels[mask], cv=5, scoring='accuracy')
        # From life properties
        Vs = StandardScaler().fit_transform(V)
        sc_life = cross_val_score(LogisticRegression(max_iter=1000,C=1.0),
                                   Vs[mask], alive_labels[mask], cv=5, scoring='accuracy')
        # From raw spectrum
        raw_spec = np.array([np.exp(m).mean(axis=0) for m in mels])
        rs = StandardScaler().fit_transform(raw_spec)
        sc_raw = cross_val_score(LogisticRegression(max_iter=1000,C=1.0),
                                  rs[mask], alive_labels[mask], cv=5, scoring='accuracy')

        print(f"  Embeddings:      {sc_emb.mean():.1%}")
        print(f"  Life properties: {sc_life.mean():.1%}")
        print(f"  Raw spectrum:    {sc_raw.mean():.1%}")
        results['test2_alive_vs_mech'] = {
            'embeddings': float(sc_emb.mean()),
            'life_props': float(sc_life.mean()),
            'raw_spectrum': float(sc_raw.mean())
        }

    # TEST 3: Binary life classification
    print("\n"+"="*60+"\nTEST 3: Binary life classifications\n"+"="*60)
    t3={}
    for name in pn:
        idx=pn.index(name)
        y=(V[:,idx]>np.median(V[:,idx])).astype(int)
        sc=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Es,y,cv=5,scoring='accuracy')
        t3[name]=float(sc.mean())
        mk="***" if sc.mean()>0.75 else "**" if sc.mean()>0.65 else ""
        print(f"  {name:14s}: {sc.mean():.1%} {mk}")
    results['test3']=t3

    # TEST 4: Life profile per category
    print("\n"+"="*60+"\nTEST 4: Life profile per sound\n"+"="*60)
    ca=np.array(catids)
    # Show alive vs mechanical categories
    print("\n  ALIVE categories:")
    for c in sorted(set(catids)):
        nm = cn[c]
        if nm in ALIVE_CATS:
            vals = V[ca==c].mean(axis=0)
            print(f"    {nm:18s}: org={vals[0]:.2f} vit={vals[1]:.2f} grw={vals[2]:.2f} "
                  f"cpx={vals[3]:.2f} met={vals[4]:.1f} hom={vals[5]:.2f}")
    print("\n  MECHANICAL categories:")
    for c in sorted(set(catids)):
        nm = cn[c]
        if nm in MECHANICAL_CATS:
            vals = V[ca==c].mean(axis=0)
            print(f"    {nm:18s}: org={vals[0]:.2f} vit={vals[1]:.2f} grw={vals[2]:.2f} "
                  f"cpx={vals[3]:.2f} met={vals[4]:.1f} hom={vals[5]:.2f}")

    # Mean alive vs mechanical
    alive_mask = np.array([catnames_list[i] in ALIVE_CATS for i in range(len(catnames_list))])
    mech_mask = np.array([catnames_list[i] in MECHANICAL_CATS for i in range(len(catnames_list))])
    if alive_mask.sum() > 0 and mech_mask.sum() > 0:
        print(f"\n  Mean ALIVE:      {V[alive_mask].mean(axis=0).round(3)}")
        print(f"  Mean MECHANICAL: {V[mech_mask].mean(axis=0).round(3)}")
        diff = V[alive_mask].mean(axis=0) - V[mech_mask].mean(axis=0)
        print(f"  Difference:      {diff.round(3)}")
        print(f"  Largest gaps:    ", end="")
        sorted_gaps = sorted(zip(pn, diff), key=lambda x: abs(x[1]), reverse=True)
        for name, d in sorted_gaps[:3]:
            print(f"{name}={d:+.3f} ", end="")
        print()
        results['alive_vs_mech_gap'] = {n: float(d) for n, d in sorted_gaps}

    sr=sum(1 for v in t1.values() if v>0.3)
    gc=sum(1 for v in t3.values() if v>0.65)
    alive_acc = results.get('test2_alive_vs_mech', {}).get('embeddings', 0)
    v = "STRONG" if alive_acc > 0.85 and sr >= 3 else "PARTIAL" if alive_acc > 0.7 or sr >= 2 else "WEAK"
    print(f"\nVERDICT: {v} (R-sq>0.3: {sr}/7, alive/mech: {alive_acc:.1%})")
    results['verdict']=v
    json.dump(results,open('data/wc_life_results.json','w'),indent=2)
    print("Saved")

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/training/mel/esc50')
    p.add_argument('--checkpoint',default='checkpoints/clap_distill_spread_a.pt')
    a=p.parse_args(); run(a.data_dir,a.checkpoint)
