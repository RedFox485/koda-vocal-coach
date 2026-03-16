#!/usr/bin/env python3
"""
Wildcard: Sound as Chemistry
==============================
IR spectroscopy literally works by measuring molecular vibration frequencies.
Every chemical bond has a resonant frequency. Frequency IS chemistry.

Chemical-analog properties from spectrograms:
  - molecular_weight: low freq dominance (heavy molecules vibrate slowly)
  - reactivity: transient energy / instability (reactive = unstable = changing)
  - volatility: high frequency energy (volatile = light = high freq)
  - bond_strength: spectral concentration (strong bond = narrow resonance)
  - polarity: spectral asymmetry (polar = asymmetric vibration pattern)
  - entropy_state: spectral disorder (high entropy = gas-like = disordered)
  - catalytic: how much the sound changes its spectral neighbors (context effect)
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_chemistry(mel_frames):
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    fb = np.linspace(0, 1, n)
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)
    centroid = np.sum(fb * ms) / np.sum(ms)
    third = n // 3
    low_r = np.sum(ms[:third]) / (np.sum(ms) + 1e-8)
    high_r = np.sum(ms[2*third:]) / (np.sum(ms) + 1e-8)
    pr = (sn.sum()**2) / (np.sum(sn**2) + 1e-10)
    concentration = 1.0 - (pr / n)
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)

    # Molecular weight: heavy = low freq dominant
    molecular_weight = low_r

    # Reactivity: how much energy changes over time (unstable)
    if T >= 3:
        reactivity = np.std(np.diff(fe)) / (np.mean(fe) + 1e-8)
    else:
        reactivity = 0.0

    # Volatility: high frequency energy (light, energetic)
    volatility = high_r + centroid

    # Bond strength: spectral concentration (narrow = strong single bond)
    bond_strength = concentration

    # Polarity: spectral asymmetry (skewness)
    if np.std(ms) > 1e-10:
        skew = np.mean(((ms - ms.mean()) / (np.std(ms) + 1e-8)) ** 3)
        polarity = abs(float(skew))
    else:
        polarity = 0.0

    # Entropy state: spectral disorder (gas=high, crystal=low)
    entropy_state = entropy

    # Catalytic: temporal influence — does this frame change the next?
    if T >= 4:
        # Cross-correlation between consecutive frame spectra
        frame_corrs = []
        for t in range(T - 1):
            r = np.corrcoef(ml[t], ml[t+1])[0, 1]
            if not np.isnan(r):
                frame_corrs.append(r)
        catalytic = 1.0 - np.mean(frame_corrs) if frame_corrs else 0.0
    else:
        catalytic = 0.0

    return {
        'molecular_weight': float(molecular_weight),
        'reactivity': float(np.clip(reactivity, 0, 5)),
        'volatility': float(volatility),
        'bond_strength': float(bond_strength),
        'polarity': float(polarity),
        'entropy_state': float(entropy_state),
        'catalytic': float(catalytic),
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
    mels,chems,catids,vf=[],[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); chems.append(compute_chemistry(m)); catids.append(c2i[cats[idx]]); vf.append(mf.name)
    print(f"Samples: {len(mels)}")
    pn=['molecular_weight','reactivity','volatility','bond_strength','polarity','entropy_state','catalytic']
    V=np.array([[c[p] for p in pn] for c in chems])
    print("\nChemistry stats:")
    for i,name in enumerate(pn): print(f"  {name:18s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results={}
    print("\n"+"="*60+"\nTEST 1: Can embeddings predict chemical properties?\n"+"="*60)
    t1={}
    for i,name in enumerate(pn):
        y=V[:,i]
        if np.std(y)<1e-10: t1[name]=0.0; continue
        sc=cross_val_score(Ridge(alpha=1.0),Es,y,cv=5,scoring='r2')
        r2=sc.mean(); t1[name]=float(r2)
        mk="***" if r2>0.5 else "**" if r2>0.3 else "*" if r2>0.1 else ""
        print(f"  {name:18s}: R-sq = {r2:.3f} {mk}")
    results['test1']=t1

    print("\n"+"="*60+"\nTEST 2: Binary chemical classification\n"+"="*60)
    t2={}
    tasks={'light_vs_heavy':('molecular_weight',lambda x:x>np.median(x)),
           'stable_vs_reactive':('reactivity',lambda x:x>np.median(x)),
           'solid_vs_volatile':('volatility',lambda x:x>np.median(x)),
           'weak_vs_strong_bond':('bond_strength',lambda x:x>np.median(x)),
           'nonpolar_vs_polar':('polarity',lambda x:x>np.median(x)),
           'ordered_vs_disordered':('entropy_state',lambda x:x>np.median(x))}
    for task,(prop,fn) in tasks.items():
        idx=pn.index(prop); y=fn(V[:,idx]).astype(int)
        sc=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Es,y,cv=5,scoring='accuracy')
        t2[task]=float(sc.mean())
        mk="***" if sc.mean()>0.75 else "**" if sc.mean()>0.65 else ""
        print(f"  {task:25s}: {sc.mean():.1%} {mk}")
    results['test2']=t2

    print("\n"+"="*60+"\nTEST 3: Chemical correlations\n"+"="*60)
    expected={('molecular_weight','volatility'):'negative',  # heavy = not volatile
              ('reactivity','catalytic'):'positive',          # reactive = changes neighbors
              ('bond_strength','entropy_state'):'negative',   # strong bonds = ordered
              ('polarity','volatility'):'negative'}           # polar = less volatile (H-bonds)
    corr=np.corrcoef(V.T); t3={}; matches=0
    for (p1,p2),exp in expected.items():
        i,j=pn.index(p1),pn.index(p2); r=corr[i,j]
        act='positive' if r>0 else 'negative'; match=act==exp; matches+=match
        t3[f"{p1}_vs_{p2}"]={'r':float(r),'match':bool(match)}
        print(f"  {p1:18s} vs {p2:18s}: r={r:+.3f} ({'MATCH' if match else 'MISS'})")
    print(f"  Score: {matches}/{len(expected)}")
    results['test3']=t3

    print("\n"+"="*60+"\nTEST 4: Chemical profile per sound\n"+"="*60)
    ca=np.array(catids); t4={}
    for i,name in enumerate(pn):
        vals={cn[c]:V[ca==c,i].mean() for c in sorted(set(catids))}
        s=sorted(vals.items(),key=lambda x:x[1])
        t4[name]={'least':s[0][0],'most':s[-1][0]}
        print(f"  {name:18s}: {s[0][0]:18s}({s[0][1]:.3f}) ... {s[-1][0]:18s}({s[-1][1]:.3f})")
    results['test4']=t4

    sr=sum(1 for v in t1.values() if v>0.3)
    gc=sum(1 for v in t2.values() if v>0.65)
    v="STRONG" if sr>=4 and matches>=3 else "PARTIAL" if sr>=2 or matches>=2 else "WEAK"
    print(f"\nVERDICT: {v} (R-sq>0.3: {sr}/7, binary>65%: {gc}/6, corr: {matches}/{len(expected)})")
    results['verdict']=v
    json.dump(results,open('data/wc_chemistry_results.json','w'),indent=2)
    print("Saved")

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/training/mel/esc50')
    p.add_argument('--checkpoint',default='checkpoints/clap_distill_spread_a.pt')
    a=p.parse_args(); run(a.data_dir,a.checkpoint)
