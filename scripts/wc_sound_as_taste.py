#!/usr/bin/env python3
"""
Wildcard: Sound as Taste
=========================
Can audio embeddings predict gustatory-analog properties?

Research shows reliable sound-taste crossmodal correspondences:
  - Sweet: smooth, consonant, low-mid frequency, legato (Crisinel & Spence 2010)
  - Sour: sharp, dissonant, high frequency, staccato
  - Bitter: rough, low, dark, complex
  - Salty: grainy, noisy, mid-frequency, crunchy texture
  - Umami: rich, full-spectrum, warm, sustained

If frequency is everything, these mappings should emerge from
pure spectral structure without any taste training.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_taste_properties(mel_frames):
    ml = np.exp(mel_frames)
    n = ml.shape[1]; T = ml.shape[0]
    fb = np.linspace(0, 1, n)
    ms = ml.mean(axis=0) + 1e-8
    sn = ms / (ms.sum() + 1e-8)
    fe = np.sum(ml ** 2, axis=1)
    centroid = np.sum(fb * ms) / np.sum(ms)
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)
    rms = np.sqrt(np.mean(ml ** 2))
    third = n // 3
    low_r = np.sum(ms[:third]) / (np.sum(ms) + 1e-8)
    mid_r = np.sum(ms[third:2*third]) / (np.sum(ms) + 1e-8)
    sd = np.abs(np.diff(ms))
    smoothness = 1.0 - (np.mean(sd) / (np.max(sd) + 1e-8))
    bw = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))
    if T >= 2:
        flux = np.mean(np.abs(np.diff(fe)))/(np.mean(fe)+1e-8)
    else:
        flux = 0.0
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)

    # SWEET: smooth + consonant + mid-freq + sustained
    sweet = 0.3*smoothness + 0.3*mid_r + 0.2*(1-flatness) + 0.2*(1-flux)

    # SOUR: sharp + high freq + staccato + dissonant
    sour = 0.3*centroid + 0.3*flux + 0.2*flatness + 0.2*(1-smoothness)

    # BITTER: rough + low + dark + complex
    bitter = 0.3*flatness + 0.3*low_r + 0.2*(1-centroid) + 0.2*entropy

    # SALTY: grainy/noisy + mid-freq + crunchy
    if T >= 4:
        am = np.std(fe) / (np.mean(fe) + 1e-8)
    else:
        am = 0.0
    salty = 0.3*flatness + 0.3*mid_r + 0.2*am + 0.2*flux

    # UMAMI: rich + full-spectrum + warm + sustained
    umami = 0.3*entropy + 0.3*low_r + 0.2*(1-flux) + 0.2*rms/(rms+100)

    return {'sweet': float(sweet), 'sour': float(sour), 'bitter': float(bitter),
            'salty': float(salty), 'umami': float(umami)}


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
    mels,tastes,catids,vf=[],[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); tastes.append(compute_taste_properties(m)); catids.append(c2i[cats[idx]]); vf.append(mf.name)
    print(f"Samples: {len(mels)}")
    pn=['sweet','sour','bitter','salty','umami']
    V=np.array([[t[p] for p in pn] for t in tastes])
    print("\nTaste stats:")
    for i,name in enumerate(pn): print(f"  {name:8s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results={}
    print("\n"+"="*60+"\nTEST 1: Can embeddings predict taste?\n"+"="*60)
    t1={}
    for i,name in enumerate(pn):
        y=V[:,i]
        if np.std(y)<1e-10: t1[name]=0.0; continue
        sc=cross_val_score(Ridge(alpha=1.0),Es,y,cv=5,scoring='r2')
        r2=sc.mean(); t1[name]=float(r2)
        mk="***" if r2>0.5 else "**" if r2>0.3 else "*" if r2>0.1 else ""
        print(f"  {name:8s}: R-sq = {r2:.3f} {mk}")
    results['test1']=t1

    print("\n"+"="*60+"\nTEST 2: Binary taste classification\n"+"="*60)
    t2={}
    for name in pn:
        idx=pn.index(name)
        y=(V[:,idx]>np.median(V[:,idx])).astype(int)
        sc=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Es,y,cv=5,scoring='accuracy')
        t2[name]=float(sc.mean())
        mk="***" if sc.mean()>0.75 else "**" if sc.mean()>0.65 else ""
        print(f"  {name:8s}: {sc.mean():.1%} {mk}")
    results['test2']=t2

    print("\n"+"="*60+"\nTEST 3: Taste correlations\n"+"="*60)
    expected={('sweet','sour'):'negative',('sweet','umami'):'positive',('sour','bitter'):'positive',('salty','umami'):'positive'}
    corr=np.corrcoef(V.T); t3={}; matches=0
    for (p1,p2),exp in expected.items():
        i,j=pn.index(p1),pn.index(p2); r=corr[i,j]
        act='positive' if r>0 else 'negative'; match=act==exp; matches+=match
        t3[f"{p1}_vs_{p2}"]={'r':float(r),'match':bool(match)}
        print(f"  {p1:8s} vs {p2:8s}: r={r:+.3f} ({exp}, {'MATCH' if match else 'MISS'})")
    print(f"  Score: {matches}/{len(expected)}")
    results['test3']=t3

    print("\n"+"="*60+"\nTEST 4: What does each sound taste like?\n"+"="*60)
    ca=np.array(catids); t4={}
    for i,name in enumerate(pn):
        vals={cn[c]:V[ca==c,i].mean() for c in sorted(set(catids))}
        s=sorted(vals.items(),key=lambda x:x[1])
        t4[name]={'least':s[0][0],'most':s[-1][0]}
        print(f"  {name:8s}: least={s[0][0]:18s}({s[0][1]:.3f})  most={s[-1][0]:18s}({s[-1][1]:.3f})")
    results['test4']=t4

    sr=sum(1 for v in t1.values() if v>0.3)
    gc=sum(1 for v in t2.values() if v>0.65)
    v="STRONG" if sr>=3 and matches>=3 else "PARTIAL" if sr>=2 or matches>=2 else "WEAK"
    print(f"\nVERDICT: {v} (R-sq>0.3: {sr}/5, binary>65%: {gc}/5, corr: {matches}/{len(expected)})")
    results['verdict']=v
    json.dump(results,open('data/wc_taste_results.json','w'),indent=2)
    print("Saved to data/wc_taste_results.json")

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/training/mel/esc50')
    p.add_argument('--checkpoint',default='checkpoints/clap_distill_spread_a.pt')
    a=p.parse_args(); run(a.data_dir,a.checkpoint)
