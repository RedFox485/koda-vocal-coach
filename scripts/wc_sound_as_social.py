#!/usr/bin/env python3
"""
Wildcard: Sound as Social Status
==================================
Can audio embeddings predict social-analog properties?

Evolutionary psychology shows sound carries social information:
  - dominance: low pitch + loud + sustained (large body = low resonance)
  - threat: sudden onset + rising energy + harsh timbre
  - attractiveness: harmonic richness + moderate complexity + smooth
  - trustworthiness: predictability + warmth + moderate pace
  - urgency: fast tempo + high energy + rising pitch
  - intimacy: quiet + close (high freq preserved) + warm
  - social_size: spectral width (crowd = wide, individual = narrow)

These are the sonic cues that evolved over millions of years.
Does our encoder discover them?
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_social(mel_frames):
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
        flux = np.mean(np.abs(np.diff(fe))) / (np.mean(fe) + 1e-8)
    else:
        flux = 0.0
    if T >= 4:
        am = np.std(fe) / (np.mean(fe) + 1e-8)
    else:
        am = 0.0

    # DOMINANCE: low pitch + loud + sustained
    loudness = np.log1p(rms)
    if T >= 4:
        sustain = 1.0 - flux  # low flux = sustained
    else:
        sustain = 0.5
    dominance = 0.35 * low_r + 0.35 * np.clip(loudness/10, 0, 1) + 0.3 * sustain

    # THREAT: sudden onset + rising energy + harsh
    if T >= 4:
        onset_slice = fe[:max(T//3, 2)]
        onset_diffs = np.diff(onset_slice)
        onset_rate = (np.max(onset_diffs) / (np.mean(fe) + 1e-8)) if len(onset_diffs) > 0 else 0.0
        energy_trend = np.polyfit(np.arange(T), fe, 1)[0]  # slope
        rising = max(energy_trend / (np.mean(fe) + 1e-8), 0)
    else:
        onset_rate = 0.0; rising = 0.0
    threat = 0.3 * np.clip(onset_rate/5, 0, 1) + 0.3 * np.clip(rising, 0, 1) + 0.4 * flatness

    # ATTRACTIVENESS: harmonic + moderate complexity + smooth
    spec_ac = np.correlate(ms - ms.mean(), ms - ms.mean(), 'full')
    spec_ac = spec_ac[len(spec_ac)//2:]
    harmonicity = max(spec_ac[1:].max() / (spec_ac[0] + 1e-8), 0) if len(spec_ac) > 1 else 0
    moderate_complexity = 1.0 - abs(entropy - 0.5) * 2  # peaks at 0.5
    attractiveness = 0.4 * harmonicity + 0.3 * smoothness + 0.3 * moderate_complexity

    # TRUSTWORTHINESS: predictability + warmth + moderate pace
    if T >= 4:
        predictability = 1.0 - np.std(np.diff(fe)) / (np.mean(np.abs(np.diff(fe))) + 1e-8)
        predictability = max(predictability, 0)
    else:
        predictability = 0.5
    trustworthiness = 0.35 * predictability + 0.35 * low_r + 0.3 * (1 - flux)

    # URGENCY: fast tempo + high energy + rising
    urgency = 0.35 * flux + 0.35 * np.clip(rms/1000, 0, 1) + 0.3 * centroid

    # INTIMACY: quiet + close (high freq preserved) + warm
    quietness = 1.0 / (rms + 1)
    intimacy = 0.35 * quietness + 0.35 * high_r + 0.3 * low_r

    # SOCIAL SIZE: spectral width (crowd=wide, individual=narrow)
    social_size = bw + entropy * 0.5  # wide bandwidth + high entropy = crowd

    return {
        'dominance': float(dominance),
        'threat': float(np.clip(threat, 0, 1)),
        'attractiveness': float(attractiveness),
        'trustworthiness': float(np.clip(trustworthiness, 0, 1)),
        'urgency': float(urgency),
        'intimacy': float(intimacy),
        'social_size': float(social_size),
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
    mels,socials,catids=[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); socials.append(compute_social(m)); catids.append(c2i[cats[idx]])
    print(f"Samples: {len(mels)}")
    pn=['dominance','threat','attractiveness','trustworthiness','urgency','intimacy','social_size']
    V=np.array([[s[p] for p in pn] for s in socials])
    print("\nSocial stats:")
    for i,name in enumerate(pn): print(f"  {name:18s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results={}
    print("\n"+"="*60+"\nTEST 1: Can embeddings predict social properties?\n"+"="*60)
    t1={}
    for i,name in enumerate(pn):
        y=V[:,i]
        if np.std(y)<1e-10: t1[name]=0.0; continue
        sc=cross_val_score(Ridge(alpha=1.0),Es,y,cv=5,scoring='r2')
        r2=sc.mean(); t1[name]=float(r2)
        mk="***" if r2>0.5 else "**" if r2>0.3 else "*" if r2>0.1 else ""
        print(f"  {name:18s}: R-sq = {r2:.3f} {mk}")
    results['test1']=t1

    print("\n"+"="*60+"\nTEST 2: Binary social classification\n"+"="*60)
    t2={}
    for name in pn:
        idx=pn.index(name); y=(V[:,idx]>np.median(V[:,idx])).astype(int)
        sc=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Es,y,cv=5,scoring='accuracy')
        t2[name]=float(sc.mean())
        mk="***" if sc.mean()>0.75 else "**" if sc.mean()>0.65 else ""
        print(f"  {name:18s}: {sc.mean():.1%} {mk}")
    results['test2']=t2

    print("\n"+"="*60+"\nTEST 3: Social correlations\n"+"="*60)
    expected={('dominance','threat'):'positive',           # dominant = threatening
              ('attractiveness','trustworthiness'):'positive', # attractive = trustworthy (halo)
              ('urgency','intimacy'):'negative',           # urgent != intimate
              ('threat','intimacy'):'negative'}            # threatening != intimate
    corr=np.corrcoef(V.T); t3={}; matches=0
    for (p1,p2),exp in expected.items():
        i,j=pn.index(p1),pn.index(p2); r=corr[i,j]
        act='positive' if r>0 else 'negative'; match=act==exp; matches+=match
        t3[f"{p1}_vs_{p2}"]={'r':float(r),'match':bool(match)}
        print(f"  {p1:18s} vs {p2:18s}: r={r:+.3f} ({'MATCH' if match else 'MISS'})")
    print(f"  Score: {matches}/{len(expected)}")
    results['test3']=t3

    print("\n"+"="*60+"\nTEST 4: Social profile per sound\n"+"="*60)
    ca=np.array(catids)
    # Most dominant, most threatening, most attractive sounds
    for pi, name in enumerate(pn):
        vals = {cn[c]: V[ca==c, pi].mean() for c in sorted(set(catids))}
        s = sorted(vals.items(), key=lambda x: x[1])
        print(f"  {name:18s}: lowest={s[0][0]:18s}({s[0][1]:.3f})  highest={s[-1][0]:18s}({s[-1][1]:.3f})")
    results['test4'] = {}

    sr=sum(1 for v in t1.values() if v>0.3)
    gc=sum(1 for v in t2.values() if v>0.65)
    v="STRONG" if sr>=4 and matches>=3 else "PARTIAL" if sr>=2 or matches>=2 else "WEAK"
    print(f"\nVERDICT: {v} (R-sq>0.3: {sr}/7, binary>65%: {gc}/7, corr: {matches}/{len(expected)})")
    results['verdict']=v
    json.dump(results,open('data/wc_social_results.json','w'),indent=2)
    print("Saved")

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/training/mel/esc50')
    p.add_argument('--checkpoint',default='checkpoints/clap_distill_spread_a.pt')
    a=p.parse_args(); run(a.data_dir,a.checkpoint)
