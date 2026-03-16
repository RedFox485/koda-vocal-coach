#!/usr/bin/env python3
"""
Wildcard: Sound as Weather
============================
Weather IS pressure waves and thermal gradients. Sound IS pressure waves.
They're the same physics at different scales.

Weather-analog properties:
  - temperature: spectral centroid (warm air = low pressure = low freq)
  - pressure: total energy (high pressure = loud)
  - humidity: spectral diffusion (humid = blurred, dry = crisp)
  - wind_speed: spectral flux rate (fast change = windy)
  - storm_intensity: energy variance * spectral complexity
  - cloud_cover: high-frequency absorption (clouds absorb highs)
  - precipitation: transient density (rain = many small impacts)

ESC-50 actually HAS weather sounds (rain, thunderstorm, wind).
Do they map to the "correct" weather properties?
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_weather(mel_frames):
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
    rms = np.sqrt(np.mean(ml ** 2))
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    geo = np.exp(np.mean(np.log(ms + 1e-10)))
    flatness = geo / (np.mean(ms) + 1e-8)

    # Temperature: warm = low centroid (warm air masses = low pressure = bass)
    temperature = 1.0 - centroid

    # Pressure: total energy (high pressure systems = dense = loud)
    pressure = np.log1p(rms)

    # Humidity: spectral blur / smoothness (humid air absorbs and smears)
    sd = np.abs(np.diff(ms))
    humidity = 1.0 - (np.mean(sd) / (np.max(sd) + 1e-8))

    # Wind speed: spectral flux (fast spectral change = wind)
    if T >= 2:
        wind_speed = np.mean(np.sqrt(np.sum(np.diff(ml, axis=0)**2, axis=1))) / (rms + 1e-8)
    else:
        wind_speed = 0.0

    # Storm intensity: energy variance * spectral complexity
    storm_intensity = np.std(fe) / (np.mean(fe) + 1e-8) * entropy

    # Cloud cover: high-frequency absorption ratio
    cloud_cover = 1.0 - high_r  # clouds absorb highs

    # Precipitation: transient density (rain = many small energy spikes)
    if T >= 4:
        ediff = np.abs(np.diff(fe))
        threshold = np.mean(ediff) + 0.5 * np.std(ediff)
        precipitation = np.sum(ediff > threshold) / T
    else:
        precipitation = 0.0

    return {
        'temperature': float(temperature),
        'pressure': float(pressure),
        'humidity': float(humidity),
        'wind_speed': float(np.clip(wind_speed, 0, 5)),
        'storm_intensity': float(storm_intensity),
        'cloud_cover': float(cloud_cover),
        'precipitation': float(precipitation),
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
    mels,weathers,catids,catnames_list=[],[],[],[]
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p:int(p.stem)):
        m=np.load(mf)
        if m.shape[0]<4: continue
        idx=int(mf.stem)
        if idx>=len(cats) or cats[idx] in ('ambient','mixed','music'): continue
        mels.append(m); weathers.append(compute_weather(m))
        catids.append(c2i[cats[idx]]); catnames_list.append(cats[idx])
    print(f"Samples: {len(mels)}")
    pn=['temperature','pressure','humidity','wind_speed','storm_intensity','cloud_cover','precipitation']
    V=np.array([[w[p] for p in pn] for w in weathers])
    print("\nWeather stats:")
    for i,name in enumerate(pn): print(f"  {name:18s}: mean={V[:,i].mean():.3f} std={V[:,i].std():.3f}")

    enc=load_encoder(cp,dev); embs=[]
    with torch.no_grad():
        for m in mels:
            x=torch.tensor(m,dtype=torch.float32).unsqueeze(0).to(dev)
            embs.append(enc(x).mean(dim=1).squeeze(0).cpu().numpy())
    E=np.array(embs); Es=StandardScaler().fit_transform(E)

    results={}
    print("\n"+"="*60+"\nTEST 1: Can embeddings predict weather?\n"+"="*60)
    t1={}
    for i,name in enumerate(pn):
        y=V[:,i]
        if np.std(y)<1e-10: t1[name]=0.0; continue
        sc=cross_val_score(Ridge(alpha=1.0),Es,y,cv=5,scoring='r2')
        r2=sc.mean(); t1[name]=float(r2)
        mk="***" if r2>0.5 else "**" if r2>0.3 else "*" if r2>0.1 else ""
        print(f"  {name:18s}: R-sq = {r2:.3f} {mk}")
    results['test1']=t1

    print("\n"+"="*60+"\nTEST 2: Binary weather classification\n"+"="*60)
    t2={}
    for name in pn:
        idx=pn.index(name); y=(V[:,idx]>np.median(V[:,idx])).astype(int)
        sc=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Es,y,cv=5,scoring='accuracy')
        t2[name]=float(sc.mean())
        mk="***" if sc.mean()>0.75 else "**" if sc.mean()>0.65 else ""
        print(f"  {name:18s}: {sc.mean():.1%} {mk}")
    results['test2']=t2

    print("\n"+"="*60+"\nTEST 3: Weather correlations\n"+"="*60)
    expected={('temperature','pressure'):'positive',      # warm = high pressure (anticyclone)
              ('wind_speed','storm_intensity'):'positive', # wind = stormy
              ('humidity','cloud_cover'):'positive',       # humid = cloudy
              ('precipitation','storm_intensity'):'positive'}
    corr=np.corrcoef(V.T); t3={}; matches=0
    for (p1,p2),exp in expected.items():
        i,j=pn.index(p1),pn.index(p2); r=corr[i,j]
        act='positive' if r>0 else 'negative'; match=act==exp; matches+=match
        t3[f"{p1}_vs_{p2}"]={'r':float(r),'match':bool(match)}
        print(f"  {p1:18s} vs {p2:18s}: r={r:+.3f} ({'MATCH' if match else 'MISS'})")
    print(f"  Score: {matches}/{len(expected)}")
    results['test3']=t3

    # SPECIAL: Do actual weather sounds map correctly?
    print("\n"+"="*60+"\nTEST 4: Do weather sounds map to correct weather?\n"+"="*60)
    weather_cats = {'rain', 'thunderstorm', 'wind', 'sea_waves', 'crackling_fire'}
    ca=np.array(catids)
    for wc in sorted(weather_cats):
        if wc in c2i:
            mask = ca == c2i[wc]
            if mask.sum() > 0:
                vals = V[mask].mean(axis=0)
                print(f"  {wc:18s}: temp={vals[0]:.2f} pres={vals[1]:.2f} humid={vals[2]:.2f} "
                      f"wind={vals[3]:.2f} storm={vals[4]:.2f} cloud={vals[5]:.2f} precip={vals[6]:.2f}")

    # Sanity checks:
    print("\n  Sanity checks (should be true):")
    checks = []
    if 'thunderstorm' in c2i and 'clock_tick' in c2i:
        ts = V[ca==c2i['thunderstorm']].mean(axis=0)
        ct = V[ca==c2i['clock_tick']].mean(axis=0)
        storm_check = ts[pn.index('storm_intensity')] > ct[pn.index('storm_intensity')]
        checks.append(storm_check)
        print(f"  Thunder stormier than clock_tick: {storm_check} "
              f"({ts[pn.index('storm_intensity')]:.3f} vs {ct[pn.index('storm_intensity')]:.3f})")
    if 'rain' in c2i and 'engine' in c2i:
        rn = V[ca==c2i['rain']].mean(axis=0)
        en = V[ca==c2i['engine']].mean(axis=0)
        precip_check = rn[pn.index('precipitation')] > en[pn.index('precipitation')]
        checks.append(precip_check)
        print(f"  Rain more precipitation than engine: {precip_check} "
              f"({rn[pn.index('precipitation')]:.3f} vs {en[pn.index('precipitation')]:.3f})")
    if 'wind' in c2i and 'snoring' in c2i:
        wi = V[ca==c2i['wind']].mean(axis=0)
        sn = V[ca==c2i['snoring']].mean(axis=0)
        wind_check = wi[pn.index('wind_speed')] > sn[pn.index('wind_speed')]
        checks.append(wind_check)
        print(f"  Wind windier than snoring: {wind_check} "
              f"({wi[pn.index('wind_speed')]:.3f} vs {sn[pn.index('wind_speed')]:.3f})")

    sanity_score = sum(checks)
    print(f"  Sanity score: {sanity_score}/{len(checks)}")
    results['sanity_checks'] = {'passed': sanity_score, 'total': len(checks)}

    sr=sum(1 for v in t1.values() if v>0.3)
    gc=sum(1 for v in t2.values() if v>0.65)
    v="STRONG" if sr>=4 and matches>=3 else "PARTIAL" if sr>=2 or matches>=2 else "WEAK"
    print(f"\nVERDICT: {v} (R-sq>0.3: {sr}/7, binary>65%: {gc}/7, corr: {matches}/{len(expected)}, sanity: {sanity_score}/{len(checks)})")
    results['verdict']=v
    json.dump(results,open('data/wc_weather_results.json','w'),indent=2)
    print("Saved")

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/training/mel/esc50')
    p.add_argument('--checkpoint',default='checkpoints/clap_distill_spread_a.pt')
    a=p.parse_args(); run(a.data_dir,a.checkpoint)
