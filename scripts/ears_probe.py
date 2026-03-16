#!/usr/bin/env python3
"""
Probe ALL EARS dimensions on 3 reference recordings to find which ones
actually discriminate easy (27) vs strained (29) vs head (30).
Shows all dimension values side-by-side.
"""
import sys, os, math, subprocess, tempfile
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARS_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(EARS_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import librosa
from mel_extractor import MelExtractor
from frequency_explorer import analyze_mel

SAMPLE_RATE = 44100

def load(path):
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(SAMPLE_RATE), tmp.name],
                   capture_output=True, check=True)
    audio, sr = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    return audio

def get_dims(audio):
    me = MelExtractor(sample_rate=SAMPLE_RATE)
    mel = me.extract_from_audio(audio)
    result = analyze_mel(mel)
    flat = {}
    for mod, d in result.get("modalities", {}).items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    flat[f"{mod}.{k}"] = float(v)
    return flat

print("Loading reference recordings...")
d27 = get_dims(load("/Users/daniel/Downloads/Macedonia St 27.m4a"))
d29 = get_dims(load("/Users/daniel/Downloads/Macedonia St 29.m4a"))
d30 = get_dims(load("/Users/daniel/Downloads/Macedonia St 30.m4a"))

# Find dimensions that separate 27 (easy) from 29 (strained)
keys = sorted(set(d27) | set(d29) | set(d30))

print(f"\n{'Dimension':<35} {'27 easy':>10} {'29 strain':>10} {'30 head':>10}  {'Δ29-27':>8}")
print("-" * 80)

diffs = []
for k in keys:
    v27 = d27.get(k, float('nan'))
    v29 = d29.get(k, float('nan'))
    v30 = d30.get(k, float('nan'))
    diff = abs(v29 - v27) if not (math.isnan(v27) or math.isnan(v29)) else 0
    diffs.append((diff, k, v27, v29, v30))

# Sort by largest difference between easy and strained
diffs.sort(reverse=True)

print("Top 30 most discriminating dimensions (easy vs strained):")
print(f"\n{'Dimension':<35} {'27 easy':>10} {'29 strain':>10} {'30 head':>10}  {'|Δ|':>8}")
print("-" * 80)
for diff, k, v27, v29, v30 in diffs[:30]:
    flag = " ← " if diff > 0.1 else ""
    print(f"{k:<35} {v27:>10.4f} {v29:>10.4f} {v30:>10.4f}  {diff:>8.4f}{flag}")
