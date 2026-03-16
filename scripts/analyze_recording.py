#!/usr/bin/env python3
"""
Analyze a vocal recording using the exact same pipeline as the live backend.
Usage: python scripts/analyze_recording.py <audio_file>

Matches backend: 44100 Hz, MelExtractor, parselmouth, same strain formula.
"""
import sys
import os
import math
import numpy as np

# EARS paths — must match backend
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARS_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(EARS_ROOT, "scripts"))
# MelExtractor lives in src/
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import librosa
import parselmouth
from parselmouth import praat

SAMPLE_RATE = 44100
SILENCE_RMS = 0.008
WINDOW_S = 1.0       # 1s analysis window — matches EARS_WINDOW_SAMPLES in backend
HOP_S = 0.25         # 250ms hop (sliding window)
# Thresholds recalibrated for MelExtractor tonos range
STRAIN_GREEN = 0.50
STRAIN_YELLOW = 0.68


def _safe(v, default=0.5):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else max(0.0, min(1.0, f))
    except Exception:
        return default


def compute_parselmouth(audio_np, sr):
    """HNR + shimmer via Praat. Returns (hnr_db, shimmer_pct)."""
    snd = parselmouth.Sound(audio_np.astype(np.float64), sampling_frequency=float(sr))
    hnr_db = 20.0
    shimmer_pct = 0.0
    try:
        harm = snd.to_harmonicity()
        vals = harm.values[0]
        valid = vals[vals > -200]
        hnr_db = float(np.mean(valid)) if len(valid) > 0 else 20.0
        hnr_db = max(0.0, hnr_db)
    except Exception:
        pass
    try:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shimmer_pct = praat.call(
            [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        ) * 100
        shimmer_pct = max(0.0, shimmer_pct)
    except Exception:
        pass
    return hnr_db, shimmer_pct


def compute_strain(tonos, thymos, hnr_db, shimmer_pct):
    # ears_score = tonos alone (thymos broken with MelExtractor input)
    ears_score = tonos
    hnr_norm = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
    shimmer_norm = min(1.0, shimmer_pct / 10.0)
    return ears_score * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10


def zone(score):
    if score < STRAIN_GREEN:
        return "GREEN"
    elif score < STRAIN_YELLOW:
        return "YELLOW"
    else:
        return "RED"


def zone_emoji(score):
    if score < STRAIN_GREEN:
        return "🟢"
    elif score < STRAIN_YELLOW:
        return "🟡"
    else:
        return "🔴"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_recording.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"\nLoading {os.path.basename(audio_path)}...")

    # Convert m4a/aac/mp3 via ffmpeg
    if audio_path.lower().endswith(('.m4a', '.aac', '.mp4', '.mp3')):
        import tempfile, subprocess
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', str(SAMPLE_RATE), tmp.name],
            capture_output=True, check=True
        )
        audio, sr = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp.name)
    else:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s | SR: {sr}Hz (backend match: {sr == SAMPLE_RATE})")

    # Import EARS
    print("Loading EARS...")
    from mel_extractor import MelExtractor
    from frequency_explorer import compute_emotion_properties, compute_tactile_properties

    mel_extractor = MelExtractor(sample_rate=SAMPLE_RATE)

    window_samples = int(WINDOW_S * sr)
    hop_samples = int(HOP_S * sr)

    results = []
    n_voiced = 0
    n_silent = 0
    n_noisy = 0   # above RMS gate but no clean signal

    i = 0
    while i + window_samples <= len(audio):
        chunk = audio[i:i + window_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        t = i / sr

        if rms < SILENCE_RMS:
            n_silent += 1
            i += hop_samples
            continue

        n_voiced += 1

        # EARS — exact backend path
        mel_frames = mel_extractor.extract_from_audio(chunk)
        em = compute_emotion_properties(mel_frames)
        tc = compute_tactile_properties(mel_frames)
        tonos = _safe(em.get("tension"), 0.5)
        thymos = _safe(em.get("arousal"), 0.5)   # display only — not used in formula

        # Parselmouth
        hnr_db, shimmer_pct = compute_parselmouth(chunk, sr)

        sc = compute_strain(tonos, thymos, hnr_db, shimmer_pct)
        results.append({
            "t": t,
            "strain": sc,
            "tonos": tonos,
            "thymos": thymos,
            "hnr_db": hnr_db,
            "shimmer_pct": shimmer_pct,
            "rms": rms,
        })

        i += hop_samples

    if not results:
        print("No voiced frames detected.")
        return

    strains = [r["strain"] for r in results]
    avg = float(np.nanmean(strains))
    peak = float(np.nanmax(strains))
    valid = [s for s in strains if s == s]
    green_pct  = sum(1 for s in valid if s < STRAIN_GREEN) / len(valid) * 100
    yellow_pct = sum(1 for s in valid if STRAIN_GREEN <= s < STRAIN_YELLOW) / len(valid) * 100
    red_pct    = sum(1 for s in valid if s >= STRAIN_YELLOW) / len(valid) * 100

    print(f"\n{'='*60}")
    print(f"STRAIN ANALYSIS: {os.path.basename(audio_path)}")
    print(f"{'='*60}")
    print(f"Active frames: {n_voiced} | Silent: {n_silent}")
    print(f"Avg strain:  {avg:.3f}  {zone_emoji(avg)} {zone(avg)}")
    print(f"Peak strain: {peak:.3f}  {zone_emoji(peak)}")
    print(f"Green: {green_pct:.0f}%  Yellow: {yellow_pct:.0f}%  Red: {red_pct:.0f}%")
    print()
    print("TIMELINE (active windows only):")
    print(f"{'Time':>6}  {'Strain':>6}  {'Zone':>7}  {'Tonos':>6}  {'Thymos':>7}  {'HNR':>6}  {'Shimmer':>7}")
    print("-" * 62)
    for r in results:
        z = zone(r['strain'])
        print(f"{r['t']:>5.1f}s  {r['strain']:>6.3f}  {z:>7}  "
              f"{r['tonos']:>6.3f}  {r['thymos']:>7.3f}  "
              f"{r['hnr_db']:>5.1f}dB  {r['shimmer_pct']:>6.2f}%")

    print(f"\nOverall: {zone_emoji(avg)} avg={avg:.3f}, peak={peak:.3f}")
    print()
    print("(Threshold guide: <0.40 GREEN | 0.40-0.65 YELLOW | >0.65 RED)")


if __name__ == "__main__":
    main()
