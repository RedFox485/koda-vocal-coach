#!/usr/bin/env python3
"""
Generate strain timeline charts for vocal recordings.
Runs offline — no live server needed. Uses exact same pipeline as backend.

Usage:
  python scripts/strain_chart.py                  # all recordings in Vocal test recording sessions/
  python scripts/strain_chart.py path/to/file.m4a # single file
  python scripts/strain_chart.py --ref            # include Macedonia St reference files too

Output: docs/strain-charts/<filename>.png
"""
import sys
import os
import math
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARS_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(EARS_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import librosa
from mel_extractor import MelExtractor
from frequency_explorer import compute_emotion_properties

try:
    import parselmouth
    from parselmouth import praat
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False
    print("[WARN] parselmouth not available — HNR/shimmer will use defaults")

SAMPLE_RATE = 44100
SILENCE_RMS = 0.008
WINDOW_S = 1.0
HOP_S = 0.25
STRAIN_GREEN = 0.50
STRAIN_YELLOW = 0.68

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "strain-charts")


def _safe(v, default=0.5):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else max(0.0, min(1.0, f))
    except Exception:
        return default


def load_audio(path):
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.run(
        ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(SAMPLE_RATE), tmp.name],
        capture_output=True, check=True
    )
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)
    return audio


def analyze(audio):
    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    window_samples = int(WINDOW_S * SAMPLE_RATE)
    hop_samples = int(HOP_S * SAMPLE_RATE)

    results = []
    i = 0
    while i + window_samples <= len(audio):
        chunk = audio[i:i + window_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        t = i / SAMPLE_RATE

        if rms < SILENCE_RMS:
            results.append({"t": t, "strain": None, "tonos": None, "rms": rms, "voiced": False})
            i += hop_samples
            continue

        # EARS
        try:
            mel_frames = mel_ex.extract_from_audio(chunk)
            em = compute_emotion_properties(mel_frames)
            tonos = _safe(em.get("tension"), 0.5)
        except Exception:
            tonos = 0.5

        # Parselmouth
        hnr_db, shimmer_pct = 20.0, 0.0
        if PRAAT_AVAILABLE:
            try:
                snd = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
                harm = snd.to_harmonicity()
                vals = harm.values[0]
                valid = vals[vals > -200]
                hnr_db = float(np.mean(valid)) if len(valid) > 0 else 20.0
                hnr_db = max(0.0, hnr_db)
            except Exception:
                pass
            try:
                snd2 = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
                pp = praat.call(snd2, "To PointProcess (periodic, cc)", 75, 600)
                shimmer_pct = praat.call(
                    [snd2, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
                ) * 100
                shimmer_pct = max(0.0, shimmer_pct)
            except Exception:
                pass

        hnr_norm = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
        shimmer_norm = min(1.0, shimmer_pct / 10.0)
        strain = tonos * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10

        results.append({"t": t, "strain": strain, "tonos": tonos, "rms": rms, "voiced": True})
        i += hop_samples

    return results


def make_chart(audio_path, results, output_path):
    label = os.path.basename(audio_path)
    duration = (len([r for r in results]) * HOP_S) + WINDOW_S

    voiced = [r for r in results if r["voiced"]]
    if not voiced:
        print(f"  [SKIP] No voiced frames in {label}")
        return

    times = [r["t"] + WINDOW_S/2 for r in voiced]   # center of window
    strains = [r["strain"] for r in voiced]
    tonos_vals = [r["tonos"] for r in voiced]

    avg = float(np.mean(strains))
    peak = float(np.max(strains))
    green_pct = sum(1 for s in strains if s < STRAIN_GREEN) / len(strains) * 100
    yellow_pct = sum(1 for s in strains if STRAIN_GREEN <= s < STRAIN_YELLOW) / len(strains) * 100
    red_pct = sum(1 for s in strains if s >= STRAIN_YELLOW) / len(strains) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#1a1a2e')

    # ── Strain timeline ──────────────────────────────────────────────────────────
    ax1.set_facecolor('#16213e')

    # Zone bands
    max_t = max(times) + WINDOW_S/2
    ax1.axhspan(0, STRAIN_GREEN, alpha=0.15, color='#22c55e', zorder=0)
    ax1.axhspan(STRAIN_GREEN, STRAIN_YELLOW, alpha=0.15, color='#f59e0b', zorder=0)
    ax1.axhspan(STRAIN_YELLOW, 1.0, alpha=0.15, color='#ef4444', zorder=0)

    # Threshold lines
    ax1.axhline(STRAIN_GREEN, color='#22c55e', linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.axhline(STRAIN_YELLOW, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.6)

    # Color points by zone
    for t, s in zip(times, strains):
        color = '#22c55e' if s < STRAIN_GREEN else '#f59e0b' if s < STRAIN_YELLOW else '#ef4444'
        ax1.scatter(t, s, color=color, s=25, zorder=3, alpha=0.85)

    # Smooth line
    if len(times) > 3:
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(strains, size=min(5, len(strains)//2 or 1))
        ax1.plot(times, smooth, color='#ffffff', linewidth=1.5, alpha=0.5, zorder=2)

    # Avg line
    ax1.axhline(avg, color='#a78bfa', linewidth=1.0, linestyle=':', alpha=0.8)
    ax1.text(max_t * 0.02, avg + 0.015, f'avg {avg:.3f}', color='#a78bfa', fontsize=9)

    ax1.set_xlim(0, max_t)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Strain Score', color='#e2e8f0', fontsize=11)
    ax1.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#334155')

    # Zone labels
    ax1.text(max_t * 0.99, STRAIN_GREEN/2, 'GREEN', color='#22c55e', fontsize=8,
             ha='right', va='center', alpha=0.7)
    ax1.text(max_t * 0.99, (STRAIN_GREEN + STRAIN_YELLOW)/2, 'YELLOW', color='#f59e0b', fontsize=8,
             ha='right', va='center', alpha=0.7)
    ax1.text(max_t * 0.99, (STRAIN_YELLOW + 1.0)/2, 'RED', color='#ef4444', fontsize=8,
             ha='right', va='center', alpha=0.7)

    # Stats box
    stats = f"avg={avg:.3f}  peak={peak:.3f}  🟢{green_pct:.0f}%  🟡{yellow_pct:.0f}%  🔴{red_pct:.0f}%  frames={len(voiced)}"
    ax1.set_title(f'{label}\n{stats}', color='#e2e8f0', fontsize=11, pad=10)

    # ── Tonos sub-plot ──────────────────────────────────────────────────────────
    ax2.set_facecolor('#16213e')
    ax2.plot(times, tonos_vals, color='#60a5fa', linewidth=1.2, alpha=0.8)
    ax2.axhline(0.5, color='#64748b', linewidth=0.6, linestyle=':')
    ax2.set_xlim(0, max_t)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Tonos', color='#60a5fa', fontsize=9)
    ax2.set_xlabel('Time (seconds)', color='#94a3b8', fontsize=10)
    ax2.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#334155')

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"    avg={avg:.3f}  peak={peak:.3f}  🟢{green_pct:.0f}%  🟡{yellow_pct:.0f}%  🔴{red_pct:.0f}%")


def process_file(audio_path):
    name = os.path.splitext(os.path.basename(audio_path))[0]
    out = os.path.join(OUTPUT_DIR, f"{name}.png")
    print(f"\nAnalyzing: {os.path.basename(audio_path)}")
    audio = load_audio(audio_path)
    dur = len(audio) / SAMPLE_RATE
    print(f"  Duration: {dur:.1f}s")
    results = analyze(audio)
    make_chart(audio_path, results, out)
    return out


def main():
    args = sys.argv[1:]
    include_ref = "--ref" in args
    args = [a for a in args if a != "--ref"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args:
        files = args
    else:
        recordings_dir = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")
        files = sorted([
            os.path.join(recordings_dir, f)
            for f in os.listdir(recordings_dir)
            if f.endswith(('.m4a', '.wav', '.mp3', '.aac'))
        ])

    if include_ref:
        ref_dir = os.path.expanduser("~/Downloads")
        for name in ["Macedonia St 27.m4a", "Macedonia St 29.m4a", "Macedonia St 30.m4a"]:
            p = os.path.join(ref_dir, name)
            if os.path.exists(p):
                files.append(p)

    print(f"Processing {len(files)} recording(s)...")
    for f in files:
        process_file(f)

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
