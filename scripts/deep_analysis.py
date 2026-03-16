#!/usr/bin/env python3
"""
Deep analysis of Liza Jane — what separates GREEN from YELLOW/RED?

Uses everything:
  - All EARS modality dimensions (emotion, tactile, life, geometry, chemistry...)
  - v8 signals (shimmer, CPP, HNR)
  - Amplitude modulation envelope (motion amplification concept)
  - Raw spectral binary heatmap
  - Edge analysis: what changes AT the green→yellow boundary?

Output: terminal report + PNG visualizations saved to data/ground_truth/

Usage:
    .venv/bin/python3 scripts/deep_analysis.py
    .venv/bin/python3 scripts/deep_analysis.py --no-plots   # text only, faster
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as pcall

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_extractor import MelExtractor
from frequency_explorer import analyze_mel, compute_temporal_dynamics, compute_acoustic_features
from wc_sound_as_emotion   import compute_emotion_properties
from wc_sound_as_touch     import compute_tactile_properties
from wc_sound_as_life      import compute_life_properties
from wc_sound_as_geometry  import compute_shape_properties
from wc_sound_as_chemistry import compute_chemistry
from wc_sound_as_taste     import compute_taste_properties
from wc_sound_as_weather   import compute_weather
from wc_sound_as_social    import compute_social
from wc_hidden_dimensions  import compute_hidden_dimensions
from wc_aliveness_dimensionality import compute_temporal_dimensionality

SR       = 44100
CHUNK_S  = 2.0
CHUNK    = int(SR * CHUNK_S)
CHUNK_MEL_WINDOW = SR   # 1s mel window for EARS

GT_PATH  = Path("data/ground_truth/lizajane_labels.json")
OUT_DIR  = Path("data/ground_truth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_mel = MelExtractor(sample_rate=SR)


# ─── Feature extraction per chunk ─────────────────────────────────────────────

def compute_cpp(chunk, sr=SR):
    try:
        N   = len(chunk)
        pre = np.append(chunk[0], chunk[1:] - 0.97 * chunk[:-1])
        spec = np.fft.rfft(pre * np.hanning(N), n=N)
        lp   = np.log(np.abs(spec)**2 + 1e-12)
        cep  = np.real(np.fft.irfft(lp))[:N//2]
        qn, qx = int(sr/600), int(sr/75)
        if qx >= len(cep): return float('nan')
        pi = qn + int(np.argmax(cep[qn:qx+1]))
        qa = np.arange(len(cep)) / float(sr)
        c  = np.polyfit(qa[qn:qx+1], cep[qn:qx+1], 1)
        return float(cep[pi] - np.polyval(c, qa[pi]))
    except Exception:
        return float('nan')


def am_envelope_features(y_chunk, sr=SR):
    """Amplitude modulation envelope analysis — the 'motion amplification' concept.

    Decompose voice into sub-bands, extract slow temporal envelope of each,
    measure modulation depth at different timescales.

    Pressed/strained phonation → different AM modulation compared to modal.
    This is the audio equivalent of Eulerian Video Magnification:
    amplify and measure the slow amplitude changes riding on the carrier signal.
    """
    # Sub-band decomposition: split into 4 frequency bands
    from scipy.signal import butter, sosfilt, hilbert
    bands = [(75, 300), (300, 800), (800, 2000), (2000, 5000)]
    feats = {}

    for (lo, hi) in bands:
        try:
            sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
            filtered = sosfilt(sos, y_chunk.astype(np.float64))
            # Hilbert envelope = instantaneous amplitude
            envelope = np.abs(hilbert(filtered))
            # Normalize
            env_rms = float(np.sqrt(np.mean(envelope**2)))
            if env_rms < 1e-8:
                continue
            envelope_norm = envelope / env_rms

            # Temporal modulation spectrum of the envelope (0-30Hz)
            # This is literally what "motion amplification" does:
            # look at how FAST the amplitude is fluctuating
            env_fft = np.fft.rfft(envelope_norm)
            env_freqs = np.fft.rfftfreq(len(envelope_norm), d=1/sr)

            # Power in modulation bands
            def mod_power(f_lo, f_hi):
                mask = (env_freqs >= f_lo) & (env_freqs < f_hi)
                return float(np.sum(np.abs(env_fft[mask])**2)) / (len(env_fft) + 1e-9)

            feats[f"am_{lo}-{hi}_slow"]   = mod_power(0.5, 4.0)    # breath-rate modulation
            feats[f"am_{lo}-{hi}_medium"] = mod_power(4.0, 10.0)   # vocal tremor range
            feats[f"am_{lo}-{hi}_fast"]   = mod_power(10.0, 30.0)  # shimmer-rate modulation
            feats[f"am_{lo}-{hi}_depth"]  = float(np.std(envelope_norm))  # total modulation depth

        except Exception:
            pass

    return feats


def praat_features(y_chunk, sr=SR):
    """Shimmer, jitter, HNR from parselmouth."""
    out = {"shimmer": float('nan'), "hnr": float('nan'), "jitter": float('nan')}
    try:
        snd = parselmouth.Sound(y_chunk.astype(np.float64), sampling_frequency=float(sr))
        harm = snd.to_harmonicity()
        vals = harm.values[0]
        valid = vals[vals > -200]
        out["hnr"] = float(np.mean(valid)) if len(valid) > 0 else float('nan')
        pp = pcall(snd, "To PointProcess (periodic, cc)", 75, 600)
        s = pcall([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        out["shimmer"] = (s or 0.0) * 100.0
        j = pcall(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["jitter"] = (j or 0.0) * 100.0
    except Exception:
        pass
    return out


def ears_features(y_chunk):
    """Full EARS battery on a 2s chunk. Returns flat dict of all dims."""
    feats = {}
    try:
        mel = _mel.extract_from_audio(y_chunk)
        for fn, name in [
            (compute_emotion_properties,    "emotion"),
            (compute_tactile_properties,    "tactile"),
            (compute_life_properties,       "life"),
            (compute_shape_properties,      "geometry"),
            (compute_chemistry,             "chemistry"),
            (compute_taste_properties,      "taste"),
            (compute_weather,               "weather"),
            (compute_social,                "social"),
            (compute_hidden_dimensions,     "hidden"),
            (compute_temporal_dimensionality, "temporal_dim"),
        ]:
            try:
                result = fn(mel)
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and not math.isnan(float(v)):
                            feats[f"{name}.{k}"] = float(v)
            except Exception:
                pass

        # Full analyze_mel for remaining dims
        full = analyze_mel(mel)
        for mod_name, mod_dict in full.get("modalities", {}).items():
            if isinstance(mod_dict, dict):
                for k, v in mod_dict.items():
                    key = f"ears.{mod_name}.{k}"
                    if key not in feats and isinstance(v, (int, float)):
                        try:
                            if not math.isnan(float(v)):
                                feats[key] = float(v)
                        except Exception:
                            pass
    except Exception:
        pass
    return feats


def extract_all(y_chunk):
    """Full feature battery for one 2s chunk."""
    feats = {}

    # v8 signals
    cpp = compute_cpp(y_chunk)
    feats["cpp"] = cpp if not math.isnan(cpp) else 0.0

    pf = praat_features(y_chunk)
    feats.update(pf)

    # Spectral summary
    y32 = y_chunk.astype(np.float32)
    feats["spectral_centroid"] = float(np.mean(
        librosa.feature.spectral_centroid(y=y32, sr=SR, n_fft=1024, hop_length=256)))
    feats["spectral_bandwidth"] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y32, sr=SR, n_fft=1024, hop_length=256)))
    feats["spectral_rolloff"] = float(np.mean(
        librosa.feature.spectral_rolloff(y=y32, sr=SR, n_fft=1024, hop_length=256)))
    feats["zcr"] = float(np.mean(
        librosa.feature.zero_crossing_rate(y=y32, hop_length=256)))
    feats["rms"] = float(np.sqrt(np.mean(y_chunk.astype(np.float64)**2)))

    mfcc = librosa.feature.mfcc(y=y32, sr=SR, n_mfcc=13, n_fft=1024, hop_length=256)
    for i in range(13):
        feats[f"mfcc_{i}"] = float(np.mean(mfcc[i]))

    # AM envelope (motion amplification)
    feats.update(am_envelope_features(y_chunk))

    # EARS full battery
    feats.update(ears_features(y_chunk))

    return feats


# ─── Statistical analysis ──────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d effect size between two groups."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not GT_PATH.exists():
        print(f"ERROR: {GT_PATH} not found. Run label_song.py first.")
        sys.exit(1)

    gt    = json.loads(GT_PATH.read_text())
    labels = gt["chunks"]

    print(f"\nLoading {gt['path']}...")
    y, _ = librosa.load(gt["path"], sr=SR, mono=True)
    n_chunks = int(len(y) / SR / CHUNK_S)

    print(f"Extracting features from {n_chunks} chunks...")
    print("(Takes ~2-3 min — running full EARS + AM analysis on each chunk)\n")

    chunk_feats = {}
    for ci in range(n_chunks):
        label = labels.get(str(ci))
        if label is None or label == "skip":
            continue
        y_chunk = y[ci * CHUNK : (ci + 1) * CHUNK]
        rms = float(np.sqrt(np.mean(y_chunk.astype(np.float64)**2)))
        if rms < 0.008:
            continue
        print(f"  chunk {ci:>3} ({ci*2:>3}-{(ci+1)*2}s) [{label}]...", end="\r")
        feats = extract_all(y_chunk)
        feats["_label"] = label
        feats["_t_start"] = ci * CHUNK_S
        chunk_feats[ci] = feats

    print(f"\nExtracted {len(chunk_feats)} chunks.       ")

    # Group by label
    by_label = {"green": [], "yellow": [], "red": []}
    for ci, feats in chunk_feats.items():
        lbl = feats["_label"]
        if lbl in by_label:
            by_label[lbl].append(feats)

    print(f"\n  green:  {len(by_label['green'])} chunks")
    print(f"  yellow: {len(by_label['yellow'])} chunks")
    print(f"  red:    {len(by_label['red'])} chunks")

    # All feature names (exclude meta)
    all_feat_names = sorted(set(
        k for feats in chunk_feats.values()
        for k in feats if not k.startswith("_")
    ))

    # Cohen's d: green vs (yellow+red) — what discriminates healthy from strained?
    green_feats  = by_label["green"]
    strained_feats = by_label["yellow"] + by_label["red"]

    print(f"\n{'═'*65}")
    print(f"  TOP DISCRIMINATING FEATURES: green vs yellow/red")
    print(f"  (Cohen's d — effect size. |d|>0.8 = large, >0.5 = medium)")
    print(f"{'═'*65}\n")

    effects = []
    for feat in all_feat_names:
        g_vals = [f[feat] for f in green_feats  if feat in f and not math.isnan(f[feat])]
        s_vals = [f[feat] for f in strained_feats if feat in f and not math.isnan(f[feat])]
        if len(g_vals) >= 3 and len(s_vals) >= 3:
            d = cohens_d(g_vals, s_vals)
            effects.append((feat, d, np.mean(g_vals), np.mean(s_vals)))

    effects.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  {'Feature':<45}  {'d':>6}  {'green_mean':>10}  {'strain_mean':>11}")
    print(f"  {'─'*45}  {'─'*6}  {'─'*10}  {'─'*11}")
    for feat, d, gm, sm in effects[:30]:
        direction = "↑strain" if sm > gm else "↑green"
        print(f"  {feat:<45}  {d:>+6.3f}  {gm:>10.4f}  {sm:>11.4f}  {direction}")

    # Focus on AM features specifically
    am_effects = [(f, d, gm, sm) for f, d, gm, sm in effects if f.startswith("am_")]
    if am_effects:
        print(f"\n{'─'*65}")
        print(f"  AMPLITUDE MODULATION FEATURES (motion amplification)")
        print(f"{'─'*65}")
        for feat, d, gm, sm in sorted(am_effects, key=lambda x: abs(x[1]), reverse=True):
            direction = "↑strain" if sm > gm else "↑green"
            print(f"  {feat:<45}  {d:>+6.3f}  {gm:>10.4f}  {sm:>11.4f}  {direction}")

    # Edge analysis: what changes at green→yellow boundaries?
    print(f"\n{'═'*65}")
    print(f"  EDGE ANALYSIS: what flips at green→yellow transitions?")
    print(f"{'═'*65}\n")

    sorted_chunks = sorted(chunk_feats.items())
    transitions = []
    for i in range(len(sorted_chunks) - 1):
        ci_a, fa = sorted_chunks[i]
        ci_b, fb = sorted_chunks[i + 1]
        la, lb = fa["_label"], fb["_label"]
        if (la == "green" and lb in ("yellow", "red")) or \
           (la in ("yellow", "red") and lb == "green"):
            transitions.append((ci_a, ci_b, fa, fb, la, lb))

    print(f"  Found {len(transitions)} green↔yellow/red transitions")
    for ci_a, ci_b, fa, fb, la, lb in transitions:
        ta, tb = fa["_t_start"], fb["_t_start"]
        print(f"\n  {la.upper()} ({ta:.0f}s) → {lb.upper()} ({tb:.0f}s):")
        diffs = []
        for feat in all_feat_names:
            if feat in fa and feat in fb:
                va, vb = fa[feat], fb[feat]
                if math.isnan(va) or math.isnan(vb): continue
                if abs(va) < 1e-10 and abs(vb) < 1e-10: continue
                rel_change = (vb - va) / (abs(va) + 1e-10)
                diffs.append((feat, va, vb, rel_change))
        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        for feat, va, vb, rc in diffs[:10]:
            arrow = "↑" if vb > va else "↓"
            print(f"    {arrow} {feat:<40}  {va:>8.4f} → {vb:>8.4f}  ({rc:>+.0%})")

    # ── Visualizations ─────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            LABEL_COLOR = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}

            # ── Plot 1: Feature heatmap (top 20 features × chunks) ──────────
            top_feats = [f for f, d, gm, sm in effects[:20]]
            n_c = len(sorted_chunks)
            mat = np.zeros((len(top_feats), n_c))
            for j, (ci, feats) in enumerate(sorted_chunks):
                for i, feat in enumerate(top_feats):
                    mat[i, j] = feats.get(feat, 0.0)

            # Z-score each feature row
            for i in range(mat.shape[0]):
                s = mat[i].std()
                if s > 1e-10:
                    mat[i] = (mat[i] - mat[i].mean()) / s

            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2)
            ax.set_yticks(range(len(top_feats)))
            ax.set_yticklabels([f[:35] for f in top_feats], fontsize=7)
            ax.set_xticks(range(n_c))
            ax.set_xticklabels(
                [f"{int(feats['_t_start'])}" for _, feats in sorted_chunks],
                rotation=45, fontsize=7)
            ax.set_xlabel("Chunk start time (s)")
            ax.set_title("Feature heatmap (z-scored) — top 20 discriminating features\nChunks colored by Daniel's label")

            # Color x-tick labels by label
            for j, (ci, feats) in enumerate(sorted_chunks):
                lbl = feats["_label"]
                ax.get_xticklabels()[j].set_color(LABEL_COLOR.get(lbl, "black"))
                # Add colored bar at top
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, -0.5), 1, 0.3,
                    color=LABEL_COLOR.get(lbl, "gray"),
                    transform=ax.transData, clip_on=False))

            plt.colorbar(im, ax=ax, label="z-score")
            plt.tight_layout()
            out1 = OUT_DIR / "feature_heatmap.png"
            plt.savefig(out1, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"\n  Saved: {out1}")

            # ── Plot 2: Box plots of top 8 features by label ──────────────
            top8 = [f for f, d, gm, sm in effects[:8]]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for i, feat in enumerate(top8):
                ax = axes[i]
                data_by_label = {}
                for lbl in ("green", "yellow", "red"):
                    vals = [f[feat] for f in by_label[lbl] if feat in f and not math.isnan(f[feat])]
                    data_by_label[lbl] = vals

                bp = ax.boxplot(
                    [data_by_label["green"], data_by_label["yellow"], data_by_label["red"]],
                    labels=["green", "yellow", "red"],
                    patch_artist=True)
                for patch, color in zip(bp['boxes'], ["#2ecc71", "#f39c12", "#e74c3c"]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                d_val = next((d for f, d, gm, sm in effects if f == feat), 0)
                ax.set_title(f"{feat[:30]}\nd={d_val:+.2f}", fontsize=8)
                ax.tick_params(labelsize=7)

            plt.suptitle("Top 8 discriminating features by Daniel's label\n(Cohen's d vs green/strain split)", fontsize=10)
            plt.tight_layout()
            out2 = OUT_DIR / "feature_boxplots.png"
            plt.savefig(out2, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out2}")

            # ── Plot 3: AM envelope (motion amplification) over full song ──
            print("\n  Generating AM envelope plot...")
            from scipy.signal import butter, sosfilt, hilbert
            y_full = y[:int(67 * SR)]

            fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
            band_configs = [
                (75, 300, "Fundamental (75-300Hz)", "#3498db"),
                (300, 800, "Lower harmonics (300-800Hz)", "#2ecc71"),
                (800, 2000, "Mid harmonics (800-2kHz)", "#f39c12"),
                (2000, 5000, "Upper harmonics (2-5kHz)", "#e74c3c"),
            ]
            t_axis = np.arange(len(y_full)) / SR

            for ax, (lo, hi, label, color) in zip(axes, band_configs):
                sos = butter(4, [lo, hi], btype='bandpass', fs=SR, output='sos')
                filtered = sosfilt(sos, y_full.astype(np.float64))
                envelope = np.abs(hilbert(filtered))
                # Smooth envelope at 10Hz (100ms resolution)
                window = int(SR * 0.1)
                smooth_env = np.convolve(envelope, np.ones(window)/window, mode='same')
                ax.plot(t_axis, smooth_env, color=color, linewidth=0.8, alpha=0.8)
                ax.set_ylabel(label, fontsize=7)
                ax.set_ylim(bottom=0)

                # Shade regions by Daniel's label
                for ci_str, lbl in labels.items():
                    ci = int(ci_str)
                    if lbl == "skip": continue
                    t0, t1 = ci * CHUNK_S, (ci + 1) * CHUNK_S
                    ax.axvspan(t0, t1, alpha=0.15,
                               color=LABEL_COLOR.get(lbl, "gray"), zorder=0)

            axes[-1].set_xlabel("Time (s)")
            # Legend
            patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
                       for l, c in LABEL_COLOR.items()]
            axes[0].legend(handles=patches, loc="upper right", fontsize=7)
            axes[0].set_title(
                "AM Envelope by frequency band — motion amplification view\n"
                "(Background shading = Daniel's labels)")
            plt.tight_layout()
            out3 = OUT_DIR / "am_envelope.png"
            plt.savefig(out3, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out3}")

            # ── Plot 4: Scatter of top 2 features, colored by label ────────
            if len(effects) >= 2:
                f1, d1 = effects[0][0], effects[0][1]
                f2, d2 = effects[1][0], effects[1][1]
                fig, ax = plt.subplots(figsize=(8, 6))
                for lbl, color in LABEL_COLOR.items():
                    xs = [f[f1] for f in by_label[lbl] if f1 in f and not math.isnan(f[f1])]
                    ys = [f[f2] for f in by_label[lbl] if f2 in f and not math.isnan(f[f2])]
                    ax.scatter(xs, ys, c=color, label=lbl, s=60, alpha=0.8, edgecolors='k', linewidths=0.5)
                ax.set_xlabel(f1[:50], fontsize=9)
                ax.set_ylabel(f2[:50], fontsize=9)
                ax.set_title(f"Top 2 discriminating features\n{f1[:40]} (d={d1:+.2f}) vs {f2[:40]} (d={d2:+.2f})", fontsize=9)
                ax.legend()
                plt.tight_layout()
                out4 = OUT_DIR / "top2_scatter.png"
                plt.savefig(out4, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {out4}")

            print(f"\n  Open all plots: open {OUT_DIR}/*.png")

        except Exception as e:
            print(f"\n  [plots skipped: {e}]")

    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    if effects:
        print(f"\n  Strongest signal separating green from strain:")
        for feat, d, gm, sm in effects[:5]:
            direction = "HIGHER in strain" if sm > gm else "HIGHER in green"
            print(f"    {feat}: d={d:+.3f}  ({direction})")
    print()


if __name__ == "__main__":
    main()
