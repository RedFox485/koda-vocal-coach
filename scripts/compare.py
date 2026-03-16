#!/usr/bin/env python3
"""
Compare EARS strain output against your manual annotations.
Generates a side-by-side chart: machine prediction vs. human ground truth.

Usage:
  python scripts/compare.py                              # pick from annotated files
  python scripts/compare.py "Danny - Chris Young R1"    # by name (no extension)
  python scripts/compare.py docs/annotations/foo.json
"""
import os
import sys
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "docs", "annotations")
CHARTS_DIR      = os.path.join(PROJECT_ROOT, "docs", "strain-charts")
RECORDINGS_DIR  = os.path.join(PROJECT_ROOT, "Vocal test recording sessions")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_ROOT), "audio-perception", "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SAMPLE_RATE  = 44100
SILENCE_RMS  = 0.008
WINDOW_S     = 1.0
HOP_S        = 0.25
STRAIN_GREEN  = 0.50
STRAIN_YELLOW = 0.68

ZONE_NUM = {"green": 0.2, "yellow": 0.5, "red": 0.8, "idle": 0.0}
ZONE_COLOR = {"green": "#22c55e", "yellow": "#f59e0b", "red": "#ef4444", "idle": "#475569"}


def zone_to_y(zone):
    return ZONE_NUM.get(zone, 0.0)


def load_annotation(path):
    with open(path) as f:
        return json.load(f)


def pick_annotation():
    files = sorted([f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".json")])
    if not files:
        print("No annotations found — run annotate.py first")
        sys.exit(1)
    print("\nPick an annotation:\n")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    print()
    while True:
        try:
            n = int(input("Enter number: ").strip())
            if 1 <= n <= len(files):
                return os.path.join(ANNOTATIONS_DIR, files[n-1])
        except (ValueError, EOFError):
            pass


def run_analysis(audio_path):
    """Run EARS strain analysis on the file."""
    import subprocess
    import tempfile
    import librosa

    print(f"  Analyzing {os.path.basename(audio_path)}...")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", str(SAMPLE_RATE), tmp.name],
        capture_output=True, check=True
    )
    audio, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp.name)

    from mel_extractor import MelExtractor
    from frequency_explorer import compute_emotion_properties
    try:
        import parselmouth
        from parselmouth import praat
        PRAAT = True
    except ImportError:
        PRAAT = False

    mel_ex = MelExtractor(sample_rate=SAMPLE_RATE)
    window_samples = int(WINDOW_S * SAMPLE_RATE)
    hop_samples    = int(HOP_S * SAMPLE_RATE)

    results = []
    i = 0
    while i + window_samples <= len(audio):
        chunk = audio[i:i + window_samples]
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        t     = i / SAMPLE_RATE

        if rms < SILENCE_RMS:
            results.append({"t": t, "strain": None, "voiced": False})
            i += hop_samples
            continue

        try:
            mel    = mel_ex.extract_from_audio(chunk)
            em     = compute_emotion_properties(mel)
            tonos_raw = em.get("tension", 0.5)
            f = float(tonos_raw) if tonos_raw is not None else 0.5
            tonos = 0.5 if (math.isnan(f) or math.isinf(f)) else max(0.0, min(1.0, f))
        except Exception:
            tonos = 0.5

        hnr_db, shimmer_pct = 20.0, 0.0
        if PRAAT:
            try:
                snd  = parselmouth.Sound(chunk.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
                harm = snd.to_harmonicity()
                vals = harm.values[0]
                valid = vals[vals > -200]
                hnr_db = float(np.mean(valid)) if len(valid) else 20.0
                hnr_db = max(0.0, hnr_db)
                pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
                shimmer_pct = max(0.0, praat.call(
                    [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
                ) * 100)
            except Exception:
                pass

        hnr_norm     = max(0.0, min(1.0, (20.0 - hnr_db) / 30.0))
        shimmer_norm = min(1.0, shimmer_pct / 10.0)
        strain = tonos * 0.70 + hnr_norm * 0.20 + shimmer_norm * 0.10
        results.append({"t": t, "strain": strain, "voiced": True})
        i += hop_samples

    return results


def annotation_to_timeseries(annots, duration, interval=0.25):
    """Convert annotation events → dense timeseries at <interval> steps."""
    events = sorted([a for a in annots if a["type"] in ("start", "change")], key=lambda x: x["t"])
    times = np.arange(0, duration, interval)
    zones = []
    for t in times:
        # Find most recent event at or before t
        z = "idle"
        for ev in events:
            if ev["t"] <= t:
                z = ev["zone"]
        zones.append(z)
    return times, zones


def make_comparison_chart(annotation_path, audio_path, results):
    ann      = load_annotation(annotation_path)
    ann_stem = os.path.splitext(os.path.basename(annotation_path))[0]
    dur      = ann["duration"]
    out      = os.path.join(CHARTS_DIR, f"{ann_stem}_comparison.png")

    # EARS data
    voiced  = [r for r in results if r["voiced"]]
    ears_t  = [r["t"] + WINDOW_S / 2 for r in voiced]
    ears_s  = [r["strain"] for r in voiced]
    if len(ears_s) > 3:
        ears_smooth = uniform_filter1d(ears_s, size=min(5, len(ears_s)//2 or 1))
    else:
        ears_smooth = ears_s

    # Human annotation timeseries
    ann_t, ann_z = annotation_to_timeseries(ann["annotations"], dur, interval=0.25)
    ann_y = [zone_to_y(z) for z in ann_z]

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1.5, 0.5]})
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"{ann_stem} — EARS vs. Human Annotation",
                 color="#e2e8f0", fontsize=12, y=0.98)

    # ── Top: EARS strain ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#16213e")
    ax.axhspan(0,            STRAIN_GREEN,  alpha=0.12, color="#22c55e")
    ax.axhspan(STRAIN_GREEN, STRAIN_YELLOW, alpha=0.12, color="#f59e0b")
    ax.axhspan(STRAIN_YELLOW, 1.0,          alpha=0.12, color="#ef4444")
    ax.axhline(STRAIN_GREEN,  color="#22c55e", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(STRAIN_YELLOW, color="#ef4444", lw=0.7, ls="--", alpha=0.5)

    for t, s in zip(ears_t, ears_s):
        c = "#22c55e" if s < STRAIN_GREEN else "#f59e0b" if s < STRAIN_YELLOW else "#ef4444"
        ax.scatter(t, s, color=c, s=20, alpha=0.7, zorder=3)
    if len(ears_smooth) > 0:
        ax.plot(ears_t, ears_smooth, color="#ffffff", lw=1.2, alpha=0.4, zorder=2)

    avg  = float(np.mean(ears_s)) if ears_s else 0
    peak = float(np.max(ears_s))  if ears_s else 0
    ax.axhline(avg, color="#a78bfa", lw=0.9, ls=":", alpha=0.7)
    ax.text(dur * 0.01, avg + 0.015, f"avg {avg:.3f}", color="#a78bfa", fontsize=8)

    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.set_ylabel("EARS Strain", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#334155")

    # Stats
    if ears_s:
        g = sum(1 for s in ears_s if s < STRAIN_GREEN) / len(ears_s) * 100
        y = sum(1 for s in ears_s if STRAIN_GREEN <= s < STRAIN_YELLOW) / len(ears_s) * 100
        r = sum(1 for s in ears_s if s >= STRAIN_YELLOW) / len(ears_s) * 100
        ax.set_title(f"EARS   avg={avg:.3f}  peak={peak:.3f}  G:{g:.0f}%  Y:{y:.0f}%  R:{r:.0f}%",
                     color="#94a3b8", fontsize=9, pad=4)

    # ── Middle: Human annotation ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")

    # Draw colored spans for each zone
    prev_t, prev_z = 0.0, ann_z[0] if ann_z else "idle"
    for t, z in zip(ann_t[1:], ann_z[1:]):
        if z != prev_z:
            c = ZONE_COLOR.get(prev_z, "#475569")
            ax2.axvspan(prev_t, t, alpha=0.35, color=c)
            prev_t = t
            prev_z = z
    ax2.axvspan(prev_t, dur, alpha=0.35, color=ZONE_COLOR.get(prev_z, "#475569"))

    # Draw y-level line
    ax2.step(ann_t, ann_y, color="#ffffff", lw=1.2, alpha=0.6, where="post")

    # Mark change events
    for ev in ann["annotations"]:
        if ev["type"] in ("start", "change"):
            c = ZONE_COLOR.get(ev["zone"], "#ffffff")
            ax2.axvline(ev["t"], color=c, lw=1.0, alpha=0.8)

    ax2.set_xlim(0, dur)
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_yticks([0.2, 0.5, 0.8])
    ax2.set_yticklabels(["GREEN", "YELLOW", "RED"], fontsize=7, color="#94a3b8")
    ax2.set_ylabel("Your Annotation", color="#e2e8f0", fontsize=10)
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#334155")

    # Annotation stats
    auto = [a for a in ann["annotations"] if a["type"] == "auto"]
    if auto:
        gz = [a for a in auto if a["zone"] == "green"]
        yz = [a for a in auto if a["zone"] == "yellow"]
        rz = [a for a in auto if a["zone"] == "red"]
        g_pct = len(gz) / len(auto) * 100
        y_pct = len(yz) / len(auto) * 100
        r_pct = len(rz) / len(auto) * 100
        ax2.set_title(f"Human   G:{g_pct:.0f}%  Y:{y_pct:.0f}%  R:{r_pct:.0f}%  "
                      f"({len(auto)} samples @ 100ms)",
                      color="#94a3b8", fontsize=9, pad=4)

    # ── Bottom: Agreement bar ────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#16213e")

    # Agreement: match EARS zone to annotation zone at each timepoint
    agree_t, agree_colors = [], []
    for t, az in zip(ann_t, ann_z):
        if az == "idle":
            continue
        # Find nearest EARS frame
        nearest = min(voiced, key=lambda r: abs(r["t"] - t)) if voiced else None
        if nearest is None:
            continue
        es = nearest["strain"]
        ez = "green" if es < STRAIN_GREEN else "yellow" if es < STRAIN_YELLOW else "red"
        color = "#22c55e" if ez == az else "#ef4444"
        agree_t.append(t)
        agree_colors.append(color)

    if agree_t:
        for at, ac in zip(agree_t, agree_colors):
            ax3.axvspan(at, at + 0.25, color=ac, alpha=0.6)
        n_agree = agree_colors.count("#22c55e")
        pct = n_agree / len(agree_colors) * 100
        ax3.text(dur * 0.5, 0.5, f"Agreement: {pct:.0f}%  ({n_agree}/{len(agree_colors)})",
                 color="#e2e8f0", fontsize=9, ha="center", va="center")

    ax3.set_xlim(0, dur)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=10)
    ax3.set_title("Zone Agreement  (green=match, red=mismatch)", color="#94a3b8", fontsize=8, pad=3)
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax3.spines.values(): sp.set_color("#334155")

    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.5)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {out}")
    return out


def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    args = sys.argv[1:]
    if args:
        path = args[0]
        if not path.endswith(".json"):
            # Try finding by name
            candidate = os.path.join(ANNOTATIONS_DIR, path + ".json")
            if os.path.exists(candidate):
                path = candidate
            else:
                for f in os.listdir(ANNOTATIONS_DIR):
                    if path.lower() in f.lower():
                        path = os.path.join(ANNOTATIONS_DIR, f)
                        break
    else:
        path = pick_annotation()

    if not os.path.exists(path):
        print(f"Annotation not found: {path}")
        sys.exit(1)

    ann = load_annotation(path)
    name = ann["file"]

    # Find audio file
    audio_path = os.path.join(RECORDINGS_DIR, name)
    if not os.path.exists(audio_path):
        # Search
        for f in os.listdir(RECORDINGS_DIR):
            if os.path.splitext(name)[0].lower() in f.lower():
                audio_path = os.path.join(RECORDINGS_DIR, f)
                break
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\nRunning EARS analysis on {name}...")
    results = run_analysis(audio_path)
    print(f"  {len([r for r in results if r['voiced']])} voiced frames")

    print("Generating comparison chart...")
    out = make_comparison_chart(path, audio_path, results)

    print(f"\nDone. Open: {out}\n")


if __name__ == "__main__":
    main()
