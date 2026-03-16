#!/usr/bin/env python3
"""
Wildcard: Listen and Predict
==============================
Can we perceive audio by reading mel spectrograms frame by frame,
describe what's happening, predict what comes next, and measure accuracy?

This is the most direct test of the thesis:
if frequency IS everything, and time IS expansion into prediction,
then reading frequency data sequentially should enable prediction.

For each sample:
1. Read frames 0 to T-1 sequentially
2. At each frame, compute perceptual properties
3. Predict properties of the next frame based on the pattern so far
4. Compare prediction vs reality
5. Measure prediction error and where surprises occur

This also tests the "vibrating membrane" idea:
track the eigenmode structure of the mel correlation matrix over time
and detect when the MODE shifts (not just when values change).
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def frame_properties(frame_linear):
    """Extract perceptual properties from a single mel frame."""
    ms = frame_linear + 1e-8
    n = len(ms)
    fb = np.linspace(0, 1, n)
    sn = ms / (ms.sum() + 1e-8)

    centroid = np.sum(fb * ms) / np.sum(ms)
    energy = np.sum(ms ** 2)
    bw = np.sqrt(max(np.sum(((fb - centroid)**2) * ms) / np.sum(ms), 0))
    entropy = -np.sum(sn * np.log(sn + 1e-10)) / (np.log(n) + 1e-8)
    third = n // 3
    low_r = np.sum(ms[:third]) / (np.sum(ms) + 1e-8)
    high_r = np.sum(ms[2*third:]) / (np.sum(ms) + 1e-8)

    return {
        'centroid': float(centroid),
        'energy': float(np.log1p(energy)),
        'bandwidth': float(bw),
        'entropy': float(entropy),
        'low_ratio': float(low_r),
        'high_ratio': float(high_r),
    }


def predict_next(history, prop_name):
    """Predict next value from history using expansion (not compression).

    Instead of fitting a line (compression), we expand into possible futures:
    - Momentum: where is it heading?
    - Mean reversion: where does it tend to return?
    - Surprise history: has it been predictable or chaotic?

    Weight these based on how well each has worked so far.
    """
    vals = [h[prop_name] for h in history]
    if len(vals) < 2:
        return vals[-1]  # no prediction possible yet

    # Expansion approach: maintain multiple hypotheses
    # H1: Momentum (last change continues)
    momentum = vals[-1] + (vals[-1] - vals[-2])

    # H2: Mean reversion (return to running mean)
    mean_val = np.mean(vals)
    reversion = vals[-1] + 0.3 * (mean_val - vals[-1])

    # H3: Persistence (stays the same)
    persist = vals[-1]

    # H4: Periodic (if we have enough history, look for repeats)
    if len(vals) >= 6:
        # Check if 2-frame or 3-frame periodicity exists
        diffs_2 = [abs(vals[i] - vals[i-2]) for i in range(2, len(vals))]
        diffs_3 = [abs(vals[i] - vals[i-3]) for i in range(3, len(vals))]
        period_2 = vals[-2] if np.mean(diffs_2) < np.std(vals) * 0.5 else None
        period_3 = vals[-3] if len(diffs_3) > 0 and np.mean(diffs_3) < np.std(vals) * 0.5 else None
    else:
        period_2 = period_3 = None

    # Weight hypotheses by recent accuracy (if we have history)
    if len(vals) >= 4:
        # How well did each strategy predict the LAST frame?
        prev_mom = vals[-2] + (vals[-2] - vals[-3])
        prev_rev = vals[-2] + 0.3 * (np.mean(vals[:-1]) - vals[-2])
        prev_per = vals[-2]

        actual = vals[-1]
        err_mom = abs(prev_mom - actual)
        err_rev = abs(prev_rev - actual)
        err_per = abs(prev_per - actual)

        # Softmax-ish weighting (lower error = higher weight)
        errs = np.array([err_mom, err_rev, err_per]) + 1e-8
        weights = 1.0 / errs
        weights = weights / weights.sum()

        prediction = weights[0] * momentum + weights[1] * reversion + weights[2] * persist
    else:
        prediction = 0.4 * persist + 0.3 * reversion + 0.3 * momentum

    # Add periodic hypothesis if detected
    if period_2 is not None:
        prediction = 0.7 * prediction + 0.3 * period_2
    elif period_3 is not None:
        prediction = 0.8 * prediction + 0.2 * period_3

    return float(prediction)


def eigenmode_analysis(frames_so_far):
    """Track the eigenmode structure of the mel correlation matrix.
    The vibrating membrane approach: modes of the correlation structure."""
    if len(frames_so_far) < 5:
        return None

    # Use last N frames as a window
    window = np.array(frames_so_far[-min(10, len(frames_so_far)):])
    if window.shape[0] < 3:
        return None

    # Correlation matrix of mel bands across time
    corr = np.corrcoef(window.T)  # 40x40 correlation matrix
    corr = np.nan_to_num(corr, nan=0.0)

    # Eigenvalues = modes of vibration
    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

    # Effective dimensionality (participation ratio)
    ev_norm = eigenvalues / (eigenvalues.sum() + 1e-8)
    participation = 1.0 / (np.sum(ev_norm**2) + 1e-8)

    # How concentrated is energy in top modes?
    top3_ratio = eigenvalues[:3].sum() / (eigenvalues.sum() + 1e-8)

    return {
        'top_eigenvalue': float(eigenvalues[0]),
        'participation_ratio': float(participation),
        'top3_concentration': float(top3_ratio),
        'effective_dims': float(participation),
    }


def listen_to_sample(mel_log, category, sample_idx):
    """Listen to a mel spectrogram frame by frame. Predict. Measure."""
    ml = np.exp(mel_log)  # convert to linear
    T, n_bands = ml.shape

    history = []
    predictions = []
    errors = []
    surprises = []
    eigenmode_history = []
    frames_raw = []

    prop_names = ['centroid', 'energy', 'bandwidth', 'entropy', 'low_ratio', 'high_ratio']

    for t in range(T):
        frame = ml[t]
        props = frame_properties(frame)
        frames_raw.append(frame)

        # Make prediction BEFORE seeing this frame (if we have history)
        if len(history) >= 2:
            pred = {p: predict_next(history, p) for p in prop_names}
            predictions.append(pred)

            # Compute prediction error
            frame_error = {}
            for p in prop_names:
                err = abs(props[p] - pred[p])
                frame_error[p] = err
            errors.append(frame_error)

            # Total surprise (normalized)
            total_surprise = sum(frame_error[p] / (abs(props[p]) + 1e-8) for p in prop_names) / len(prop_names)
            surprises.append(total_surprise)

        # Track eigenmode structure
        eig = eigenmode_analysis(frames_raw)
        if eig:
            eigenmode_history.append(eig)

        history.append(props)

    return {
        'history': history,
        'predictions': predictions,
        'errors': errors,
        'surprises': surprises,
        'eigenmodes': eigenmode_history,
    }


def run(data_dir='data/training/mel/esc50'):
    meta = json.load(open('data/clap_meta.json'))
    cats = meta['categories']

    # Pick diverse samples
    test_indices = [9, 10, 22, 32, 12, 16, 19, 25]  # birds, vacuum, dog, chainsaw, thunder, crow, clapping, fireworks
    test_names = {9: 'chirping_birds', 10: 'vacuum_cleaner', 22: 'dog', 32: 'chainsaw',
                  12: 'thunderstorm', 16: 'crow', 19: 'clapping', 25: 'fireworks'}

    results = {}

    for idx in test_indices:
        mf = Path(data_dir) / f'{idx}.npy'
        if not mf.exists():
            continue
        m = np.load(mf)
        cat = cats[idx] if idx < len(cats) else 'unknown'
        print(f"\n{'='*65}")
        print(f"LISTENING TO: {cat} (sample {idx}, {m.shape[0]} frames = {m.shape[0]/10:.1f}s)")
        print(f"{'='*65}")

        result = listen_to_sample(m, cat, idx)

        # Summarize what we "heard"
        h = result['history']
        print(f"\n  Perception summary:")
        print(f"    Frames: {len(h)}")

        # Opening (first 5 frames)
        if len(h) >= 5:
            open_e = np.mean([h[t]['energy'] for t in range(5)])
            open_c = np.mean([h[t]['centroid'] for t in range(5)])
            open_bw = np.mean([h[t]['bandwidth'] for t in range(5)])
            print(f"    Opening:  energy={open_e:.2f}  centroid={open_c:.3f}  bandwidth={open_bw:.3f}")

        # Middle
        mid = len(h) // 2
        if len(h) >= 10:
            mid_e = np.mean([h[t]['energy'] for t in range(mid-2, mid+3)])
            mid_c = np.mean([h[t]['centroid'] for t in range(mid-2, mid+3)])
            print(f"    Middle:   energy={mid_e:.2f}  centroid={mid_c:.3f}")

        # Ending
        if len(h) >= 5:
            end_e = np.mean([h[t]['energy'] for t in range(-5, 0)])
            end_c = np.mean([h[t]['centroid'] for t in range(-5, 0)])
            print(f"    Ending:   energy={end_e:.2f}  centroid={end_c:.3f}")

        # Prediction accuracy
        if result['errors']:
            prop_names = ['centroid', 'energy', 'bandwidth', 'entropy', 'low_ratio', 'high_ratio']
            print(f"\n  Prediction accuracy (lower = better):")
            for p in prop_names:
                errs = [e[p] for e in result['errors']]
                vals = [h[p] for h in result['history'][2:]]
                if np.std(vals) > 1e-10:
                    # Normalize by value range
                    nrmse = np.sqrt(np.mean(np.array(errs)**2)) / (np.std(vals) + 1e-8)
                    r2 = 1 - np.sum(np.array(errs)**2) / (np.sum((np.array(vals) - np.mean(vals))**2) + 1e-8)
                    print(f"    {p:12s}: NRMSE={nrmse:.3f}  R²={r2:.3f}")

            # Overall prediction R²
            all_pred_vals = []
            all_true_vals = []
            for t, (pred, true) in enumerate(zip(result['predictions'], result['history'][2:])):
                for p in prop_names:
                    all_pred_vals.append(pred[p])
                    all_true_vals.append(true[p])
            all_pred = np.array(all_pred_vals)
            all_true = np.array(all_true_vals)
            overall_r2 = 1 - np.sum((all_true - all_pred)**2) / (np.sum((all_true - np.mean(all_true))**2) + 1e-8)
            print(f"\n    OVERALL PREDICTION R²: {overall_r2:.3f}")

        # Surprise analysis
        if result['surprises']:
            surp = np.array(result['surprises'])
            print(f"\n  Surprise profile:")
            print(f"    Mean surprise:   {surp.mean():.3f}")
            print(f"    Max surprise at: frame {np.argmax(surp)+2} ({(np.argmax(surp)+2)/10:.1f}s)")
            print(f"    Min surprise at: frame {np.argmin(surp)+2} ({(np.argmin(surp)+2)/10:.1f}s)")

            # When was I most surprised? (top 3)
            top_surp = np.argsort(surp)[-3:][::-1]
            print(f"    Most surprising moments:")
            for ts in top_surp:
                t = ts + 2  # offset for prediction start
                print(f"      Frame {t} ({t/10:.1f}s): surprise={surp[ts]:.3f}")
                # What changed?
                if ts > 0:
                    for p in prop_names:
                        pred_val = result['predictions'][ts][p]
                        true_val = result['history'][t][p]
                        if abs(true_val - pred_val) > 0.1:
                            print(f"        {p}: predicted {pred_val:.3f}, got {true_val:.3f}")

        # Eigenmode analysis
        if result['eigenmodes']:
            eigs = result['eigenmodes']
            print(f"\n  Vibrating membrane (eigenmode structure):")
            print(f"    Mean effective dimensions: {np.mean([e['effective_dims'] for e in eigs]):.1f}")
            print(f"    Top-3 concentration:       {np.mean([e['top3_concentration'] for e in eigs]):.3f}")

            # Mode shifts: when does the eigenstructure change suddenly?
            if len(eigs) >= 3:
                mode_changes = []
                for i in range(1, len(eigs)):
                    delta_part = abs(eigs[i]['participation_ratio'] - eigs[i-1]['participation_ratio'])
                    delta_conc = abs(eigs[i]['top3_concentration'] - eigs[i-1]['top3_concentration'])
                    mode_changes.append(delta_part + delta_conc)

                if mode_changes:
                    mc = np.array(mode_changes)
                    biggest_shift = np.argmax(mc)
                    print(f"    Biggest mode shift at:     frame {biggest_shift+6} ({(biggest_shift+6)/10:.1f}s)")
                    print(f"    Mode shift magnitude:      {mc[biggest_shift]:.3f}")

                    # Does mode shift correlate with surprise?
                    if len(result['surprises']) >= len(mode_changes):
                        surp_aligned = result['surprises'][:len(mode_changes)]
                        r = np.corrcoef(mc, surp_aligned[:len(mc)])[0, 1]
                        if not np.isnan(r):
                            print(f"    Mode-shift ↔ surprise:     r = {r:+.3f}")

        # Alive vs mechanical assessment
        if result['errors']:
            energy_var = np.std([h['energy'] for h in result['history']])
            irregularity_vals = [abs(result['surprises'][i] - result['surprises'][i-1])
                                for i in range(1, len(result['surprises']))] if len(result['surprises']) > 1 else [0]
            irregularity = np.mean(irregularity_vals)
            predictability = 1.0 - np.mean(result['surprises']) if result['surprises'] else 0.5
            print(f"\n  Life-machine assessment:")
            print(f"    Energy variability: {energy_var:.3f}")
            print(f"    Irregularity:       {irregularity:.3f}")
            print(f"    Predictability:     {predictability:.3f}")
            if energy_var > 0.5 and irregularity > 0.05:
                print(f"    Assessment:         Feels ALIVE")
            elif energy_var < 0.2 and irregularity < 0.02:
                print(f"    Assessment:         Feels MECHANICAL")
            else:
                print(f"    Assessment:         Ambiguous / in-between")

        results[cat] = {
            'overall_r2': float(overall_r2) if result['errors'] else None,
            'mean_surprise': float(np.mean(result['surprises'])) if result['surprises'] else None,
            'n_frames': len(h),
        }

    # Summary
    print(f"\n{'='*65}")
    print(f"LISTENING SUMMARY")
    print(f"{'='*65}")
    print(f"\n  {'Category':20s} {'Pred R²':>8s} {'Surprise':>10s} {'Assessment':>12s}")
    for cat, r in results.items():
        r2 = r['overall_r2']
        surp = r['mean_surprise']
        if r2 is not None:
            assess = "ALIVE" if surp and surp > 0.15 else "MECHANICAL" if surp and surp < 0.08 else "ambiguous"
            print(f"  {cat:20s} {r2:8.3f} {surp:10.3f} {assess:>12s}")

    # THE KEY TEST: Can we predict mechanical better than alive?
    # (Because mechanical = compressed/periodic, alive = expanded/surprising)
    alive_cats = {'chirping_birds', 'dog', 'crow'}
    mech_cats = {'vacuum_cleaner', 'chainsaw'}
    alive_r2 = [results[c]['overall_r2'] for c in alive_cats if c in results and results[c]['overall_r2'] is not None]
    mech_r2 = [results[c]['overall_r2'] for c in mech_cats if c in results and results[c]['overall_r2'] is not None]

    if alive_r2 and mech_r2:
        print(f"\n  Mean prediction R² for ALIVE sounds:      {np.mean(alive_r2):.3f}")
        print(f"  Mean prediction R² for MECHANICAL sounds:  {np.mean(mech_r2):.3f}")
        if np.mean(mech_r2) > np.mean(alive_r2):
            print(f"  → Mechanical IS more predictable. Compression confirmed.")
            print(f"  → Alive sounds expand into more temporal dimensions.")
        else:
            print(f"  → Surprising: alive sounds were MORE predictable.")

    json.dump(results, open('data/wc_listen_predict_results.json', 'w'), indent=2)
    print("\nSaved")


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    a = p.parse_args(); run(a.data_dir)
