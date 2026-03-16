#!/usr/bin/env python3
"""
Wildcard: Musicality and Prediction
=====================================
Music is the most interesting case for the alive/mechanical spectrum:
it's CREATED by alive things but has mechanical-level structure.

Music is intentionally predictable-with-surprises. It sets up expectations
(verse, chorus, rhythm) then violates them (bridge, key change, fill).

Test: Measure how PREDICTABLE each sound is at different timescales:
- Frame-level (100ms): raw acoustic predictability
- Pattern-level (0.5-2s): rhythmic/periodic structure
- Structural-level (2-5s): phrase/section predictability

Hypothesis: Music-like sounds have HIGH pattern-level predictability
but MODERATE structural-level surprise. Mechanical sounds are predictable
at ALL levels. Alive sounds are unpredictable at ALL levels.

Also: which ESC-50 sounds are most "musical"? (structured prediction with surprise)
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


ALIVE_CATS = {
    'dog', 'cat', 'rooster', 'pig', 'cow', 'frog', 'hen', 'insects',
    'sheep', 'crow', 'chirping_birds', 'crying_baby', 'sneezing',
    'coughing', 'laughing', 'breathing', 'snoring', 'drinking_sipping',
}
MECHANICAL_CATS = {
    'helicopter', 'chainsaw', 'engine', 'train', 'airplane',
    'vacuum_cleaner', 'washing_machine', 'clock_tick', 'clock_alarm',
    'keyboard_typing', 'mouse_click', 'hand_saw', 'siren'
}


def compute_multi_scale_predictability(mel_frames):
    """Measure predictability at multiple timescales."""
    ml = np.exp(mel_frames)
    T, n = ml.shape
    if T < 10:
        return None

    fe = np.sum(ml ** 2, axis=1)  # energy per frame
    centroid = np.array([np.sum(np.linspace(0,1,n) * (ml[t]+1e-8)) / (np.sum(ml[t]+1e-8))
                         for t in range(T)])

    results = {}

    # === FRAME-LEVEL PREDICTABILITY (100ms) ===
    # How well does the previous frame predict the next?
    if T >= 4:
        frame_errors = []
        for t in range(2, T):
            # Simple prediction: weighted average of last 2 frames
            pred = 0.7 * fe[t-1] + 0.3 * fe[t-2]
            frame_errors.append(abs(pred - fe[t]) / (abs(fe[t]) + 1e-8))
        frame_pred = 1.0 - np.mean(frame_errors)
        results['frame_predictability'] = float(np.clip(frame_pred, 0, 1))

        # Same for centroid
        cent_errors = []
        for t in range(2, T):
            pred = 0.7 * centroid[t-1] + 0.3 * centroid[t-2]
            cent_errors.append(abs(pred - centroid[t]) / (abs(centroid[t]) + 1e-8))
        results['frame_pred_centroid'] = float(np.clip(1.0 - np.mean(cent_errors), 0, 1))
    else:
        results['frame_predictability'] = 0.5
        results['frame_pred_centroid'] = 0.5

    # === PATTERN-LEVEL PREDICTABILITY (0.5-2s = 5-20 frames) ===
    # Does the energy have periodic structure?
    if T >= 20:
        ec = fe - fe.mean()
        norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            # Look for peaks in the 5-20 frame lag range (0.5-2s)
            pattern_range = ac[5:min(21, len(ac))]
            if len(pattern_range) > 0:
                pattern_periodicity = float(np.max(pattern_range))
                pattern_period_frames = int(np.argmax(pattern_range)) + 5
            else:
                pattern_periodicity = 0.0
                pattern_period_frames = 0
        else:
            pattern_periodicity = 0.0
            pattern_period_frames = 0

        results['pattern_periodicity'] = pattern_periodicity
        results['pattern_period_seconds'] = pattern_period_frames / 10.0
    else:
        results['pattern_periodicity'] = 0.0
        results['pattern_period_seconds'] = 0.0

    # === STRUCTURAL-LEVEL PREDICTABILITY (2-5s) ===
    # Do segments repeat at longer timescales?
    if T >= 30:
        # Split into ~1s segments and compute similarity
        seg_len = 10  # 1 second
        n_segs = T // seg_len
        if n_segs >= 3:
            seg_spectra = [ml[i*seg_len:(i+1)*seg_len].mean(axis=0) for i in range(n_segs)]
            seg_energies = [fe[i*seg_len:(i+1)*seg_len].mean() for i in range(n_segs)]

            # Spectral self-similarity matrix
            sim_matrix = np.zeros((n_segs, n_segs))
            for i in range(n_segs):
                for j in range(n_segs):
                    r = np.corrcoef(seg_spectra[i], seg_spectra[j])[0, 1]
                    sim_matrix[i, j] = r if not np.isnan(r) else 0

            # Off-diagonal similarity (how much do non-adjacent segments resemble each other?)
            off_diag = []
            for i in range(n_segs):
                for j in range(i+2, n_segs):  # skip adjacent
                    off_diag.append(sim_matrix[i, j])
            structural_repetition = float(np.mean(off_diag)) if off_diag else 0.0

            # Energy contour predictability
            seg_e = np.array(seg_energies)
            if len(seg_e) >= 3:
                structural_surprise = np.std(np.diff(seg_e)) / (np.mean(seg_e) + 1e-8)
            else:
                structural_surprise = 0.0
        else:
            structural_repetition = 0.0
            structural_surprise = 0.0

        results['structural_repetition'] = float(structural_repetition)
        results['structural_surprise'] = float(structural_surprise)
    else:
        results['structural_repetition'] = 0.0
        results['structural_surprise'] = 0.0

    # === MUSICALITY SCORE ===
    # Music = high pattern predictability + moderate structural surprise
    # (predictable rhythm, surprising melody/dynamics)
    pattern_p = results.get('pattern_periodicity', 0)
    struct_s = results.get('structural_surprise', 0)
    struct_r = results.get('structural_repetition', 0)
    frame_p = results.get('frame_predictability', 0.5)

    # Musicality: structured (periodic + repetitive) but with controlled surprise
    musicality = (0.3 * pattern_p +                    # has rhythm
                  0.2 * struct_r +                      # has recurring sections
                  0.2 * min(struct_s, 0.5) * 2 +       # some surprise (peaks at 0.5)
                  0.3 * frame_p)                        # locally predictable
    results['musicality'] = float(musicality)

    # === PREDICTION HORIZON ===
    # How far ahead can you predict? (at what lag does autocorrelation drop below 0.3?)
    if T >= 10:
        ec = fe - fe.mean()
        norm = np.sum(ec**2)
        if norm > 1e-10:
            ac = np.correlate(ec, ec, 'full')[len(ec)-1:]
            ac = ac / (norm + 1e-8)
            # Find where autocorrelation drops below 0.3
            horizon = T  # default: always predictable
            for lag in range(1, len(ac)):
                if ac[lag] < 0.3:
                    horizon = lag
                    break
            results['prediction_horizon_frames'] = int(horizon)
            results['prediction_horizon_seconds'] = float(horizon / 10.0)
        else:
            results['prediction_horizon_frames'] = 1
            results['prediction_horizon_seconds'] = 0.1
    else:
        results['prediction_horizon_frames'] = 1
        results['prediction_horizon_seconds'] = 0.1

    # === SURPRISE DISTRIBUTION ===
    # Music has surprise at SPECIFIC moments (beats, transitions)
    # Noise has surprise everywhere. Mechanism has surprise nowhere.
    if T >= 6:
        frame_surp = np.abs(np.diff(fe)) / (np.mean(fe) + 1e-8)
        # Kurtosis of surprise distribution
        # High kurtosis = surprise concentrated in peaks (musical)
        # Low kurtosis = surprise spread evenly (noise) or absent (mechanical)
        if np.std(frame_surp) > 1e-10:
            surprise_kurtosis = float(np.mean(((frame_surp - frame_surp.mean()) / (np.std(frame_surp) + 1e-8))**4) - 3)
        else:
            surprise_kurtosis = 0.0
        results['surprise_kurtosis'] = surprise_kurtosis
        results['surprise_mean'] = float(frame_surp.mean())
        results['surprise_std'] = float(frame_surp.std())
    else:
        results['surprise_kurtosis'] = 0.0
        results['surprise_mean'] = 0.0
        results['surprise_std'] = 0.0

    return results


def run(data_dir='data/training/mel/esc50'):
    meta = json.load(open('data/clap_meta.json'))
    cats = meta['categories']
    uc = sorted(set(c for c in cats if c not in ('ambient', 'mixed', 'music')))
    c2i = {c: i for i, c in enumerate(uc)}
    cn = {i: c for c, i in c2i.items()}

    props_list, catids, catnames = [], [], []
    for mf in sorted(Path(data_dir).glob('*.npy'), key=lambda p: int(p.stem)):
        m = np.load(mf)
        if m.shape[0] < 10: continue
        idx = int(mf.stem)
        if idx >= len(cats) or cats[idx] in ('ambient', 'mixed', 'music'): continue
        p = compute_multi_scale_predictability(m)
        if p is None: continue
        props_list.append(p)
        catids.append(c2i[cats[idx]])
        catnames.append(cats[idx])
    print(f"Samples: {len(props_list)}")

    prop_names = list(props_list[0].keys())
    V = np.array([[p[k] for k in prop_names] for p in props_list])
    ca = np.array(catids)

    results = {}

    # ================================================================
    # TEST 1: Predictability profiles — alive vs mechanical
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 1: Multi-scale predictability — alive vs mechanical")
    print("=" * 65)

    labels = np.array([1 if catnames[i] in ALIVE_CATS else
                       0 if catnames[i] in MECHANICAL_CATS else -1
                       for i in range(len(catnames))])

    key_props = ['frame_predictability', 'pattern_periodicity', 'structural_repetition',
                 'structural_surprise', 'musicality', 'prediction_horizon_seconds',
                 'surprise_kurtosis']

    print(f"\n  {'Property':28s} {'Alive':>8s} {'Mech':>8s} {'Delta':>8s}")
    t1 = {}
    for name in key_props:
        idx = prop_names.index(name)
        alive_vals = V[labels == 1, idx]
        mech_vals = V[labels == 0, idx]
        delta = alive_vals.mean() - mech_vals.mean()
        print(f"  {name:28s} {alive_vals.mean():8.3f} {mech_vals.mean():8.3f} {delta:+8.3f}")
        t1[name] = {'alive': float(alive_vals.mean()), 'mech': float(mech_vals.mean())}
    results['test1'] = t1

    # ================================================================
    # TEST 2: Which sounds are most "musical"?
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 2: Musicality ranking (structured prediction + surprise)")
    print("=" * 65)

    mus_idx = prop_names.index('musicality')
    cat_musicality = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        cat_musicality[nm] = float(V[cmask, mus_idx].mean())

    sorted_mus = sorted(cat_musicality.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Category':20s} {'Musicality':>10s} {'Group':>12s}")
    for nm, m in sorted_mus:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        bar = "#" * int(m * 30)
        print(f"  {nm:20s} {m:10.3f} {group:>12s}  {bar}")
    results['musicality_ranking'] = cat_musicality

    # ================================================================
    # TEST 3: Prediction horizon — how far into the future is visible?
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 3: Prediction horizon (how much future is visible?)")
    print("=" * 65)

    horiz_idx = prop_names.index('prediction_horizon_seconds')
    cat_horizon = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        cat_horizon[nm] = float(V[cmask, horiz_idx].mean())

    sorted_hor = sorted(cat_horizon.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Category':20s} {'Horizon(s)':>10s} {'Group':>12s}")
    for nm, h in sorted_hor[:10]:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"  {nm:20s} {h:10.1f} {group:>12s}  (most predictable future)")
    print(f"  {'...':20s}")
    for nm, h in sorted_hor[-10:]:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"  {nm:20s} {h:10.1f} {group:>12s}")

    alive_hor = [cat_horizon[nm] for nm in cat_horizon if nm in ALIVE_CATS]
    mech_hor = [cat_horizon[nm] for nm in cat_horizon if nm in MECHANICAL_CATS]
    print(f"\n  Mean horizon ALIVE:      {np.mean(alive_hor):.1f}s")
    print(f"  Mean horizon MECHANICAL: {np.mean(mech_hor):.1f}s")
    print(f"  → Mechanical sounds have {np.mean(mech_hor)/np.mean(alive_hor):.1f}x more visible future")
    results['horizons'] = cat_horizon

    # ================================================================
    # TEST 4: Surprise distribution — where does surprise HAPPEN?
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 4: Surprise distribution (concentrated vs spread)")
    print("=" * 65)

    kurt_idx = prop_names.index('surprise_kurtosis')
    cat_kurt = {}
    for c in sorted(set(catids)):
        nm = cn[c]
        cmask = ca == c
        if cmask.sum() == 0: continue
        cat_kurt[nm] = float(V[cmask, kurt_idx].mean())

    # High kurtosis = surprise at specific moments (musical)
    # Low/negative kurtosis = surprise everywhere or nowhere
    sorted_kurt = sorted(cat_kurt.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Most 'musical' surprise distribution (concentrated surprises):")
    for nm, k in sorted_kurt[:10]:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"    {nm:20s}: kurtosis={k:+.2f} {group}")

    print(f"\n  Most 'noisy' surprise distribution (uniform surprises):")
    for nm, k in sorted_kurt[-5:]:
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"    {nm:20s}: kurtosis={k:+.2f} {group}")

    results['surprise_distribution'] = cat_kurt

    # ================================================================
    # TEST 5: The three modes — Mechanical / Musical / Wild
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 5: Three temporal modes")
    print("=" * 65)

    # Classify each category into one of three modes:
    # MECHANICAL: high predictability at all scales
    # MUSICAL: high pattern predictability + concentrated surprise
    # WILD: low predictability, spread surprise (genuinely alive)
    print(f"\n  {'Category':20s} {'Mode':>10s} {'Pattern':>8s} {'Kurt':>8s} {'Horizon':>8s}")

    modes = {}
    for nm in sorted(cat_musicality.keys()):
        mus = cat_musicality.get(nm, 0)
        hor = cat_horizon.get(nm, 0)
        kurt = cat_kurt.get(nm, 0)
        pat_idx = prop_names.index('pattern_periodicity')
        pat = float(V[ca == c2i[nm], pat_idx].mean()) if nm in c2i else 0

        # Classification logic
        if hor > 3.0 and pat > 0.3:
            mode = "MECHANICAL"
        elif kurt > 2.0 and pat > 0.1:
            mode = "MUSICAL"
        elif hor < 1.5:
            mode = "WILD"
        else:
            mode = "mixed"

        modes[nm] = mode
        group = "ALIVE" if nm in ALIVE_CATS else "MECH" if nm in MECHANICAL_CATS else "ambig"
        print(f"  {nm:20s} {mode:>10s} {pat:8.3f} {kurt:8.2f} {hor:8.1f}  ({group})")

    from collections import Counter
    mode_counts = Counter(modes.values())
    print(f"\n  Mode distribution: {dict(mode_counts)}")

    # How many alive sounds are WILD vs MUSICAL?
    alive_modes = Counter(modes[nm] for nm in modes if nm in ALIVE_CATS)
    mech_modes = Counter(modes[nm] for nm in modes if nm in MECHANICAL_CATS)
    print(f"  Alive sound modes:      {dict(alive_modes)}")
    print(f"  Mechanical sound modes:  {dict(mech_modes)}")

    results['modes'] = modes

    # ================================================================
    # TEST 6: Listen to "ours" samples (longer recordings)
    # ================================================================
    ours_dir = Path('data/training/mel/ours')
    if ours_dir.exists():
        ours_files = sorted(ours_dir.glob('*.npy'))
        if ours_files:
            print(f"\n" + "=" * 65)
            print(f"TEST 6: Listening to custom recordings")
            print("=" * 65)
            for mf in ours_files:
                m = np.load(mf)
                if m.shape[0] < 10: continue
                p = compute_multi_scale_predictability(m)
                if p is None: continue
                print(f"\n  {mf.name} ({m.shape[0]} frames = {m.shape[0]/10:.1f}s):")
                print(f"    Frame predict:   {p['frame_predictability']:.3f}")
                print(f"    Pattern period:  {p['pattern_periodicity']:.3f} (period={p['pattern_period_seconds']:.1f}s)")
                print(f"    Struct repeat:   {p['structural_repetition']:.3f}")
                print(f"    Struct surprise: {p['structural_surprise']:.3f}")
                print(f"    Musicality:      {p['musicality']:.3f}")
                print(f"    Pred horizon:    {p['prediction_horizon_seconds']:.1f}s")
                print(f"    Surprise kurt:   {p['surprise_kurtosis']:.2f}")

                # Mode classification
                if p['prediction_horizon_seconds'] > 3.0 and p['pattern_periodicity'] > 0.3:
                    mode = "MECHANICAL"
                elif p['surprise_kurtosis'] > 2.0:
                    mode = "MUSICAL"
                elif p['prediction_horizon_seconds'] < 1.5:
                    mode = "WILD"
                else:
                    mode = "mixed"
                print(f"    Mode:            {mode}")

    # Verdict
    mech_more_predictable = np.mean(mech_hor) > np.mean(alive_hor)
    musical_sounds_exist = sum(1 for m in modes.values() if m == "MUSICAL") > 3
    wild_alive = sum(1 for nm, m in modes.items() if m == "WILD" and nm in ALIVE_CATS)

    print(f"\nSUMMARY:")
    print(f"  Mechanical more predictable: {mech_more_predictable}")
    print(f"  Musical-mode sounds:         {sum(1 for m in modes.values() if m == 'MUSICAL')}")
    print(f"  Wild alive sounds:           {wild_alive}")

    verdict = "STRONG" if mech_more_predictable and musical_sounds_exist else "PARTIAL"
    print(f"\nVERDICT: {verdict}")
    results['verdict'] = verdict

    json.dump(results, open('data/wc_musicality_prediction_results.json', 'w'), indent=2)
    print("Saved")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/training/mel/esc50')
    a = p.parse_args()
    run(a.data_dir)
