#!/usr/bin/env python3
"""
Frequency Explorer — Comprehensive multi-modal audio analysis tool.

Takes audio files (.wav) or pre-computed mel spectrograms (.npy) and produces
a rich cross-modal frequency analysis: what sound "looks like," "feels like,"
its emotional character, chemical analog, spatial properties, temporal dynamics.

Reuses property extraction functions from the wildcard experiment scripts.

Usage:
    # Single mel file
    python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy

    # WAV file
    python3 scripts/frequency_explorer.py recording.wav

    # Batch mode — all mels in a directory
    python3 scripts/frequency_explorer.py data/training/mel/esc50/ --batch

    # Filter by category
    python3 scripts/frequency_explorer.py data/training/mel/esc50/ --batch --category dog

    # Compare specific files
    python3 scripts/frequency_explorer.py file1.npy file2.npy file3.npy

    # Output JSON
    python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy --json output.json

    # Generate visualization
    python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy --plot

    # Temporal windowed analysis
    python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy --windows 0.5,1.0,2.0
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Import cross-modal property functions from wildcard scripts ───

try:
    from scripts.wc_light_as_sound import compute_visual_properties
    from scripts.wc_sound_as_touch import compute_tactile_properties
    from scripts.wc_sound_as_emotion import compute_emotion_properties
    from scripts.wc_sound_as_chemistry import compute_chemistry
    from scripts.wc_sound_as_geometry import compute_shape_properties
    from scripts.wc_sound_as_taste import compute_taste_properties
    from scripts.wc_sound_as_weather import compute_weather
    from scripts.wc_sound_as_life import compute_life_properties
    from scripts.wc_sound_as_social import compute_social
    _HAS_CROSSMODAL = True
except ImportError:
    _HAS_CROSSMODAL = False


# ─── Acoustic feature extraction (new — not in wildcard scripts) ───

def compute_acoustic_features(mel_frames):
    """Compute basic acoustic features from log-mel spectrogram.

    Args:
        mel_frames: numpy array (T, n_mels) — log-mel spectrogram

    Returns:
        dict of feature_name -> scalar value
    """
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    freq_bins = np.linspace(0, 1, n_mels)
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # Spectral centroid
    centroid = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum)

    # Spectral bandwidth
    bw_sq = np.sum(((freq_bins - centroid) ** 2) * mean_spectrum) / np.sum(mean_spectrum)
    bandwidth = np.sqrt(max(bw_sq, 0))

    # Spectral flatness (Wiener entropy)
    geo_mean = np.exp(np.mean(np.log(mean_spectrum + 1e-10)))
    flatness = geo_mean / (np.mean(mean_spectrum) + 1e-8)

    # RMS energy
    rms = np.sqrt(np.mean(mel_linear ** 2))
    rms_db = 20 * np.log10(max(rms, 1e-10))

    # Zero-crossing rate proxy (energy envelope sign changes)
    if T >= 4:
        e_centered = frame_energy - frame_energy.mean()
        zcr = np.sum(np.abs(np.diff(np.sign(e_centered)))) / (2 * T)
    else:
        zcr = 0.0

    # Spectral rolloff (frequency below which 85% of energy lies)
    cumsum = np.cumsum(mean_spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = rolloff_idx / n_mels

    # Spectral contrast (difference between peaks and valleys in bands)
    n_bands = min(6, n_mels // 4)
    band_size = n_mels // n_bands if n_bands > 0 else n_mels
    contrasts = []
    for b in range(n_bands):
        band = mean_spectrum[b * band_size:(b + 1) * band_size]
        if len(band) > 1:
            contrasts.append(float(np.max(band) - np.min(band)))
    spectral_contrast = np.mean(contrasts) if contrasts else 0.0

    return {
        'kentron': float(centroid),           # κέντρον — spectral center point
        'euros': float(bandwidth),             # εὖρος — breadth, bandwidth
        'homalotes': float(flatness),          # ὁμαλότης — evenness (spectral flatness)
        'climax': float(rolloff),              # climax (Latin) — spectral rolloff point
        'antithesis': float(spectral_contrast),# ἀντίθεσις — opposition (spectral contrast)
        'dynamis': float(rms),                 # δύναμις — power, force (RMS energy)
        'dynamis_db': float(rms_db),           # δύναμις in decibels
        'metabole': float(zcr),                # μεταβολή — change (zero-crossing rate)
    }


def compute_temporal_dynamics(mel_frames, frame_rate=10.0):
    """Compute temporal dynamics features from log-mel spectrogram.

    Args:
        mel_frames: numpy array (T, n_mels)
        frame_rate: frames per second (default 10Hz for ESC-50 mels)

    Returns:
        dict of feature_name -> scalar or array
    """
    mel_linear = np.exp(mel_frames)
    T = mel_linear.shape[0]
    frame_energy = np.sum(mel_linear ** 2, axis=1)

    # Energy envelope stats
    energy_mean = float(np.mean(frame_energy))
    energy_std = float(np.std(frame_energy))
    energy_max = float(np.max(frame_energy))
    energy_dynamic_range = float(energy_max / (np.min(frame_energy) + 1e-8))

    # Spectral flux over time
    if T >= 2:
        flux = np.sqrt(np.sum(np.diff(mel_linear, axis=0) ** 2, axis=1))
        flux_mean = float(np.mean(flux))
        flux_std = float(np.std(flux))
        flux_max = float(np.max(flux))
    else:
        flux_mean = flux_std = flux_max = 0.0
        flux = np.array([0.0])

    # Predictability via autocorrelation
    if T >= 10:
        e_centered = frame_energy - frame_energy.mean()
        norm = np.sum(e_centered ** 2)
        if norm > 1e-10:
            autocorr = np.correlate(e_centered, e_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (norm + 1e-8)
            # Lag-1 autocorrelation = short-term predictability
            lag1 = float(autocorr[1]) if len(autocorr) > 1 else 0.0
            # Peak autocorrelation (periodicity)
            peak_ac = float(np.max(autocorr[1:])) if len(autocorr) > 1 else 0.0
        else:
            lag1 = peak_ac = 0.0
    else:
        lag1 = peak_ac = 0.0

    # Onset detection (energy rise rate)
    if T >= 4:
        energy_diff = np.diff(frame_energy)
        n_onsets = int(np.sum(energy_diff > np.mean(np.abs(energy_diff)) + np.std(energy_diff)))
        onset_density = float(n_onsets / (T / frame_rate))  # onsets per second
    else:
        onset_density = 0.0

    # Duration and temporal shape
    duration_s = float(T / frame_rate)

    # Attack-sustain-decay shape
    peak_idx = np.argmax(frame_energy)
    attack_ratio = float(peak_idx / max(T - 1, 1))  # 0=immediate peak, 1=peak at end
    if peak_idx < T - 1:
        decay_energy = frame_energy[peak_idx:]
        sustain_ratio = float(np.mean(decay_energy > 0.5 * frame_energy[peak_idx]))
    else:
        sustain_ratio = 1.0

    return {
        'chronos': duration_s,                 # χρόνος — time, duration
        'dynamis_mese': energy_mean,           # δύναμις μέση — mean power
        'dynamis_diaphora': energy_std,        # δύναμις διαφορά — power variation
        'dynamis_heurema_db': float(10 * np.log10(max(energy_dynamic_range, 1e-10))),  # dynamic range
        'rhoe_mese': flux_mean,                # ῥοή μέση — mean flow (spectral flux)
        'rhoe_diaphora': flux_std,             # ῥοή variation
        'rhoe_megiste': flux_max,              # ῥοή μεγίστη — maximum flow
        'pronoia': lag1,                       # πρόνοια — foresight, predictability
        'periodos': peak_ac,                   # περίοδος — going-around, periodicity
        'pyknotes_hormon': onset_density,      # πυκνότης ὁρμῶν — density of onsets
        'horme': attack_ratio,                 # ὁρμή — impulse, attack
        'epimone': sustain_ratio,              # ἐπιμονή — persistence, sustain
    }


def compute_harmonic_features(mel_frames):
    """Compute harmonic analysis features from log-mel spectrogram.

    Args:
        mel_frames: numpy array (T, n_mels)

    Returns:
        dict of harmonic features
    """
    mel_linear = np.exp(mel_frames)
    n_mels = mel_linear.shape[1]
    T = mel_linear.shape[0]
    mean_spectrum = mel_linear.mean(axis=0) + 1e-8

    # Estimate fundamental frequency (spectral autocorrelation method)
    spec_centered = mean_spectrum - mean_spectrum.mean()
    spec_norm = np.sum(spec_centered ** 2)
    if spec_norm > 1e-10:
        spec_ac = np.correlate(spec_centered, spec_centered, mode='full')
        spec_ac = spec_ac[len(spec_ac) // 2:]
        spec_ac = spec_ac / (spec_norm + 1e-8)

        # Find peaks in spectral autocorrelation (harmonics)
        if len(spec_ac) > 2:
            # Look for first significant peak after lag 0
            peaks = []
            for i in range(2, len(spec_ac) - 1):
                if spec_ac[i] > spec_ac[i - 1] and spec_ac[i] > spec_ac[i + 1]:
                    if spec_ac[i] > 0.1:  # significance threshold
                        peaks.append((i, float(spec_ac[i])))

            if peaks:
                f0_bin = peaks[0][0]
                f0_estimate = float(f0_bin / n_mels)  # normalized
                harmonic_strength = peaks[0][1]
                n_harmonics = len(peaks)
            else:
                f0_estimate = 0.0
                harmonic_strength = 0.0
                n_harmonics = 0
        else:
            f0_estimate = harmonic_strength = 0.0
            n_harmonics = 0
    else:
        f0_estimate = harmonic_strength = 0.0
        n_harmonics = 0
        spec_ac = np.array([1.0])

    # Harmonic-to-noise ratio proxy
    # Compare spectral peaks to spectral valleys
    if n_mels >= 4:
        sorted_spec = np.sort(mean_spectrum)
        top_quarter = sorted_spec[3 * n_mels // 4:]
        bottom_quarter = sorted_spec[:n_mels // 4]
        hnr_proxy = float(np.mean(top_quarter) / (np.mean(bottom_quarter) + 1e-8))
        hnr_db = float(10 * np.log10(max(hnr_proxy, 1e-10)))
    else:
        hnr_db = 0.0

    # Spectral crest (peakiness — tonal sounds have high crest)
    crest = float(np.max(mean_spectrum) / (np.mean(mean_spectrum) + 1e-8))

    # Inharmonicity (deviation from perfect harmonic series)
    if n_harmonics >= 2 and len(spec_ac) > 2:
        # Check regularity of peak spacing
        peak_positions = [p[0] for p in peaks[:min(5, len(peaks))]]
        if len(peak_positions) >= 2:
            spacings = np.diff(peak_positions)
            inharmonicity = float(np.std(spacings) / (np.mean(spacings) + 1e-8))
        else:
            inharmonicity = 1.0
    else:
        inharmonicity = 1.0  # no harmonics = maximally inharmonic

    return {
        'arche_tonou': f0_estimate,            # ἀρχή τόνου — fundamental tone
        'harmonia': float(harmonic_strength),   # ἁρμονία — fitting-together strength
        'arithmos': n_harmonics,                # ἀριθμός — number of harmonics
        'katharotes_db': hnr_db,               # καθαρότης — purity (harmonic-to-noise)
        'akme': crest,                          # ἀκμή — peak, zenith (spectral crest)
        'anharmonia': inharmonicity,            # ἀναρμοστία — discord, unfittingness
    }


# ─── Windowed temporal analysis ───

def compute_windowed_properties(mel_frames, window_sizes_s, frame_rate=10.0):
    """Compute cross-modal properties in sliding windows to show temporal evolution.

    Args:
        mel_frames: numpy array (T, n_mels)
        window_sizes_s: list of window sizes in seconds
        frame_rate: frames per second

    Returns:
        dict of {window_size: [{time: t, properties: {...}}, ...]}
    """
    T = mel_frames.shape[0]
    results = {}

    for ws in window_sizes_s:
        win_frames = max(int(ws * frame_rate), 4)  # minimum 4 frames
        if win_frames >= T:
            # Window larger than signal — compute once
            props = _compute_all_modalities(mel_frames)
            results[f'{ws}s'] = [{'time': 0.0, 'properties': props}]
            continue

        step = max(win_frames // 2, 1)  # 50% overlap
        windows = []
        for start in range(0, T - win_frames + 1, step):
            chunk = mel_frames[start:start + win_frames]
            t = float(start / frame_rate)
            props = _compute_all_modalities(chunk)
            windows.append({'time': t, 'properties': props})

        results[f'{ws}s'] = windows

    return results


def _compute_all_modalities(mel_frames):
    """Compute all cross-modal properties for a mel chunk. Returns flat dict.

    Uses Greek/Latin translation maps so windowed analysis matches the
    naming convention used by analyze_mel().
    """
    props = {}

    # Translation maps for each modality (English→Greek/Latin)
    TRANSLATION_MAPS = {
        'light': PHOS_MAP,
        'touch': HAPHE_MAP,
        'emotion': PATHOS_MAP,
        'chemistry': CHEMEIA_MAP,
        'geometry': GEOMETRIA_MAP,
        'taste': GEUSIS_MAP,
        'weather': KAIROS_MAP,
        'life': ZOE_MAP,
        'social': KOINONIA_MAP,
    }

    modality_fns = [
        ('light', compute_visual_properties),
        ('touch', compute_tactile_properties),
        ('emotion', compute_emotion_properties),
        ('chemistry', compute_chemistry),
        ('geometry', compute_shape_properties),
        ('taste', compute_taste_properties),
        ('weather', compute_weather),
        ('life', compute_life_properties),
        ('social', compute_social),
    ] if _HAS_CROSSMODAL else []

    for prefix, fn in modality_fns:
        try:
            result = fn(mel_frames)
            keymap = TRANSLATION_MAPS.get(prefix, {})
            translated = _translate_keys(result, keymap)
            for key, val in translated.items():
                props[f'{prefix}.{key}'] = val
        except Exception:
            # Skip modalities that fail on short/degenerate windows
            pass

    # Also add acoustic, harmonic, temporal (already use Greek keys)
    try:
        for key, val in compute_acoustic_features(mel_frames).items():
            props[f'acoustic.{key}'] = val
    except Exception:
        pass
    try:
        for key, val in compute_harmonic_features(mel_frames).items():
            props[f'harmonic.{key}'] = val
    except Exception:
        pass
    try:
        for key, val in compute_temporal_dynamics(mel_frames).items():
            props[f'temporal.{key}'] = val
    except Exception:
        pass

    return props


# ─── Audio loading ───

def load_audio_as_mel(wav_path, n_mels=40, sr=22050, hop_length=2205):
    """Load an audio file and extract log-mel spectrogram matching ESC-50 format.

    Supports any format ffmpeg can decode: wav, m4a, mp3, aac, ogg, flac, etc.

    Args:
        wav_path: path to audio file (wav, m4a, mp3, etc.)
        n_mels: number of mel bands (40 for ESC-50 mels, 80 for MelExtractor)
        sr: sample rate
        hop_length: hop length (2205 for ~10Hz at 22050 sr)

    Returns:
        mel_frames: numpy array (T, n_mels) — log-mel spectrogram
    """
    import librosa

    y, actual_sr = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length,
        n_mels=n_mels, fmin=20.0, fmax=sr / 2
    )
    log_mel = np.log(mel + 1e-6).T  # (T, n_mels)
    return log_mel.astype(np.float32)


def load_mel_file(npy_path):
    """Load a pre-computed mel spectrogram .npy file.

    Returns:
        mel_frames: numpy array (T, n_mels)
    """
    mel = np.load(npy_path)
    if mel.ndim == 1:
        mel = mel.reshape(-1, 40)  # assume 40 mel bands
    return mel


# ─── Category lookup ───

def load_category_map():
    """Load category mapping from clap_meta.json."""
    meta_path = PROJECT_ROOT / 'data' / 'clap_meta.json'
    if meta_path.exists():
        meta = json.load(open(meta_path))
        return meta.get('categories', [])
    return []


# ─── Terminal output formatting ───

MODALITY_ICONS = {
    'acoustic':  'AKOUSTIKA',              # Ακουστικά
    'harmonic':  'HARMONIKA',              # Αρμονικά
    'temporal':  'CHRONIKA',               # Χρονικά
    'light':     'PHOS',                   # Φώς — light
    'touch':     'HAPHE',                  # Αφή — touch
    'emotion':   'PATHOS',                 # Πάθος — feeling/suffering
    'chemistry': 'CHEMEIA',                # Χημεία
    'geometry':  'GEOMETRIA',              # Γεωμετρία
    'taste':     'GEUSIS',                 # Γεύσις — taste
    'weather':   'KAIROS',                 # Καιρός — weather/time/moment
    'life':      'ZOE',                    # Ζωή — life (the quality of being alive)
    'social':    'KOINONIA',               # Κοινωνία — community/communion
}


def format_value(val, name=''):
    """Format a value for display with a visual bar."""
    if isinstance(val, (int, np.integer)):
        return f'{val:>8d}'
    if isinstance(val, (float, np.floating)):
        # Determine a reasonable range for the bar
        if 'db' in name.lower():
            return f'{val:>8.1f} dB'
        if val < 0:
            return f'{val:>8.3f}'
        if val > 10:
            return f'{val:>8.1f}'
        # Show bar for 0-1 range values
        if 0 <= val <= 1.5:
            bar_len = int(val * 20)
            bar = '#' * min(bar_len, 20) + '.' * max(20 - bar_len, 0)
            return f'{val:>6.3f} |{bar}|'
        return f'{val:>8.3f}'
    return str(val)


def print_modality(name, properties):
    """Print a modality section with nice formatting."""
    header = f'  [{MODALITY_ICONS.get(name, name.upper())}]'
    print(f'\n{header}')
    print(f'  {"-" * (len(header) + 20)}')
    for key, val in properties.items():
        print(f'    {key:.<28s} {format_value(val, key)}')


def print_analysis(filepath, results):
    """Print full analysis results to terminal."""
    print()
    print('=' * 70)
    print(f'  FREQUENCY EXPLORER — {filepath}')
    print('=' * 70)

    if 'category' in results:
        print(f'  Category: {results["category"]}')
    if 'shape' in results:
        print(f'  Mel shape: {results["shape"]}  ({results.get("duration_s", "?"):.1f}s)')

    # Cross-modal properties
    modality_order = [
        'acoustic', 'harmonic', 'temporal',
        'light', 'touch', 'emotion', 'chemistry',
        'geometry', 'taste', 'weather', 'life', 'social',
    ]
    for mod in modality_order:
        if mod in results.get('modalities', {}):
            print_modality(mod, results['modalities'][mod])

    # Summary fingerprint
    if 'fingerprint' in results:
        print('\n  [DAKTYLION] — Frequency fingerprint (Greek dimensional axes)')
        print(f'  {"-" * 50}')
        fp = results['fingerprint']
        for key, val in sorted(fp.items()):
            bar_len = int(val * 30)
            bar = '#' * bar_len + '.' * (30 - bar_len)
            print(f'    {key:.<30s} {val:.2f} |{bar}|')

    print()
    print('=' * 70)


# ─── Core analysis function ───

# ─── Greek/Latin translation maps for imported cross-modal functions ───
# These translate English property names from the wildcard scripts to Greek/Latin.
# Where a Greek word unifies what English separates, we use the Greek.
# Latin fills gaps where Greek doesn't have a clean single word.
# See docs/GREEK-DIMENSION-MAP.md for full etymology.

def _translate_keys(props, keymap):
    """Translate English property keys to Greek/Latin using a mapping dict."""
    return {keymap.get(k, k): v for k, v in props.items()}

# Phos (Light) — most map cleanly to Greek
PHOS_MAP = {
    'color': 'chroia',              # χροιά — color AND timbre
    'brightness': 'lamprotes',      # λαμπρότης — brilliance (light + sound + mind)
    'saturation': 'koros',          # κόρος — fullness, satiety
    'texture': 'hyphe',             # ὑφή — weave, fabric
    'flicker': 'palmos',            # παλμός — pulsation (= touch.vibration!)
    'warmth': 'thermos',            # θερμός — warmth (Aristotle's fundamental quality)
    'glow': 'auge',                 # αὐγή — radiance, dawn-light
}

# Haphe (Touch)
HAPHE_MAP = {
    'roughness': 'trachytes',       # τραχύτης — roughness (touch + voice + character)
    'hardness': 'sklerotes',        # σκληρότης — hardness
    'weight': 'baros',              # βάρος — heaviness AND deep pitch AND grief
    'temperature': 'thermos',       # θερμός — same as light.warmth (ONE dimension!)
    'elasticity': 'elastikos',      # ἐλαστικός — already Greek
    'stickiness': 'glischrotes',    # γλισχρότης — viscosity
    'vibration': 'palmos',          # παλμός — SAME as light.flicker (ONE phenomenon!)
}

# Pathos (Emotion)
PATHOS_MAP = {
    'valence': 'hedone',            # ἡδονή — pleasure (or algos for pain)
    'arousal': 'thymos',            # θυμός — spirit, passion, life-force
    'dominance': 'kratos',          # κράτος — power (= social.dominance!)
    'tension': 'tonos',             # τόνος — tension AND tone (physics = psyche)
    'beauty': 'kallos',             # κάλλος — beauty = proportion across ALL domains
}

# Chemeia (Chemistry)
CHEMEIA_MAP = {
    'molecular_weight': 'barytes',  # βαρύτης — heaviness
    'reactivity': 'kinesis',        # κίνησις — motion, change
    'volatility': 'kouphotes',      # κουφότης — lightness (opposite of barytes)
    'bond_strength': 'desmos',      # δεσμός — bond, fetter
    'polarity': 'enantiosis',       # ἐναντίωσις — opposition
    'entropy_state': 'ataxia',      # ἀταξία — disorder
    'catalytic': 'katalysis',       # κατάλυσις — loosening (already Greek!)
}

# Geometria (Geometry)
GEOMETRIA_MAP = {
    'size': 'megethos',             # μέγεθος — magnitude
    'angularity': 'gonia',          # γωνία — angle
    'symmetry': 'symmetria',        # συμμετρία — already Greek!
    'density': 'pyknotes',          # πυκνότης — thickness
    'motion': 'kinesis',            # κίνησις — same as chemistry.reactivity!
    'roundness': 'strongylotes',    # στρογγυλότης — roundness
    'depth': 'bathos',              # βάθος — depth (spatial + intellectual + emotional)
}

# Geusis (Taste)
GEUSIS_MAP = {
    'sweet': 'glykys',              # γλυκύς — sweet (taste AND sound AND sleep)
    'sour': 'oxys',                 # ὀξύς — sour AND sharp AND high-pitched (5 modalities!)
    'bitter': 'pikros',             # πικρός — bitter, pungent
    'salty': 'halmyros',            # ἁλμυρός — briny
    'umami': 'hedys',               # ἡδύς — pleasant, savory (closest Greek to umami)
}

# Kairos (Weather)
KAIROS_MAP = {
    'temperature': 'therme',        # θέρμη — heat
    'pressure': 'thlipsis',         # θλῖψις — compression, pressure
    'humidity': 'hygrotes',         # ὑγρότης — moisture
    'wind_speed': 'anemos',         # ἄνεμος — wind
    'storm_intensity': 'cheimon',   # χειμών — storm AND winter
    'cloud_cover': 'nephele',       # νεφέλη — cloud
    'precipitation': 'hyetos',      # ὑετός — rain
}

# Zoe (Life)
ZOE_MAP = {
    'organic': 'physis',            # φύσις — nature, what grows from itself
    'vitality': 'pneuma',           # πνεῦμα — breath/spirit/life-force
    'growth': 'auxesis',            # αὔξησις — increase
    'complexity': 'poikilia',       # ποικιλία — variety, intricacy
    'metabolism': 'metabole',       # μεταβολή — change (already Greek!)
    'homeostasis': 'homoiostasis',  # ὁμοιόστασις — already Greek!
    'reproduction': 'genesis',      # γένεσις — birth, coming-into-being
}

# Koinonia (Social)
KOINONIA_MAP = {
    'dominance': 'kratos',          # κράτος — power (same as emotion!)
    'threat': 'phobos',             # φόβος — fear, terror
    'attractiveness': 'eros',       # ἔρως — desire, attraction
    'trustworthiness': 'pistis',    # πίστις — trust, faith
    'urgency': 'ananke',            # ἀνάγκη — necessity, compulsion
    'intimacy': 'oikeiosis',        # οἰκείωσις — making-one's-own, belonging
    'social_size': 'plethos',       # πλῆθος — multitude
}


def analyze_mel(mel_frames, filepath='<unknown>', category=None, window_sizes=None):
    """Run full cross-modal analysis on a mel spectrogram.

    Args:
        mel_frames: numpy array (T, n_mels)
        filepath: source file path for labeling
        category: optional category name
        window_sizes: optional list of window sizes in seconds for temporal analysis

    Returns:
        dict with all analysis results
    """
    T, n_mels = mel_frames.shape
    frame_rate = 10.0  # ESC-50 standard

    results = {
        'filepath': str(filepath),
        'shape': list(mel_frames.shape),
        'duration_s': T / frame_rate,
        'n_mels': n_mels,
        'timestamp': datetime.now().isoformat(),
        'modalities': {},
    }

    if category:
        results['category'] = category

    # Compute all modalities (acoustic/harmonic/temporal already return Greek keys)
    results['modalities']['acoustic'] = compute_acoustic_features(mel_frames)
    results['modalities']['harmonic'] = compute_harmonic_features(mel_frames)
    results['modalities']['temporal'] = compute_temporal_dynamics(mel_frames, frame_rate)
    # Cross-modal functions return English keys — translate to Greek/Latin
    if _HAS_CROSSMODAL:
        results['modalities']['light'] = _translate_keys(compute_visual_properties(mel_frames), PHOS_MAP)
        results['modalities']['touch'] = _translate_keys(compute_tactile_properties(mel_frames), HAPHE_MAP)
        results['modalities']['emotion'] = _translate_keys(compute_emotion_properties(mel_frames), PATHOS_MAP)
        results['modalities']['chemistry'] = _translate_keys(compute_chemistry(mel_frames), CHEMEIA_MAP)
        results['modalities']['geometry'] = _translate_keys(compute_shape_properties(mel_frames), GEOMETRIA_MAP)
        results['modalities']['taste'] = _translate_keys(compute_taste_properties(mel_frames), GEUSIS_MAP)
        results['modalities']['weather'] = _translate_keys(compute_weather(mel_frames), KAIROS_MAP)
        results['modalities']['life'] = _translate_keys(compute_life_properties(mel_frames), ZOE_MAP)
        results['modalities']['social'] = _translate_keys(compute_social(mel_frames), KOINONIA_MAP)

    # Build normalized fingerprint (select key properties, normalize to 0-1)
    results['fingerprint'] = _build_fingerprint(results['modalities'])

    # Temporal windowed analysis if requested
    if window_sizes:
        results['temporal_windows'] = compute_windowed_properties(
            mel_frames, window_sizes, frame_rate
        )

    return results


def _build_fingerprint(modalities):
    """Extract key properties and normalize to 0-1 for radar chart.

    Uses Greek dimensional names where possible (cross-modal terms that unify
    what English separates), Latin for gaps, English as last resort.
    See docs/GREEK-DIMENSION-MAP.md for full etymology.
    """
    fp = {}

    # Greek/Latin dimensional names → source modality.property
    # Where Greek collapses multiple English dimensions, we pick the primary source
    picks = {
        'lamprotes':    ('light', 'lamprotes'),        # λαμπρότης — brilliance (light + sound + mind)
        'thermos':      ('light', 'thermos'),           # θερμός — warmth (Aristotle's fundamental quality)
        'chroia':       ('light', 'chroia'),             # χροιά — color AND timbre (sight + hearing)
        'trachytes':    ('touch', 'trachytes'),         # τραχύτης — roughness (touch + voice + character)
        'sklerotes':    ('touch', 'sklerotes'),          # σκληρότης — hardness
        'baros':        ('touch', 'baros'),              # βάρος — heaviness AND deep pitch AND grief
        'hedone':       ('emotion', 'hedone'),           # ἡδονή — pleasure/pain
        'thymos':       ('emotion', 'thymos'),           # θυμός — spirit, passion, life-force
        'tonos':        ('emotion', 'tonos'),            # τόνος — tension AND tone (physics = music = psyche)
        'kallos':       ('emotion', 'kallos'),           # κάλλος — beauty = proportion across ALL domains
        'kinesis':      ('chemistry', 'kinesis'),        # κίνησις — motion, change (= geometry.motion too)
        'ataxia':       ('chemistry', 'ataxia'),         # ἀταξία — disorder
        'megethos':     ('geometry', 'megethos'),        # μέγεθος — magnitude
        'strongylotes': ('geometry', 'strongylotes'),    # στρογγυλότης — roundness
        'glykys':       ('taste', 'glykys'),             # γλυκύς — sweet (taste AND sound AND sleep)
        'pikros':       ('taste', 'pikros'),             # πικρός — bitter, pungent
        'cheimon':      ('weather', 'cheimon'),          # χειμών — storm AND winter
        'pneuma':       ('life', 'pneuma'),              # πνεῦμα — breath/spirit/life-force
        'poikilia':     ('life', 'poikilia'),            # ποικιλία — variety, intricacy
        'kratos':       ('social', 'kratos'),            # κράτος — power, sovereign strength
        'phobos':       ('social', 'phobos'),            # φόβος — fear, terror
        'katharotes':   ('acoustic', 'homalotes'),          # καθαρότης — purity (inv. flatness = tonal purity)
        'periodos':     ('temporal', 'periodos'),          # περίοδος — going-around, cycle
    }

    raw = {}
    for fp_name, (mod, prop) in picks.items():
        if mod in modalities and prop in modalities[mod]:
            raw[fp_name] = modalities[mod][prop]

    # Normalize: clip to reasonable range then scale to 0-1
    for key, val in raw.items():
        if key == 'katharotes':
            # Flatness is already 0-1, but invert (high flatness = noise, low = pure tone)
            fp[key] = 1.0 - min(max(val, 0), 1)
        elif key == 'kinesis':
            fp[key] = min(val / 2.0, 1.0)
        elif key == 'pneuma':
            fp[key] = min(val / 2.0, 1.0)
        elif key == 'cheimon':
            fp[key] = min(val / 1.5, 1.0)
        elif key in ('lamprotes',):
            # Log scale for brightness which can be large
            fp[key] = min(np.log1p(val) / 10.0, 1.0)
        else:
            fp[key] = min(max(val, 0), 1.0)

    return fp


# ─── Visualization ───

def plot_analysis(results, save_path=None):
    """Generate multi-panel visualization of frequency analysis.

    Panel 1: Mel spectrogram
    Panel 2: Light properties as color strip
    Panel 3: Emotion circumplex (valence x arousal)
    Panel 4: Temporal evolution of key properties
    Panel 5: Fingerprint radar chart
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.colors as mcolors
    except ImportError:
        print('  [WARN] matplotlib not available — skipping visualization')
        return None

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Frequency Explorer: {Path(results["filepath"]).name}', fontsize=14, fontweight='bold')

    # ─── Panel 1: Mel spectrogram ───
    ax1 = fig.add_subplot(2, 3, 1)
    mel_path = results['filepath']
    if mel_path.endswith('.npy'):
        mel = np.load(mel_path)
    else:
        mel = np.zeros((50, 40))  # placeholder
    ax1.imshow(mel.T, aspect='auto', origin='lower', cmap='magma',
               interpolation='nearest')
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Mel band')
    ax1.set_title('Mel Spectrogram')

    # ─── Panel 2: Light properties as color strip ───
    ax2 = fig.add_subplot(2, 3, 2)
    light = results['modalities'].get('light', {})
    if light and 'temporal_windows' in results:
        # Use windowed light properties to build color strip
        windows = None
        for ws_key in sorted(results['temporal_windows'].keys()):
            windows = results['temporal_windows'][ws_key]
            break
        if windows:
            colors = []
            for w in windows:
                lp = {k.replace('light.', ''): v for k, v in w['properties'].items() if k.startswith('light.')}
                if lp:
                    hue = lp.get('color', 0.5)
                    sat = lp.get('saturation', 0.5)
                    bright = min(lp.get('brightness', 0.5), 1.0)
                    # Map to RGB via HSV
                    rgb = mcolors.hsv_to_rgb([hue, sat, max(bright, 0.1)])
                    colors.append(rgb)
            if colors:
                color_strip = np.array(colors).reshape(1, -1, 3)
                ax2.imshow(np.repeat(color_strip, 20, axis=0), aspect='auto',
                           interpolation='nearest')
                ax2.set_xlabel('Time (windows)')
                ax2.set_yticks([])
                ax2.set_title('Light: Color Evolution')
            else:
                _draw_single_color(ax2, light)
        else:
            _draw_single_color(ax2, light)
    else:
        _draw_single_color(ax2, light)

    # ─── Panel 3: Emotion circumplex ───
    ax3 = fig.add_subplot(2, 3, 3)
    emotion = results['modalities'].get('emotion', {})
    v = emotion.get('valence', 0.5)
    a = emotion.get('arousal', 0.5)
    d = emotion.get('dominance', 0.5)

    # Draw quadrant labels
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax3.text(0.75, 0.85, 'Happy/Excited', ha='center', fontsize=8, color='gray')
    ax3.text(0.25, 0.85, 'Angry/Tense', ha='center', fontsize=8, color='gray')
    ax3.text(0.25, 0.15, 'Sad/Depressed', ha='center', fontsize=8, color='gray')
    ax3.text(0.75, 0.15, 'Calm/Relaxed', ha='center', fontsize=8, color='gray')

    # Plot point with trajectory if windowed
    if 'temporal_windows' in results:
        for ws_key in sorted(results['temporal_windows'].keys()):
            windows = results['temporal_windows'][ws_key]
            vs = [w['properties'].get('emotion.valence', 0.5) for w in windows]
            ars = [w['properties'].get('emotion.arousal', 0.5) for w in windows]
            if len(vs) > 1:
                ax3.plot(vs, ars, 'b-', alpha=0.3, linewidth=1)
                # Color points by time
                times = np.linspace(0, 1, len(vs))
                ax3.scatter(vs, ars, c=times, cmap='viridis', s=20, zorder=5)
            break

    ax3.scatter([v], [a], c='red', s=100, zorder=10, edgecolors='black', linewidth=1.5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Valence')
    ax3.set_ylabel('Arousal')
    ax3.set_title(f'Emotion Circumplex (dom={d:.2f})')

    # ─── Panel 4: Temporal evolution ───
    ax4 = fig.add_subplot(2, 3, 4)
    if 'temporal_windows' in results:
        for ws_key in sorted(results['temporal_windows'].keys()):
            windows = results['temporal_windows'][ws_key]
            times = [w['time'] for w in windows]
            # Pick a few key properties to plot
            traces = {
                'light.brightness': [],
                'emotion.arousal': [],
                'touch.roughness': [],
                'life.vitality': [],
                'chemistry.reactivity': [],
            }
            for w in windows:
                for key in traces:
                    val = w['properties'].get(key, 0)
                    # Normalize for plotting
                    if key == 'light.brightness':
                        val = min(np.log1p(val) / 10.0, 1.0)
                    elif key == 'chemistry.reactivity':
                        val = min(val / 2.0, 1.0)
                    elif key == 'life.vitality':
                        val = min(val / 2.0, 1.0)
                    traces[key].append(val)
            for key, vals in traces.items():
                label = key.split('.')[1]
                ax4.plot(times, vals, label=label, linewidth=1.5, alpha=0.8)
            ax4.legend(fontsize=7, loc='upper right')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Value (normalized)')
            ax4.set_title(f'Temporal Evolution ({ws_key} windows)')
            break
    else:
        ax4.text(0.5, 0.5, 'No temporal windows\n(use --windows)',
                 ha='center', va='center', fontsize=10, color='gray')
        ax4.set_title('Temporal Evolution')

    # ─── Panel 5: Fingerprint radar chart ───
    ax5 = fig.add_subplot(2, 3, 5, polar=True)
    fp = results.get('fingerprint', {})
    if fp:
        labels = list(fp.keys())
        values = [fp[k] for k in labels]
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]

        ax5.plot(angles_closed, values_closed, 'b-', linewidth=1.5)
        ax5.fill(angles_closed, values_closed, 'b', alpha=0.15)
        ax5.set_xticks(angles)
        ax5.set_xticklabels(labels, fontsize=6)
        ax5.set_ylim(0, 1)
        ax5.set_title('Frequency Fingerprint', pad=20)

    # ─── Panel 6: Modality summary bars ───
    ax6 = fig.add_subplot(2, 3, 6)
    _draw_modality_summary(ax6, results['modalities'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Plot saved to {save_path}')
    else:
        # Default save path
        stem = Path(results['filepath']).stem
        save_path = PROJECT_ROOT / 'data' / f'freq_explorer_{stem}.png'
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f'  Plot saved to {save_path}')

    plt.close()
    return str(save_path)


def _draw_single_color(ax, light):
    """Draw a single color swatch from light properties."""
    import matplotlib.colors as mcolors

    hue = light.get('color', 0.5)
    sat = light.get('saturation', 0.5)
    bright = min(light.get('brightness', 0.5), 1.0)
    bright = max(bright, 0.1)
    rgb = mcolors.hsv_to_rgb([hue, sat, bright])
    ax.imshow([[rgb]], aspect='auto')
    ax.set_title(f'Light: hue={hue:.2f} sat={sat:.2f} bright={bright:.2f}')
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_modality_summary(ax, modalities):
    """Draw horizontal bars summarizing each modality's mean activation."""
    import matplotlib.pyplot as plt

    mod_means = {}
    for mod_name, props in modalities.items():
        if mod_name in ('acoustic', 'harmonic', 'temporal'):
            continue  # skip non-cross-modal
        vals = [v for v in props.values() if isinstance(v, (int, float, np.floating, np.integer))]
        if vals:
            # Normalize roughly to 0-1
            mean_val = np.mean([min(max(v, 0), 2) for v in vals]) / 1.0
            mod_means[mod_name] = min(mean_val, 1.0)

    if mod_means:
        names = list(mod_means.keys())
        values = [mod_means[n] for n in names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars = ax.barh(names, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Mean Activation')
        ax.set_title('Modality Summary')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=8)


# ─── Batch comparison ───

def compare_results(all_results):
    """Compare multiple analysis results side by side."""
    if len(all_results) < 2:
        return

    print('\n' + '=' * 70)
    print('  COMPARISON')
    print('=' * 70)

    # Compare fingerprints
    names = [Path(r['filepath']).stem for r in all_results]
    fp_keys = set()
    for r in all_results:
        fp_keys.update(r.get('fingerprint', {}).keys())
    fp_keys = sorted(fp_keys)

    if fp_keys:
        # Header
        header = f'  {"Property":<25s}'
        for name in names:
            header += f' {name[:12]:>12s}'
        print(header)
        print('  ' + '-' * (25 + 13 * len(names)))

        for key in fp_keys:
            row = f'  {key:<25s}'
            for r in all_results:
                val = r.get('fingerprint', {}).get(key, float('nan'))
                row += f' {val:>12.3f}'
            print(row)

    # Find most distinctive differences
    if len(all_results) == 2:
        fp1 = all_results[0].get('fingerprint', {})
        fp2 = all_results[1].get('fingerprint', {})
        diffs = {}
        for key in fp_keys:
            if key in fp1 and key in fp2:
                diffs[key] = abs(fp1[key] - fp2[key])

        if diffs:
            sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
            print(f'\n  Most distinctive differences:')
            for key, diff in sorted_diffs[:5]:
                print(f'    {key:<25s}: delta = {diff:.3f} '
                      f'({names[0]}={fp1.get(key, 0):.3f}, {names[1]}={fp2.get(key, 0):.3f})')


def plot_comparison(all_results, save_path=None):
    """Generate comparison visualization for multiple files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not available')
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, min(n, 4), figsize=(6 * min(n, 4), 6),
                             subplot_kw=dict(polar=True))
    if n == 1:
        axes = [axes]

    for i, (ax, r) in enumerate(zip(axes, all_results[:4])):
        fp = r.get('fingerprint', {})
        if fp:
            labels = list(fp.keys())
            values = [fp[k] for k in labels]
            N = len(labels)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values_closed = values + [values[0]]
            angles_closed = angles + [angles[0]]

            ax.plot(angles_closed, values_closed, linewidth=1.5)
            ax.fill(angles_closed, values_closed, alpha=0.15)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, fontsize=5)
            ax.set_ylim(0, 1)
            name = Path(r['filepath']).stem
            cat = r.get('category', '')
            ax.set_title(f'{name}\n({cat})', fontsize=9, pad=15)

    plt.tight_layout()
    if save_path is None:
        save_path = str(PROJECT_ROOT / 'data' / 'freq_explorer_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Comparison plot saved to {save_path}')
    plt.close()


# ─── Main CLI ───

def main():
    parser = argparse.ArgumentParser(
        description='Frequency Explorer — Multi-modal audio analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy
  python3 scripts/frequency_explorer.py recording.wav
  python3 scripts/frequency_explorer.py data/training/mel/esc50/ --batch --category dog
  python3 scripts/frequency_explorer.py file1.npy file2.npy --compare
  python3 scripts/frequency_explorer.py data/training/mel/esc50/42.npy --plot --windows 0.5,1.0
        """
    )
    parser.add_argument('inputs', nargs='+', help='Audio files (.wav), mel files (.npy), or directory')
    parser.add_argument('--batch', action='store_true', help='Process all .npy files in directory')
    parser.add_argument('--category', type=str, default=None, help='Filter by ESC-50 category (with --batch)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple files side by side')
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON file')
    parser.add_argument('--plot', action='store_true', help='Generate visualization')
    parser.add_argument('--windows', type=str, default=None,
                        help='Temporal window sizes in seconds, comma-separated (e.g., 0.5,1.0,2.0)')
    parser.add_argument('--limit', type=int, default=20, help='Max files to process in batch mode')
    parser.add_argument('--quiet', action='store_true', help='Suppress terminal output')

    args = parser.parse_args()

    # Parse window sizes
    window_sizes = None
    if args.windows:
        window_sizes = [float(w) for w in args.windows.split(',')]

    # Load category map
    categories = load_category_map()

    # Collect files to process
    files_to_process = []

    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            if args.batch:
                mel_files = sorted(p.glob('*.npy'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
                for mf in mel_files:
                    idx = int(mf.stem) if mf.stem.isdigit() else -1
                    cat = categories[idx] if 0 <= idx < len(categories) else None
                    if args.category and cat != args.category:
                        continue
                    files_to_process.append((str(mf), cat))
                    if len(files_to_process) >= args.limit:
                        break
            else:
                print(f'  {p} is a directory — use --batch to process all files')
                sys.exit(1)
        elif p.suffix == '.npy':
            idx = int(p.stem) if p.stem.isdigit() else -1
            cat = categories[idx] if 0 <= idx < len(categories) else None
            files_to_process.append((str(p), cat))
        elif p.suffix in ('.wav', '.mp3', '.flac', '.ogg'):
            files_to_process.append((str(p), None))
        else:
            print(f'  [WARN] Unknown file type: {p}')

    if not files_to_process:
        print('  No files to process.')
        if args.category:
            print(f'  (No files matched category "{args.category}")')
        sys.exit(1)

    print(f'\n  Processing {len(files_to_process)} file(s)...')

    all_results = []

    for filepath, category in files_to_process:
        p = Path(filepath)

        # Load mel spectrogram
        if p.suffix == '.npy':
            mel = load_mel_file(filepath)
        elif p.suffix in ('.wav', '.mp3', '.flac', '.ogg'):
            print(f'  Loading audio: {filepath}')
            mel = load_audio_as_mel(filepath)
        else:
            continue

        if mel.shape[0] < 4:
            print(f'  [SKIP] {p.name}: too short ({mel.shape[0]} frames)')
            continue

        # Run analysis
        results = analyze_mel(mel, filepath, category, window_sizes)
        all_results.append(results)

        # Print to terminal
        if not args.quiet:
            print_analysis(filepath, results)

        # Generate plot for individual files
        if args.plot and not args.batch:
            plot_analysis(results)

    # Comparison mode
    if args.compare and len(all_results) >= 2:
        compare_results(all_results)
        if args.plot:
            plot_comparison(all_results)

    # Batch summary
    if args.batch and len(all_results) > 1:
        print(f'\n  Processed {len(all_results)} files.')
        if args.compare or not args.quiet:
            compare_results(all_results)
        if args.plot:
            plot_comparison(all_results[:8])  # max 8 for readability

    # Save JSON
    if args.json:
        out = all_results if len(all_results) > 1 else all_results[0]
        # Convert numpy types to native Python
        out_clean = json.loads(json.dumps(out, default=_numpy_serializer))
        with open(args.json, 'w') as f:
            json.dump(out_clean, f, indent=2)
        print(f'\n  Results saved to {args.json}')
    elif len(all_results) == 1 and not args.quiet:
        # Auto-save single file analysis
        stem = Path(all_results[0]['filepath']).stem
        auto_json = PROJECT_ROOT / 'data' / f'freq_explorer_{stem}.json'
        out_clean = json.loads(json.dumps(all_results[0], default=_numpy_serializer))
        with open(auto_json, 'w') as f:
            json.dump(out_clean, f, indent=2)
        print(f'  JSON saved to {auto_json}')


def _numpy_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Not serializable: {type(obj)}')


if __name__ == '__main__':
    main()
