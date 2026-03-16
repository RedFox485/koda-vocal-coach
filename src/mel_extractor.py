"""Mel Spectrogram Extractor — Primary input for EARS Mamba-2 encoder.

Replaces Essentia as the primary feature extraction pathway.
Produces 80 log-mel bands per frame at 100Hz (10ms hop, 20ms window).

The old Essentia extractor (extractor.py) remains available for:
- Tier 1 physics prior sanity checks (roughness, harmonics)
- A/B comparison during validation
- Legacy compatibility with existing recordings

Research backing: R2 (r2-optimal-mamba-input-features.md)
- Every published Audio Mamba paper uses log-mel spectrograms
- 80 bands is the standard for AudioSet-scale tasks
- Per-band normalization after log compression is critical
"""
import numpy as np
import librosa


class MelExtractor:
    """Extracts log-mel spectrograms from audio frames.

    Designed for streaming: processes one hop at a time, maintains
    state for overlapping windows.

    Parameters match Audio Mamba conventions:
    - 80 mel bands
    - 20ms window (882 samples at 44.1kHz)
    - 10ms hop (441 samples at 44.1kHz) = 100Hz frame rate
    - Log compression: log(1 + C * mel_energy)
    """

    def __init__(self, sample_rate=44100, n_mels=80, win_ms=20, hop_ms=10,
                 log_offset=1e-6, log_scale=1.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = int(sample_rate * win_ms / 1000)  # 882 samples
        self.hop_length = int(sample_rate * hop_ms / 1000)  # 441 samples
        self.n_fft = 2 ** int(np.ceil(np.log2(self.win_length)))  # 1024
        self.log_offset = log_offset
        self.log_scale = log_scale

        # Pre-compute mel filterbank (reused every frame)
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=self.n_fft, n_mels=n_mels,
            fmin=20.0, fmax=sample_rate / 2
        )

        # Running normalization stats (online Welford's algorithm)
        self._count = 0
        self._mean = np.zeros(n_mels, dtype=np.float64)
        self._m2 = np.zeros(n_mels, dtype=np.float64)
        self._std = np.ones(n_mels, dtype=np.float64)  # avoid div by zero

        # State for streaming: buffer for overlapping windows
        self._buffer = np.zeros(0, dtype=np.float32)

        # Frame counter
        self.frame_count = 0

    def extract_frame(self, audio_frame):
        """Extract mel features from a single window of audio.

        Args:
            audio_frame: numpy array of shape (win_length,) or larger.
                         If larger than win_length, uses the last win_length samples.

        Returns:
            mel_features: numpy array of shape (n_mels,) — log-mel band energies
        """
        if len(audio_frame) < self.win_length:
            # Pad short frames with zeros
            padded = np.zeros(self.win_length, dtype=np.float32)
            padded[-len(audio_frame):] = audio_frame
            audio_frame = padded
        elif len(audio_frame) > self.win_length:
            audio_frame = audio_frame[-self.win_length:]

        # Window
        windowed = audio_frame * np.hanning(len(audio_frame))

        # FFT
        fft = np.fft.rfft(windowed, n=self.n_fft)
        power_spectrum = np.abs(fft) ** 2

        # Mel filterbank
        mel_energy = self.mel_basis @ power_spectrum

        # Log compression
        log_mel = np.log(self.log_offset + self.log_scale * mel_energy)

        self.frame_count += 1
        return log_mel.astype(np.float32)

    def extract_from_audio(self, audio, normalize=False):
        """Extract mel spectrograms from a full audio array.

        Args:
            audio: numpy array of audio samples (mono, float32)
            normalize: if True, apply per-band standardization

        Returns:
            mel_spectrogram: numpy array of shape (n_frames, n_mels)
        """
        # Use librosa for batch extraction (faster than frame-by-frame)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            n_mels=self.n_mels, fmin=20.0, fmax=self.sample_rate / 2
        )

        # Log compression
        log_mel = np.log(self.log_offset + self.log_scale * mel)

        # Transpose to (n_frames, n_mels)
        log_mel = log_mel.T.astype(np.float32)

        if normalize:
            self._fit_normalization(log_mel)
            log_mel = self._normalize(log_mel)

        self.frame_count += len(log_mel)
        return log_mel

    def process_chunk(self, audio_chunk):
        """Process a streaming audio chunk, yielding mel frames.

        Handles buffering for overlapping windows. Feed audio in any
        chunk size; get back however many complete frames fit.

        Args:
            audio_chunk: numpy array of new audio samples

        Returns:
            frames: numpy array of shape (n_new_frames, n_mels)
        """
        # Append to buffer
        self._buffer = np.concatenate([self._buffer, audio_chunk])

        frames = []
        while len(self._buffer) >= self.win_length:
            frame = self.extract_frame(self._buffer[:self.win_length])
            frames.append(frame)
            # Advance by hop_length
            self._buffer = self._buffer[self.hop_length:]

        if frames:
            return np.array(frames, dtype=np.float32)
        return np.zeros((0, self.n_mels), dtype=np.float32)

    def _fit_normalization(self, mel_spectrogram):
        """Compute per-band mean and std from a mel spectrogram."""
        self._mean = mel_spectrogram.mean(axis=0)
        self._std = mel_spectrogram.std(axis=0)
        self._std[self._std < 1e-8] = 1e-8  # prevent division by zero
        self._count = len(mel_spectrogram)

    def update_normalization(self, mel_frame):
        """Online update of normalization stats (Welford's algorithm)."""
        self._count += 1
        delta = mel_frame - self._mean
        self._mean += delta / self._count
        delta2 = mel_frame - self._mean
        self._m2 += delta * delta2
        if self._count > 1:
            self._std = np.sqrt(self._m2 / (self._count - 1))
            self._std[self._std < 1e-8] = 1e-8

    def _normalize(self, mel_spectrogram):
        """Apply per-band standardization."""
        return (mel_spectrogram - self._mean) / self._std

    def normalize_frame(self, mel_frame):
        """Normalize a single frame using current stats."""
        return (mel_frame - self._mean) / self._std

    def get_config(self):
        """Return extractor configuration as dict."""
        return {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'n_fft': self.n_fft,
            'frame_rate_hz': self.sample_rate / self.hop_length,
            'win_ms': self.win_length / self.sample_rate * 1000,
            'hop_ms': self.hop_length / self.sample_rate * 1000,
        }


class PhysicsPriors:
    """Tier 1 physics-based prior computations.

    Lightweight deterministic sanity checks that run alongside mel extraction.
    These are NOT input to the Mamba-2 encoder — they feed the meta-prior
    system for precision weighting and validation.

    Each prior outputs (value, confidence).
    """

    def __init__(self, sample_rate=44100, frame_size=882):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._prev_spectrum = None

    def compute(self, audio_frame):
        """Compute all Tier 1 physics priors from raw audio.

        Returns dict of {prior_name: (value, confidence)}.
        """
        priors = {}

        # Ensure float32
        audio = np.asarray(audio_frame, dtype=np.float32)
        if len(audio) < 64:
            return self._empty_priors()

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        # FFT for spectral analysis
        windowed = audio * np.hanning(len(audio))
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

        # 1. Roughness: amplitude modulation in 30-150Hz range
        # Detected via envelope modulation spectrum
        envelope = np.abs(audio)  # simple rectification
        if len(envelope) >= 64:
            env_fft = np.abs(np.fft.rfft(envelope))
            env_freqs = np.fft.rfftfreq(len(envelope), 1.0 / self.sample_rate)
            roughness_mask = (env_freqs >= 30) & (env_freqs <= 150)
            if roughness_mask.any():
                roughness_energy = np.sum(env_fft[roughness_mask] ** 2)
                total_env_energy = np.sum(env_fft ** 2) + 1e-10
                roughness = float(np.clip(roughness_energy / total_env_energy, 0, 1))
            else:
                roughness = 0.0
        else:
            roughness = 0.0
        priors['roughness'] = (roughness, 0.8 if rms_db > -50 else 0.3)

        # 2. Onset sharpness: energy derivative
        onset_sharpness = 0.0
        if self._prev_spectrum is not None and len(magnitude) == len(self._prev_spectrum):
            spectral_diff = np.sum(np.maximum(magnitude - self._prev_spectrum, 0))
            onset_sharpness = float(np.clip(spectral_diff / (np.sum(magnitude) + 1e-10), 0, 1))
        priors['onset_sharpness'] = (onset_sharpness, 0.9 if rms_db > -55 else 0.3)
        self._prev_spectrum = magnitude.copy()

        # 3. Spectral regularity: how harmonic is the spectrum
        # High regularity = tonal (harmonic), low = noisy
        if len(magnitude) > 4:
            reg = np.mean(np.abs(np.diff(magnitude[1:]))) / (np.mean(magnitude[1:]) + 1e-10)
            spectral_regularity = float(np.clip(1.0 - reg, 0, 1))
        else:
            spectral_regularity = 0.0
        priors['spectral_regularity'] = (spectral_regularity, 0.7)

        # 4. Harmonic series detection: autocorrelation peaks
        if len(audio) >= 128 and rms_db > -55:
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            # Look for peaks after the first zero crossing
            min_lag = int(self.sample_rate / 1400)  # max freq ~1400Hz
            max_lag = int(self.sample_rate / 50)    # min freq ~50Hz
            max_lag = min(max_lag, len(autocorr) - 1)
            if max_lag > min_lag:
                search = autocorr[min_lag:max_lag]
                if len(search) > 0:
                    peak_val = float(np.max(search))
                    harmonicity = np.clip(peak_val, 0, 1)
                else:
                    harmonicity = 0.0
            else:
                harmonicity = 0.0
        else:
            harmonicity = 0.0
        priors['harmonicity'] = (harmonicity, 0.8 if rms_db > -50 else 0.2)

        # 5. Spectral flux (rate of spectral change)
        priors['spectral_flux'] = (onset_sharpness, 0.9)  # reuse onset calc

        # 6. Energy level
        priors['energy_db'] = (float(rms_db), 0.95)

        return priors

    def _empty_priors(self):
        return {
            'roughness': (0.0, 0.0),
            'onset_sharpness': (0.0, 0.0),
            'spectral_regularity': (0.0, 0.0),
            'harmonicity': (0.0, 0.0),
            'spectral_flux': (0.0, 0.0),
            'energy_db': (-100.0, 0.0),
        }
