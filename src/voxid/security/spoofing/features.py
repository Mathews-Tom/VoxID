from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SpoofingConfig


def extract_mel_spectrogram(
    audio: NDArray[np.floating],
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> NDArray[np.float32]:
    """Extract log-mel spectrogram (80-band default).

    Uses librosa for STFT and mel filterbank construction.

    Args:
        audio: Mono waveform, any float dtype.
        sr: Sample rate in Hz.
        n_mels: Number of mel filter bands.
        n_fft: FFT window size.
        hop_length: Hop between STFT frames.

    Returns:
        Log-mel spectrogram of shape ``(n_mels, T)``.
    """
    import librosa
    audio_f32 = audio.astype(np.float32)
    mel: NDArray[np.float32] = librosa.feature.melspectrogram(
        y=audio_f32,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel: NDArray[np.float32] = librosa.power_to_db(mel, ref=np.max).astype(
        np.float32
    )
    return log_mel


def extract_cqt(
    audio: NDArray[np.floating],
    sr: int,
    n_bins: int = 84,
    hop_length: int = 256,
) -> NDArray[np.float32]:
    """Extract Constant-Q Transform magnitude spectrogram (84-bin default).

    CQT provides logarithmic frequency resolution, making harmonic
    structure irregularities from synthesis more visible.

    Args:
        audio: Mono waveform, any float dtype.
        sr: Sample rate in Hz.
        n_bins: Number of CQT frequency bins.
        hop_length: Hop between frames.

    Returns:
        Log-magnitude CQT of shape ``(n_bins, T)``.
    """
    import librosa
    audio_f32 = audio.astype(np.float32)
    cqt_complex: NDArray[np.complexfloating] = librosa.cqt(
        y=audio_f32,
        sr=sr,
        n_bins=n_bins,
        hop_length=hop_length,
    )
    cqt_mag: NDArray[np.float32] = librosa.amplitude_to_db(
        np.abs(cqt_complex), ref=np.max
    ).astype(np.float32)
    return cqt_mag


def extract_lfcc(
    audio: NDArray[np.floating],
    sr: int,
    n_lfcc: int = 60,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> NDArray[np.float32]:
    """Extract Linear Frequency Cepstral Coefficients (60-dim default).

    LFCCs use a linear (not mel) filterbank, preserving high-frequency
    detail where vocoder and diffusion artifacts are concentrated.
    Computed via: linear-scale power spectrogram → linear filterbank →
    log → DCT.

    Args:
        audio: Mono waveform, any float dtype.
        sr: Sample rate in Hz.
        n_lfcc: Number of cepstral coefficients.
        n_fft: FFT window size.
        hop_length: Hop between frames.

    Returns:
        LFCC matrix of shape ``(n_lfcc, T)``.
    """
    from scipy.fft import dct
    audio_f32 = audio.astype(np.float32)

    # Compute power spectrogram via STFT
    n_freqs = n_fft // 2 + 1
    frames = _frame_signal(audio_f32, n_fft, hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=-1)
    power_spec = np.abs(spectrum) ** 2

    # Linear filterbank (n_filters evenly spaced in linear frequency)
    n_filters = max(n_lfcc * 2, 128)
    filterbank = _linear_filterbank(n_filters, n_freqs, sr, n_fft)

    # Apply filterbank → log → DCT
    filtered = power_spec @ filterbank.T
    log_filtered = np.log(filtered + 1e-9)
    lfcc: NDArray[np.float32] = dct(
        log_filtered, type=2, n=n_lfcc, axis=-1, norm="ortho"
    ).astype(np.float32)

    # Return shape (n_lfcc, T) to match mel/CQT convention
    return lfcc.T


def _frame_signal(
    signal: NDArray[np.float32],
    frame_length: int,
    hop_length: int,
) -> NDArray[np.float32]:
    """Split signal into overlapping frames.

    Returns:
        Array of shape ``(n_frames, frame_length)``.
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    if n_frames <= 0:
        # Signal shorter than one frame — zero-pad
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded.reshape(1, frame_length)

    indices = np.arange(frame_length)[None, :] + (
        hop_length * np.arange(n_frames)[:, None]
    )
    result: NDArray[np.float32] = signal[indices]
    return result


def _linear_filterbank(
    n_filters: int,
    n_freqs: int,
    sr: int,
    n_fft: int,
) -> NDArray[np.float32]:
    """Build a linearly-spaced triangular filterbank.

    Filters are evenly distributed across [0, sr/2] in linear Hz,
    unlike mel filterbanks which compress higher frequencies.

    Returns:
        Filterbank matrix of shape ``(n_filters, n_freqs)``.
    """
    low_freq = 0.0
    high_freq = sr / 2.0
    points = np.linspace(low_freq, high_freq, n_filters + 2)
    bin_points = np.floor((n_fft + 1) * points / sr).astype(int)

    filterbank = np.zeros((n_filters, n_freqs), dtype=np.float32)
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        if center > left:
            filterbank[i, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        # Falling slope
        if right > center:
            filterbank[i, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filterbank


def resample_if_needed(
    audio: NDArray[np.floating],
    sr: int,
    config: SpoofingConfig,
) -> tuple[NDArray[np.float32], int]:
    """Resample audio to the config's expected sample rate if necessary.

    Returns:
        Tuple of (resampled_audio, target_sr).
    """
    target_sr = config.sample_rate
    audio_f32 = audio.astype(np.float32)
    if sr == target_sr:
        return audio_f32, target_sr

    import librosa
    resampled: NDArray[np.float32] = librosa.resample(
        audio_f32, orig_sr=sr, target_sr=target_sr
    ).astype(np.float32)
    return resampled, target_sr
