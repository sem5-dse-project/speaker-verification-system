"""Log-Mel spectrogram feature extraction (NumPy placeholder)."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import ensure_mono
from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Waveform


def extract_logmel(
    waveform: Waveform,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    duration_sec: float = 3.0,
) -> np.ndarray:
    """
    Extract a fixed-size Log-Mel feature map.

    This is a lightweight STFT + mel-filterbank approximation for scaffolding.
    Member 2 should replace it with ``torchaudio`` / ``librosa`` features.

    Args:
        waveform: Mono audio ``[1, num_samples]``.
        sample_rate: Sample rate in Hz.
        n_mels: Number of Mel bins.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        duration_sec: Target duration (pad/crop).

    Returns:
        Array of shape ``[1, n_mels, time]`` (channel-first for CNN input).
    """
    wave = ensure_mono(waveform).astype(np.float32).squeeze(0)
    target_len = int(duration_sec * sample_rate)
    if wave.shape[0] < target_len:
        wave = np.pad(wave, (0, target_len - wave.shape[0]))
    else:
        wave = wave[:target_len]

    # Simple magnitude spectrogram
    if wave.shape[0] < n_fft:
        wave = np.pad(wave, (0, n_fft - wave.shape[0]))

    frames = []
    for start in range(0, wave.shape[0] - n_fft + 1, hop_length):
        frame = wave[start : start + n_fft] * np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(frame))
        frames.append(spec)

    if not frames:
        frames = [np.abs(np.fft.rfft(wave[:n_fft] * np.hanning(n_fft).astype(np.float32)))]

    spectrogram = np.stack(frames, axis=1)  # [freq, time]
    # Crude mel projection via linear bins
    freq_bins = spectrogram.shape[0]
    mel = np.zeros((n_mels, spectrogram.shape[1]), dtype=np.float32)
    edges = np.linspace(0, freq_bins, n_mels + 1).astype(int)
    for i in range(n_mels):
        start, end = edges[i], max(edges[i + 1], edges[i] + 1)
        mel[i] = spectrogram[start:end].mean(axis=0)

    logmel = np.log(mel + 1e-6).astype(np.float32)
    return logmel[np.newaxis, ...]
