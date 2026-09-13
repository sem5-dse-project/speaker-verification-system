"""Noise augmentation helpers (placeholders — no dataset downloads)."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import ensure_mono
from voice_auth.common.types import Waveform


def snr_mix(clean: Waveform, noise: Waveform, snr_db: float) -> Waveform:
    """
    Mix clean speech with noise at a target SNR (dB).

    Args:
        clean: Clean waveform ``[1, N]``.
        noise: Noise waveform ``[1, M]`` (will be cropped/tiled to N).
        snr_db: Target signal-to-noise ratio in dB.

    Returns:
        Noisy waveform ``[1, N]``, float32.
    """
    speech = ensure_mono(clean).astype(np.float32).squeeze(0)
    noise_w = ensure_mono(noise).astype(np.float32).squeeze(0)

    n = speech.shape[0]
    if noise_w.shape[0] < n:
        reps = int(np.ceil(n / max(noise_w.shape[0], 1)))
        noise_w = np.tile(noise_w, reps)[:n]
    else:
        noise_w = noise_w[:n]

    speech_power = float(np.mean(speech**2) + 1e-10)
    noise_power = float(np.mean(noise_w**2) + 1e-10)
    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = (speech + scale * noise_w).astype(np.float32)
    return mixed[np.newaxis, :]


def add_gaussian_noise(
    waveform: Waveform, snr_db: float, rng: np.random.Generator | None = None
) -> Waveform:
    """Add white Gaussian noise at a target SNR."""
    rng = rng or np.random.default_rng()
    speech = ensure_mono(waveform).astype(np.float32)
    noise = rng.standard_normal(speech.shape).astype(np.float32)
    return snr_mix(speech, noise, snr_db)
