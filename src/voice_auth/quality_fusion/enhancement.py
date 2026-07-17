"""Frozen speech-enhancement placeholders."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import ensure_mono
from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Waveform


class IdentityEnhancer:
    """Pass-through enhancer for scaffolding and tests."""

    def enhance(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Waveform:
        """Return a copy of the input waveform."""
        _ = sample_rate
        return ensure_mono(waveform).astype(np.float32).copy()


class FrozenSpeechEnhancer:
    """
    Placeholder for a frozen speech-enhancement model (e.g. spectral mask / DNN).

    Does not download weights. Until a checkpoint is provided, falls back to a
    simple high-shelf spectral attenuation as a stand-in.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path
        # Member 3: load frozen enhancer weights here when available.

    def enhance(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Waveform:
        """
        Apply a lightweight placeholder enhancement.

        Real implementation should run a frozen model and return float32 mono audio.
        """
        wave = ensure_mono(waveform).astype(np.float32).squeeze(0)
        # Mild spectral tilt as a deterministic stand-in (not a real enhancer)
        if wave.size < 16:
            return wave[np.newaxis, :]
        spectrum = np.fft.rfft(wave)
        freqs = np.fft.rfftfreq(wave.shape[0], d=1.0 / sample_rate)
        # Attenuate high frequencies slightly
        mask = 1.0 / (1.0 + (freqs / 4000.0) ** 2)
        enhanced = np.fft.irfft(spectrum * mask, n=wave.shape[0]).astype(np.float32)
        peak = np.max(np.abs(enhanced)) + 1e-8
        enhanced = enhanced / peak * min(1.0, float(np.max(np.abs(wave)) + 1e-8))
        return enhanced[np.newaxis, :]
