"""Five-dimensional audio-quality feature extraction."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import ensure_mono
from voice_auth.common.types import DEFAULT_QUALITY_DIM, DEFAULT_SAMPLE_RATE, QualityVector, Waveform


class BasicQualityEstimator:
    """
    Extract a configurable quality vector (default 5-D).

    Features (placeholder set for scaffolding):
        0. RMS energy
        1. Zero-crossing rate
        2. Spectral centroid (normalized)
        3. Spectral flatness
        4. Crest factor
    """

    def __init__(self, quality_dim: int = DEFAULT_QUALITY_DIM) -> None:
        if quality_dim != DEFAULT_QUALITY_DIM:
            # Keep API flexible; extra dims are zero-padded / truncated.
            pass
        self.quality_dim = quality_dim

    def extract(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> QualityVector:
        """Return a quality vector of shape ``[quality_dim]``."""
        wave = ensure_mono(waveform).astype(np.float32).squeeze(0)
        if wave.size == 0:
            return np.zeros(self.quality_dim, dtype=np.float32)

        rms = float(np.sqrt(np.mean(wave**2) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.sign(wave)))) / 2.0)

        spectrum = np.abs(np.fft.rfft(wave)) + 1e-12
        freqs = np.fft.rfftfreq(wave.shape[0], d=1.0 / max(sample_rate, 1))
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
        centroid_n = centroid / max(sample_rate / 2.0, 1.0)

        geo = float(np.exp(np.mean(np.log(spectrum))))
        arith = float(np.mean(spectrum))
        flatness = geo / arith

        peak = float(np.max(np.abs(wave)) + 1e-12)
        crest = peak / (rms + 1e-12)

        feats = np.asarray([rms, zcr, centroid_n, flatness, crest], dtype=np.float32)
        if self.quality_dim == feats.shape[0]:
            return feats
        if self.quality_dim < feats.shape[0]:
            return feats[: self.quality_dim]
        out = np.zeros(self.quality_dim, dtype=np.float32)
        out[: feats.shape[0]] = feats
        return out
