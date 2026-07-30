"""Shared fixtures for ML server unit tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf


def make_wav_bytes(
    seconds: float = 0.5,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Synthesize a short mono PCM WAV in memory."""
    n = max(1, int(seconds * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave = 0.2 * np.sin(2 * np.pi * frequency * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()
