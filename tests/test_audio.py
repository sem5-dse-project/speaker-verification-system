"""Tests for audio preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from voice_auth.common.audio import (
    ensure_mono,
    l2_normalize,
    preprocess_audio,
    resample_placeholder,
    to_float32,
)


def test_ensure_mono_from_1d() -> None:
    wave = np.random.randn(1600).astype(np.float32)
    mono = ensure_mono(wave)
    assert mono.shape == (1, 1600)
    assert mono.dtype == np.float32


def test_ensure_mono_averages_channels() -> None:
    wave = np.ones((2, 100), dtype=np.float32)
    wave[1] = 3.0
    mono = ensure_mono(wave)
    assert mono.shape == (1, 100)
    assert np.allclose(mono, 2.0)


def test_to_float32_integer() -> None:
    wave = np.array([[0, 16384, -16384]], dtype=np.int16)
    out = to_float32(wave)
    assert out.dtype == np.float32
    assert out.shape[0] == 1


def test_resample_changes_length() -> None:
    wave = np.random.randn(1, 8000).astype(np.float32)
    out = resample_placeholder(wave, orig_sr=8000, target_sr=16000)
    assert out.shape[0] == 1
    assert out.shape[1] == 16000


def test_preprocess_audio_pipeline() -> None:
    wave = np.random.randn(2, 8000).astype(np.float32)
    processed, sr = preprocess_audio(wave, sample_rate=8000, target_sr=16000)
    assert sr == 16000
    assert processed.shape[0] == 1
    assert processed.dtype == np.float32


def test_l2_normalize_unit_norm() -> None:
    vec = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(vec)
    assert pytest.approx(1.0, rel=1e-5) == float(np.linalg.norm(out))
