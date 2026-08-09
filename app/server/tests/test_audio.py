"""Unit tests for WAV decoding helpers."""

from __future__ import annotations

import pytest

from ml_server.audio import load_audio_bytes
from ml_server.config import SAMPLE_RATE
from tests.conftest import make_wav_bytes


def test_load_audio_bytes_reads_wav(wav_bytes: bytes):
    wave = load_audio_bytes(wav_bytes)
    assert wave.ndim == 1
    assert wave.numel() > 0
    # Synthetic clip is short; should be near requested length at model SR
    assert wave.numel() <= int(SAMPLE_RATE * 4)


def test_load_audio_bytes_rejects_garbage():
    with pytest.raises(ValueError, match="Could not decode audio"):
        load_audio_bytes(b"not-a-real-audio-file")


def test_load_audio_bytes_resamples_if_needed():
    wav_8k = make_wav_bytes(seconds=0.4, sample_rate=8000)
    wave = load_audio_bytes(wav_8k, sample_rate=16000)
    # 0.4s at 8k -> resampled to ~0.4s at 16k
    assert wave.numel() == pytest.approx(6400, abs=50)


def test_has_sufficient_speech_rejects_silence():
    import torch

    from ml_server.audio import has_sufficient_speech

    ok, rms = has_sufficient_speech(torch.zeros(16000), min_rms=0.01)
    assert ok is False
    assert rms == pytest.approx(0.0, abs=1e-6)


def test_has_sufficient_speech_rejects_steady_ambient():
    """Constant low noise can have RMS > 0.01 but is not speech."""
    import torch

    from ml_server.audio import has_sufficient_speech

    ambient = torch.randn(16000 * 2) * 0.02
    ok, rms = has_sufficient_speech(ambient)
    assert rms > 0.01
    assert ok is False


def test_has_sufficient_speech_rejects_moderate_ambient_bursts():
    """Laptop empty-mic style: some mid frames, no strong speech peaks."""
    import torch

    from ml_server.audio import has_sufficient_speech

    wave = torch.randn(16000 * 2) * 0.01
    wave[2000:6000] = torch.randn(4000) * 0.05  # mid bursts only
    ok, _ = has_sufficient_speech(wave)
    assert ok is False


def test_has_sufficient_speech_accepts_speech_like_bursts():
    import torch

    from ml_server.audio import has_sufficient_speech

    wave = torch.randn(16000 * 2) * 0.004
    wave[8000:17600] = torch.randn(9600) * 0.15
    ok, rms = has_sufficient_speech(wave)
    assert ok is True
    assert rms > 0.01


def test_has_sufficient_speech_accepts_loud_signal():
    import torch

    from ml_server.audio import has_sufficient_speech

    ok, rms = has_sufficient_speech(torch.ones(16000) * 0.2, min_rms=0.01)
    assert ok is True
    assert rms > 0.01

