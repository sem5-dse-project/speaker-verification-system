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
