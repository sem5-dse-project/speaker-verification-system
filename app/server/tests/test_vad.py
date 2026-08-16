"""Unit tests for Silero VAD speech extraction behavior."""

from __future__ import annotations

import pytest
import torch

from ml_server.config import SAMPLE_RATE
import ml_server.vad as vad


def test_vad_accepts_normal_speech(monkeypatch: pytest.MonkeyPatch):
    wave = torch.randn(int(SAMPLE_RATE * 1.2)) * 0.1

    monkeypatch.setattr(vad, "VAD_ENABLED", True)
    monkeypatch.setattr(vad, "_get_vad_model", lambda: object())
    monkeypatch.setattr(
        vad,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 2000, "end": 12000}],
    )

    result = vad.extract_speech_segments(wave)
    assert result.has_speech is True
    assert result.num_speech_segments == 1
    assert result.speech_waveform.numel() == 10000
    assert result.speech_ms > 0


def test_vad_rejects_silence(monkeypatch: pytest.MonkeyPatch):
    wave = torch.zeros(int(SAMPLE_RATE * 1.0))

    monkeypatch.setattr(vad, "VAD_ENABLED", True)
    monkeypatch.setattr(vad, "_get_vad_model", lambda: object())
    monkeypatch.setattr(vad, "get_speech_timestamps", lambda *args, **kwargs: [])

    result = vad.extract_speech_segments(wave)
    assert result.has_speech is False
    assert result.speech_ms == 0.0
    assert result.num_speech_segments == 0


def test_vad_handles_speech_with_pauses(monkeypatch: pytest.MonkeyPatch):
    wave = torch.randn(int(SAMPLE_RATE * 2.0)) * 0.08

    monkeypatch.setattr(vad, "VAD_ENABLED", True)
    monkeypatch.setattr(vad, "_get_vad_model", lambda: object())
    monkeypatch.setattr(
        vad,
        "get_speech_timestamps",
        lambda *args, **kwargs: [
            {"start": 1000, "end": 5000},
            {"start": 9000, "end": 14000},
        ],
    )

    result = vad.extract_speech_segments(wave)
    assert result.has_speech is True
    assert result.num_speech_segments == 2
    assert result.speech_waveform.numel() == 9000


def test_vad_rejects_noisy_short_bursts(monkeypatch: pytest.MonkeyPatch):
    wave = torch.randn(int(SAMPLE_RATE * 2.0)) * 0.02

    monkeypatch.setattr(vad, "VAD_ENABLED", True)
    monkeypatch.setattr(vad, "VAD_MIN_TOTAL_SPEECH_MS", 300.0)
    monkeypatch.setattr(vad, "_get_vad_model", lambda: object())
    monkeypatch.setattr(
        vad,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 1000, "end": 2500}],
    )

    result = vad.extract_speech_segments(wave)
    assert result.has_speech is False
    assert result.speech_ms < 300.0


def test_vad_rejects_audio_that_is_too_short(monkeypatch: pytest.MonkeyPatch):
    wave = torch.randn(int(SAMPLE_RATE * 0.1)) * 0.1

    monkeypatch.setattr(vad, "VAD_MIN_AUDIO_MS", 250.0)
    result = vad.extract_speech_segments(wave)
    assert result.has_speech is False
    assert result.total_ms < 250.0
