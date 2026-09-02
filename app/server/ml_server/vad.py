"""Silero VAD utilities for extracting speech-only waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ml_server.config import (
    SAMPLE_RATE,
    VAD_ENABLED,
    VAD_MIN_AUDIO_MS,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_TOTAL_SPEECH_MS,
    VAD_SPEECH_PAD_MS,
    VAD_SPEECH_THRESHOLD,
    VAD_USE_ONNX,
)

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except Exception:  # pragma: no cover - import availability differs by environment
    get_speech_timestamps = None
    load_silero_vad = None


@dataclass
class VadResult:
    has_speech: bool
    speech_waveform: torch.Tensor
    total_ms: float
    speech_ms: float
    num_speech_segments: int
    rms: float


_vad_model = None
_vad_model_onnx: bool | None = None


def _waveform_rms(waveform: torch.Tensor) -> float:
    wave = waveform.detach().float().reshape(-1)
    if wave.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(wave * wave)).item())


def _get_vad_model():
    global _vad_model, _vad_model_onnx

    if not VAD_ENABLED:
        return None

    if load_silero_vad is None or get_speech_timestamps is None:
        raise RuntimeError("Silero VAD dependencies are not installed")

    if _vad_model is None or _vad_model_onnx != VAD_USE_ONNX:
        _vad_model = load_silero_vad(onnx=VAD_USE_ONNX)
        _vad_model_onnx = VAD_USE_ONNX

    return _vad_model


def extract_speech_segments(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
) -> VadResult:
    """Return speech-only waveform using Silero VAD."""
    wave = waveform.detach().float().reshape(-1)
    total_ms = (wave.numel() / sample_rate) * 1000.0 if sample_rate else 0.0
    rms = _waveform_rms(wave)

    if wave.numel() == 0 or total_ms < VAD_MIN_AUDIO_MS:
        return VadResult(
            has_speech=False,
            speech_waveform=wave,
            total_ms=total_ms,
            speech_ms=0.0,
            num_speech_segments=0,
            rms=rms,
        )

    if not VAD_ENABLED:
        return VadResult(
            has_speech=True,
            speech_waveform=wave,
            total_ms=total_ms,
            speech_ms=total_ms,
            num_speech_segments=1,
            rms=rms,
        )

    model = _get_vad_model()
    timestamps = get_speech_timestamps(
        wave,
        model,
        sampling_rate=sample_rate,
        threshold=VAD_SPEECH_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    if not timestamps:
        return VadResult(
            has_speech=False,
            speech_waveform=wave,
            total_ms=total_ms,
            speech_ms=0.0,
            num_speech_segments=0,
            rms=rms,
        )

    chunks: list[torch.Tensor] = []
    speech_samples = 0
    for segment in timestamps:
        start = int(segment["start"])
        end = int(segment["end"])
        if end > start:
            chunk = wave[start:end]
            chunks.append(chunk)
            speech_samples += chunk.numel()

    speech_ms = (speech_samples / sample_rate) * 1000.0
    if speech_samples == 0 or speech_ms < VAD_MIN_TOTAL_SPEECH_MS:
        return VadResult(
            has_speech=False,
            speech_waveform=wave,
            total_ms=total_ms,
            speech_ms=speech_ms,
            num_speech_segments=len(timestamps),
            rms=rms,
        )

    speech_wave = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    return VadResult(
        has_speech=True,
        speech_waveform=speech_wave,
        total_ms=total_ms,
        speech_ms=speech_ms,
        num_speech_segments=len(timestamps),
        rms=rms,
    )
