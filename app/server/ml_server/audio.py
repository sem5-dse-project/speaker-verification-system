"""Audio loading helpers for ECAPA inference."""

from __future__ import annotations

import io

import torch
import torchaudio
import soundfile as sf

from ml_server.config import MAX_SECONDS, SAMPLE_RATE


def load_audio_bytes(data: bytes, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Decode bytes to mono float32 waveform at sample_rate."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio).mean(dim=1)
    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)
    if wave.numel() == 0:
        raise ValueError("Empty audio")
    max_len = int(sample_rate * MAX_SECONDS)
    if wave.numel() > max_len:
        start = max(0, (wave.numel() - max_len) // 2)
        wave = wave[start : start + max_len]
    return wave
