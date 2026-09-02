"""Audio loading helpers for ECAPA inference."""

from __future__ import annotations

import io

import torch
import torchaudio
import soundfile as sf

from ml_server.config import (
    MAX_SECONDS,
    MIN_SPEECH_FRAME_RMS,
    MIN_SPEECH_MS,
    MIN_SPEECH_PEAK,
    MIN_SPEECH_RMS,
    MIN_SPEECH_STRONG_FRAME_RMS,
    MIN_SPEECH_STRONG_MS,
    SAMPLE_RATE,
)
from ml_server.vad import VadResult, extract_speech_segments


def load_audio_bytes(data: bytes, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Decode bytes to mono float32 waveform at sample_rate.

    Prefers soundfile (WAV/FLAC). Falls back to torchaudio when available.
    """
    wave, sr = _decode_waveform(data)
    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)
    if wave.numel() == 0:
        raise ValueError("Empty audio")
    max_len = int(sample_rate * MAX_SECONDS)
    if wave.numel() > max_len:
        start = max(0, (wave.numel() - max_len) // 2)
        wave = wave[start : start + max_len]
    return wave


def extract_speech_audio(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
) -> VadResult:
    """Run Silero VAD and return speech-only waveform plus speech metadata."""
    result = extract_speech_segments(waveform, sample_rate=sample_rate)

    # Keep max clip length consistent after silence removal.
    max_len = int(sample_rate * MAX_SECONDS)
    if result.has_speech and result.speech_waveform.numel() > max_len:
        speech_wave = result.speech_waveform[:max_len]
        speech_ms = (speech_wave.numel() / sample_rate) * 1000.0
        return VadResult(
            has_speech=True,
            speech_waveform=speech_wave,
            total_ms=result.total_ms,
            speech_ms=speech_ms,
            num_speech_segments=result.num_speech_segments,
            rms=result.rms,
        )

    return result


def waveform_rms(waveform: torch.Tensor) -> float:
    """Root-mean-square level of a mono waveform."""
    wave = waveform.detach().float().reshape(-1)
    if wave.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(wave * wave)).item())


def has_sufficient_speech(
    waveform: torch.Tensor,
    min_rms: float | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[bool, float]:
    """Return (ok, rms). Requires sustained louder frames, not ambient hiss.

    Ambient / "empty mic" on this laptop often has RMS 0.01–0.04 and clears a
    weak gate, then ECAPA can still ACCEPT (~0.5–0.7 cosine). We require either:
      - >= MIN_SPEECH_MS of frames above MIN_SPEECH_FRAME_RMS (default 0.08), or
      - a short very loud burst (strong frames + high peak).
    """
    wave = waveform.detach().float().reshape(-1)
    rms = waveform_rms(wave)
    floor = float(MIN_SPEECH_RMS if min_rms is None else min_rms)
    if wave.numel() == 0 or rms < floor:
        return False, rms

    frame_ms = 20
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = wave.numel() // frame_len
    if n_frames <= 0:
        return False, rms

    frames = wave[: n_frames * frame_len].view(n_frames, frame_len)
    frame_rms = torch.sqrt(torch.mean(frames * frames, dim=1))
    peak = float(wave.abs().max().item())

    active_ms = float((frame_rms >= MIN_SPEECH_FRAME_RMS).sum().item() * frame_ms)
    strong_ms = float(
        (frame_rms >= MIN_SPEECH_STRONG_FRAME_RMS).sum().item() * frame_ms
    )

    long_enough = active_ms >= MIN_SPEECH_MS
    short_loud = strong_ms >= MIN_SPEECH_STRONG_MS and peak >= MIN_SPEECH_PEAK
    return bool(long_enough or short_loud), rms


def _decode_waveform(data: bytes) -> tuple[torch.Tensor, int]:
    errors: list[str] = []

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).mean(dim=1)
        return wave, int(sr)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"soundfile: {exc}")

    try:
        wave, sr = torchaudio.load(io.BytesIO(data))
        if wave.ndim > 1:
            wave = wave.mean(dim=0)
        return wave, int(sr)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torchaudio: {exc}")

    hint = (
        "Upload 16-bit PCM WAV (RIFF). Browser WebM is not supported without ffmpeg."
    )
    raise ValueError(f"Could not decode audio ({'; '.join(errors)}). {hint}")
