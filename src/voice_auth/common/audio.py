"""Audio loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Waveform


def ensure_mono(waveform: Waveform) -> Waveform:
    """
    Convert a waveform to mono shape ``[1, num_samples]``.

    Args:
        waveform: Array of shape ``[num_samples]``, ``[1, num_samples]``,
            or ``[channels, num_samples]``.

    Returns:
        Mono waveform of shape ``[1, num_samples]``, float32.
    """
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr.astype(np.float32, copy=False)
        # Average channels → mono
        return np.mean(arr, axis=0, keepdims=True).astype(np.float32)
    raise ValueError(f"Expected 1-D or 2-D waveform, got shape {arr.shape}")


def resample_placeholder(waveform: Waveform, orig_sr: int, target_sr: int) -> Waveform:
    """
    Placeholder linear resampling (not production-quality).

    Member implementations should replace this with ``librosa`` or ``torchaudio``.

    Args:
        waveform: Mono waveform ``[1, num_samples]``.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform ``[1, new_num_samples]``.
    """
    if orig_sr == target_sr:
        return ensure_mono(waveform)
    mono = ensure_mono(waveform).squeeze(0)
    duration = mono.shape[0] / orig_sr
    new_length = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=mono.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    resampled = np.interp(x_new, x_old, mono).astype(np.float32)
    return resampled[np.newaxis, :]


def to_float32(waveform: Waveform) -> Waveform:
    """Cast waveform to float32 and clip to ``[-1, 1]`` if integer-sourced."""
    arr = np.asarray(waveform)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / max(abs(info.min), info.max)
    else:
        arr = arr.astype(np.float32, copy=False)
    return ensure_mono(arr)


def preprocess_audio(
    waveform: Waveform,
    sample_rate: int,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[Waveform, int]:
    """
    Standard preprocessing: mono, float32, resample to target rate.

    Args:
        waveform: Input waveform (any common layout).
        sample_rate: Original sample rate.
        target_sr: Target sample rate (default 16000).

    Returns:
        Tuple of ``(preprocessed_waveform, target_sr)``.
    """
    wave = to_float32(waveform)
    wave = ensure_mono(wave)
    wave = resample_placeholder(wave, sample_rate, target_sr)
    return wave, target_sr


def load_audio(path: Path, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[Waveform, int]:
    """
    Load an audio file from disk and preprocess it (mono, float32, resampled).

    Requires the ``audio`` extra (``pip install -e ".[audio]"``) for ``soundfile``.

    Args:
        path: Path to a WAV/FLAC/OGG file readable by ``soundfile``.
        target_sr: Target sample rate (default 16000).

    Returns:
        Tuple of ``(preprocessed_waveform, target_sr)``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImportError: If ``soundfile`` is not installed.
        RuntimeError: If the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Loading audio requires 'soundfile'. Install it with "
            "'pip install -e \".[audio]\"' or 'pip install soundfile'."
        ) from exc

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode audio file: {path}") from exc

    # soundfile returns [num_samples] or [num_samples, channels]; our convention is [channels, num_samples]
    if data.ndim == 2:
        data = data.T
    return preprocess_audio(data, sample_rate, target_sr)


def l2_normalize(vector: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """L2-normalize a 1-D vector."""
    vec = np.asarray(vector, dtype=np.float32)
    if vec.ndim != 1:
        raise ValueError(f"Expected 1-D vector, got shape {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return (vec / norm).astype(np.float32)
