"""LFCC / Mel / inverted-Mel replay CNN loader and scoring for the ML server."""

from __future__ import annotations

from pathlib import Path

import torch

from ml_server.audio import has_sufficient_speech
from ml_server.config import (
    DEVICE,
    REPLAY_CHECKPOINT,
    REPLAY_MARGIN,
    REPLAY_T_HIGH,
    REPLAY_T_LOW,
    REPLAY_THRESHOLD,
)
from ml_server.replay_model import AudioConfig, ReplayCNN, fix_length

_replay_model = None
_replay_threshold: float | None = None
_replay_config: AudioConfig | None = None
_replay_ckpt_path: Path | None = None


def _audio_config_from_ckpt(ckpt: dict) -> AudioConfig:
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg and "feature_type" in ckpt:
        cfg["feature_type"] = ckpt["feature_type"]
    allowed = set(AudioConfig.__dataclass_fields__)
    return AudioConfig(**{k: v for k, v in cfg.items() if k in allowed})


def resolve_band_thresholds(center: float) -> tuple[float, float, float]:
    """Return (center, t_low, t_high) clamped to [0, 1] with t_low < t_high."""
    center = float(center)
    if REPLAY_T_LOW is not None and REPLAY_T_HIGH is not None:
        t_low = float(REPLAY_T_LOW)
        t_high = float(REPLAY_T_HIGH)
    else:
        margin = max(0.0, float(REPLAY_MARGIN))
        t_low = center - margin
        t_high = center + margin
    t_low = max(0.0, min(1.0, t_low))
    t_high = max(0.0, min(1.0, t_high))
    if t_low >= t_high:
        # Degenerate band → fall back to binary cut at center
        eps = 1e-4
        t_low = max(0.0, center - eps)
        t_high = min(1.0, center + eps)
        if t_low >= t_high:
            t_low, t_high = 0.0, 1.0
    return center, t_low, t_high


def decide_replay_band(score: float, t_low: float, t_high: float) -> str:
    """Map score to LIVE | UNCERTAIN | REPLAY."""
    if score < t_low:
        return "LIVE"
    if score >= t_high:
        return "REPLAY"
    return "UNCERTAIN"


def get_replay_detector(device: str = DEVICE):
    """Lazy-load ReplayCNN + center threshold from checkpoint."""
    global _replay_model, _replay_threshold, _replay_config, _replay_ckpt_path

    ckpt_path = Path(REPLAY_CHECKPOINT)
    if _replay_model is not None and _replay_ckpt_path == ckpt_path:
        return _replay_model, float(_replay_threshold), _replay_config

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Replay checkpoint not found: {ckpt_path}. "
            "Train inverted_mel_mixed_2017_pa2019 or set REPLAY_CHECKPOINT."
        )

    map_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ckpt = torch.load(ckpt_path, map_location=map_device, weights_only=False)
    config = _audio_config_from_ckpt(ckpt)
    model = ReplayCNN(config).to(map_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    thr = (
        float(REPLAY_THRESHOLD)
        if REPLAY_THRESHOLD is not None
        else float(ckpt["threshold"])
    )

    _replay_model = model
    _replay_threshold = thr
    _replay_config = config
    _replay_ckpt_path = ckpt_path
    return model, thr, config


@torch.inference_mode()
def score_replay(
    waveform: torch.Tensor,
    threshold: float | None = None,
    device: str = DEVICE,
) -> dict:
    """Score mono waveform for replay. Returns LIVE|UNCERTAIN|REPLAY|NO_SPEECH."""
    model, ckpt_thr, config = get_replay_detector(device=device)
    center = ckpt_thr if threshold is None else float(threshold)
    center, t_low, t_high = resolve_band_thresholds(center)
    map_device = next(model.parameters()).device

    wave = waveform.detach().float().cpu()
    if wave.ndim > 1:
        wave = wave.mean(dim=0)

    ok_speech, rms = has_sufficient_speech(wave)
    if not ok_speech:
        return {
            "score": 0.0,
            "threshold": center,
            "threshold_low": t_low,
            "threshold_high": t_high,
            "is_replay": False,
            "accepted": False,
            "decision": "NO_SPEECH",
            "feature_type": config.feature_type,
            "rms": rms,
        }

    wave = fix_length(wave, config.samples, random_crop=False)
    batch = wave.unsqueeze(0).to(map_device)

    logit = model(batch).reshape(-1)[0]
    score = float(torch.sigmoid(logit).item())
    decision = decide_replay_band(score, t_low, t_high)
    is_replay = decision == "REPLAY"
    return {
        "score": score,
        "threshold": center,
        "threshold_low": t_low,
        "threshold_high": t_high,
        "is_replay": is_replay,
        "accepted": decision == "LIVE",
        "decision": decision,
        "feature_type": config.feature_type,
        "rms": rms,
    }
