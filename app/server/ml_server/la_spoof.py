"""LFCC-LA (ASVspoof 2019 Logical Access) synthetic-spoof scorer."""

from __future__ import annotations

from pathlib import Path

import torch

from ml_server.audio import has_sufficient_speech
from ml_server.config import (
    DEVICE,
    LA_CHECKPOINT,
    LA_MARGIN,
    LA_T_HIGH,
    LA_T_LOW,
    LA_THRESHOLD,
)
from ml_server.replay import decide_replay_band
from ml_server.replay_model import AudioConfig, ReplayCNN, fix_length

_la_model = None
_la_threshold: float | None = None
_la_config: AudioConfig | None = None
_la_ckpt_path: Path | None = None


def _audio_config_from_ckpt(ckpt: dict) -> AudioConfig:
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg and "feature_type" in ckpt:
        cfg["feature_type"] = ckpt["feature_type"]
    allowed = set(AudioConfig.__dataclass_fields__)
    return AudioConfig(**{k: v for k, v in cfg.items() if k in allowed})


def resolve_la_band_thresholds(center: float) -> tuple[float, float, float]:
    center = float(center)
    if LA_T_LOW is not None and LA_T_HIGH is not None:
        t_low = float(LA_T_LOW)
        t_high = float(LA_T_HIGH)
    else:
        margin = max(0.0, float(LA_MARGIN))
        t_low = center - margin
        t_high = center + margin
    t_low = max(0.0, min(1.0, t_low))
    t_high = max(0.0, min(1.0, t_high))
    if t_low >= t_high:
        eps = 1e-4
        t_low = max(0.0, center - eps)
        t_high = min(1.0, center + eps)
        if t_low >= t_high:
            t_low, t_high = 0.0, 1.0
    return center, t_low, t_high


def decide_la_band(score: float, t_low: float, t_high: float) -> str:
    """Map score to LIVE | UNCERTAIN | SYNTHETIC."""
    band = decide_replay_band(score, t_low, t_high)
    if band == "REPLAY":
        return "SYNTHETIC"
    return band


def get_la_detector(device: str = DEVICE):
    """Lazy-load LFCC-LA ReplayCNN + center threshold from checkpoint."""
    global _la_model, _la_threshold, _la_config, _la_ckpt_path

    ckpt_path = Path(LA_CHECKPOINT)
    if _la_model is not None and _la_ckpt_path == ckpt_path:
        return _la_model, float(_la_threshold), _la_config

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"LA checkpoint not found: {ckpt_path}. "
            "Train lfcc_la2019 or set LA_CHECKPOINT / LA_ENABLED=false."
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
        float(LA_THRESHOLD)
        if LA_THRESHOLD is not None
        else float(ckpt["threshold"])
    )

    _la_model = model
    _la_threshold = thr
    _la_config = config
    _la_ckpt_path = ckpt_path
    return model, thr, config


@torch.inference_mode()
def score_la(
    waveform: torch.Tensor,
    threshold: float | None = None,
    device: str = DEVICE,
    check_speech: bool = True,
) -> dict:
    """Score mono waveform for synthetic spoof. LIVE|UNCERTAIN|SYNTHETIC|NO_SPEECH."""
    model, ckpt_thr, config = get_la_detector(device=device)
    center = ckpt_thr if threshold is None else float(threshold)
    center, t_low, t_high = resolve_la_band_thresholds(center)
    map_device = next(model.parameters()).device

    wave = waveform.detach().float().cpu()
    if wave.ndim > 1:
        wave = wave.mean(dim=0)

    rms = None
    if check_speech:
        ok_speech, rms = has_sufficient_speech(wave)
        if not ok_speech:
            return {
                "score": 0.0,
                "threshold": center,
                "threshold_low": t_low,
                "threshold_high": t_high,
                "is_synthetic": False,
                "accepted": False,
                "decision": "NO_SPEECH",
                "feature_type": config.feature_type,
                "rms": rms,
            }

    wave = fix_length(wave, config.samples, random_crop=False)
    batch = wave.unsqueeze(0).to(map_device)

    logit = model(batch).reshape(-1)[0]
    score = float(torch.sigmoid(logit).item())
    decision = decide_la_band(score, t_low, t_high)
    is_synthetic = decision == "SYNTHETIC"
    return {
        "score": score,
        "threshold": center,
        "threshold_low": t_low,
        "threshold_high": t_high,
        "is_synthetic": is_synthetic,
        "accepted": decision == "LIVE",
        "decision": decision,
        "feature_type": config.feature_type,
        "rms": rms,
    }
