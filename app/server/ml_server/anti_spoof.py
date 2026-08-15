"""Cascade: inverted-Mel replay + optional LFCC-LA synthetic scores."""

from __future__ import annotations

from ml_server.config import DEVICE, LA_ENABLED, LA_HARD_GATE
from ml_server.la_spoof import score_la
from ml_server.replay import score_replay


def _stage_view(result: dict) -> dict:
    return {
        "score": result["score"],
        "threshold": result["threshold"],
        "threshold_low": result.get("threshold_low"),
        "threshold_high": result.get("threshold_high"),
        "decision": result["decision"],
        "feature_type": result.get("feature_type"),
    }


def score_anti_spoof(
    waveform,
    threshold: float | None = None,
    la_threshold: float | None = None,
    device: str = DEVICE,
) -> dict:
    """
    Speech gate + inverted-Mel replay, then optional LFCC-LA scoring.

    Final decision is driven by inverted-Mel unless ``LA_HARD_GATE`` is on
    (LFCC-LA over-fires ~1.0 on browser-mic speech).

    Decision: NO_SPEECH | REPLAY | SYNTHETIC | UNCERTAIN | LIVE.
    """
    replay = score_replay(waveform, threshold=threshold, device=device)
    replay_stage = _stage_view(replay)

    if replay["decision"] == "NO_SPEECH":
        return {
            **replay,
            "is_synthetic": False,
            "la": None,
            "replay": replay_stage,
        }

    la = None
    la_stage = None
    if LA_ENABLED:
        la = score_la(
            waveform,
            threshold=la_threshold,
            device=device,
            check_speech=False,
        )
        la_stage = _stage_view(la)

    if replay["decision"] == "REPLAY":
        return {
            **replay,
            "is_synthetic": False,
            "accepted": False,
            "decision": "REPLAY",
            "feature_type": _feature_label(replay, la),
            "replay": replay_stage,
            "la": la_stage,
        }

    # Hard LA gate only when explicitly enabled (lab / ASVspoof files).
    if (
        LA_HARD_GATE
        and la is not None
        and la["decision"] == "SYNTHETIC"
    ):
        return {
            "score": la["score"],
            "threshold": la["threshold"],
            "threshold_low": la.get("threshold_low"),
            "threshold_high": la.get("threshold_high"),
            "is_replay": False,
            "is_synthetic": True,
            "accepted": False,
            "decision": "SYNTHETIC",
            "feature_type": _feature_label(replay, la),
            "rms": replay.get("rms"),
            "replay": replay_stage,
            "la": la_stage,
        }

    uncertain = replay["decision"] == "UNCERTAIN"
    if LA_HARD_GATE and la is not None and la["decision"] == "UNCERTAIN":
        uncertain = True

    if uncertain:
        src = replay
        if (
            LA_HARD_GATE
            and la is not None
            and la["decision"] == "UNCERTAIN"
            and replay["decision"] != "UNCERTAIN"
        ):
            src = la
        return {
            "score": src["score"],
            "threshold": src["threshold"],
            "threshold_low": src.get("threshold_low"),
            "threshold_high": src.get("threshold_high"),
            "is_replay": False,
            "is_synthetic": False,
            "accepted": False,
            "decision": "UNCERTAIN",
            "feature_type": _feature_label(replay, la),
            "rms": replay.get("rms"),
            "replay": replay_stage,
            "la": la_stage,
        }

    return {
        "score": replay["score"],
        "threshold": replay["threshold"],
        "threshold_low": replay.get("threshold_low"),
        "threshold_high": replay.get("threshold_high"),
        "is_replay": False,
        "is_synthetic": False,
        "accepted": True,
        "decision": "LIVE",
        "feature_type": _feature_label(replay, la),
        "rms": replay.get("rms"),
        "replay": replay_stage,
        "la": la_stage,
    }


def _feature_label(replay: dict, la: dict | None) -> str:
    r = replay.get("feature_type") or "replay"
    if la is None:
        return r
    return f"{r}+{la.get('feature_type') or 'la'}"
