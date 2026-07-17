"""Threshold calibration helpers."""

from __future__ import annotations

from typing import Sequence

from voice_auth.common.metrics import compute_eer, far_frr


def calibrate_threshold_eer(
    scores: Sequence[float],
    labels: Sequence[int],
) -> dict[str, float]:
    """
    Calibrate a decision threshold at the Equal Error Rate operating point.

    Args:
        scores: Development-set similarity scores.
        labels: Binary labels (1 = target, 0 = non-target).

    Returns:
        Dict with ``threshold``, ``eer``, ``far``, and ``frr``.
    """
    eer, threshold = compute_eer(scores, labels)
    far, frr = far_frr(scores, labels, threshold)
    return {
        "threshold": float(threshold),
        "eer": float(eer),
        "far": float(far),
        "frr": float(frr),
    }


def calibrate_threshold_target_far(
    scores: Sequence[float],
    labels: Sequence[int],
    target_far: float = 0.01,
    num_thresholds: int = 1001,
) -> dict[str, float]:
    """
    Choose the highest threshold whose FAR is <= ``target_far``.

    Placeholder sweep; Member 1 may replace with isotonic / logistic calibration.
    """
    import numpy as np

    scores_arr = np.asarray(scores, dtype=np.float64)
    thresholds = np.linspace(scores_arr.min(), scores_arr.max(), num=num_thresholds)
    chosen = float(thresholds[-1])
    chosen_far, chosen_frr = 1.0, 1.0

    for thr in sorted(thresholds, reverse=True):
        far, frr = far_frr(scores, labels, float(thr))
        if far <= target_far:
            chosen = float(thr)
            chosen_far, chosen_frr = far, frr
            break

    return {
        "threshold": chosen,
        "far": float(chosen_far),
        "frr": float(chosen_frr),
        "target_far": float(target_far),
    }
