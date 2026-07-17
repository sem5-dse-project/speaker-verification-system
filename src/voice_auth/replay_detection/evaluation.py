"""Replay-detection evaluation helpers."""

from __future__ import annotations

from typing import Sequence

from voice_auth.common.metrics import binary_classification_metrics, compute_eer


def evaluate_replay_scores(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate replay detection with EER, precision, recall, F1 and confusion matrix.

    Args:
        probabilities: Predicted replay probabilities in ``[0, 1]``.
        labels: Ground truth (0 = bona fide, 1 = replay).
        threshold: Classification threshold.
    """
    preds = [1 if p >= threshold else 0 for p in probabilities]
    clf = binary_classification_metrics(labels, preds)
    # For EER, treat probability as score and label 1 = positive (replay)
    eer, eer_thr = compute_eer(probabilities, labels)
    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "threshold": float(threshold),
        **clf,
    }
