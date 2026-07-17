"""Evaluation metrics for speaker verification and replay detection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def far_frr(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> tuple[float, float]:
    """
    Compute False Acceptance Rate and False Rejection Rate.

    Args:
        scores: Similarity scores (higher = more likely same speaker).
        labels: Binary labels where 1 = target (same speaker), 0 = non-target.
        threshold: Decision threshold; accept if score >= threshold.

    Returns:
        Tuple of ``(FAR, FRR)``.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError("scores and labels must have the same shape")

    target = labels_arr == 1
    nontarget = labels_arr == 0

    if nontarget.sum() == 0:
        far = 0.0
    else:
        far = float(np.mean(scores_arr[nontarget] >= threshold))

    if target.sum() == 0:
        frr = 0.0
    else:
        frr = float(np.mean(scores_arr[target] < threshold))

    return far, frr


def compute_eer(
    scores: Sequence[float],
    labels: Sequence[int],
    num_thresholds: int = 1001,
) -> tuple[float, float]:
    """
    Estimate Equal Error Rate by sweeping thresholds.

    Args:
        scores: Similarity scores.
        labels: Binary labels (1 = target, 0 = non-target).
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        Tuple of ``(eer, threshold_at_eer)``.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.size == 0:
        return 0.0, 0.0

    thresholds = np.linspace(scores_arr.min(), scores_arr.max(), num=num_thresholds)
    best_eer = 1.0
    best_thr = float(thresholds[0])
    best_diff = float("inf")

    for thr in thresholds:
        far, frr = far_frr(scores, labels, float(thr))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = 0.5 * (far + frr)
            best_thr = float(thr)

    return float(best_eer), best_thr


def binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float | list[list[int]]]:
    """
    Compute precision, recall, F1 and confusion matrix for binary labels.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Dict with precision, recall, f1, and confusion_matrix [[TN, FP], [FN, TP]].
    """
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
