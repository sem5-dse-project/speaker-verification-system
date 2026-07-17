"""Evaluation helpers comparing original / enhanced / fused embedding paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from voice_auth.common.audio import l2_normalize
from voice_auth.common.metrics import compute_eer, far_frr
from voice_auth.speaker_verification.scoring import cosine_similarity


def fixed_alpha_fuse(
    original_embedding: np.ndarray,
    enhanced_embedding: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Fuse with a constant alpha and L2-normalize the result."""
    e_o = l2_normalize(np.asarray(original_embedding, dtype=np.float32))
    e_e = l2_normalize(np.asarray(enhanced_embedding, dtype=np.float32))
    fused = alpha * e_o + (1.0 - alpha) * e_e
    return l2_normalize(fused)


def evaluate_path_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float]:
    """Compute EER / FAR / FRR for one embedding path."""
    eer, eer_thr = compute_eer(scores, labels)
    far, frr = far_frr(scores, labels, threshold)
    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "far": float(far),
        "frr": float(frr),
        "threshold": float(threshold),
    }


def score_against_templates(
    templates: Sequence[np.ndarray],
    embeddings: Sequence[np.ndarray],
) -> list[float]:
    """Cosine-score each embedding against its paired enrollment template."""
    return [
        cosine_similarity(t, e)
        for t, e in zip(templates, embeddings, strict=True)
    ]
