"""Cosine-similarity scoring and threshold decisions."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import l2_normalize
from voice_auth.common.types import Embedding


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """
    Cosine similarity between two embeddings.

    Both vectors are L2-normalized before the dot product.
    """
    a_n = l2_normalize(np.asarray(a, dtype=np.float32))
    b_n = l2_normalize(np.asarray(b, dtype=np.float32))
    if a_n.shape != b_n.shape:
        raise ValueError(f"Embedding shapes differ: {a_n.shape} vs {b_n.shape}")
    return float(np.dot(a_n, b_n))


def decide(score: float, threshold: float) -> bool:
    """
    Accept if ``score >= threshold``.

    Args:
        score: Cosine similarity score.
        threshold: Calibrated decision threshold.

    Returns:
        True if the claimed identity is accepted.
    """
    return bool(score >= threshold)
