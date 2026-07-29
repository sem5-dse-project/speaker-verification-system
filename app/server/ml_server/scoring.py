"""Template averaging and cosine verification scoring."""

from __future__ import annotations

import numpy as np

from ml_server.config import DEFAULT_THRESHOLD


def average_template(embeddings: np.ndarray) -> np.ndarray:
    """Mean of L2-normalized embeddings, then L2-normalize again."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("Need at least one embedding row")
    mean = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(mean)) + 1e-8
    return (mean / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def decide(score: float, threshold: float | None = None) -> dict:
    thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    accepted = score >= thr
    return {
        "score": float(score),
        "threshold": thr,
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECT",
    }
