"""Speaker-model comparison evaluation (placeholder)."""

from __future__ import annotations

import time
from collections.abc import Sequence

from voice_auth.common.metrics import compute_eer, far_frr
from voice_auth.speaker_verification.scoring import cosine_similarity


def evaluate_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float]:
    """Compute EER, FAR, and FRR for a scored trial list."""
    eer, eer_thr = compute_eer(scores, labels)
    far, frr = far_frr(scores, labels, threshold)
    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "far": float(far),
        "frr": float(frr),
        "threshold": float(threshold),
    }


def compare_encoder_inference_time(
    encode_fn,
    waveforms: Sequence,
    sample_rate: int,
    warmup: int = 1,
) -> dict[str, float]:
    """
    Measure average encode latency (placeholder timing harness).

    Args:
        encode_fn: Callable ``(waveform, sample_rate) -> embedding``.
        waveforms: List of waveforms to encode.
        sample_rate: Sample rate for all waveforms.
        warmup: Number of warm-up calls excluded from timing.
    """
    if not waveforms:
        return {"mean_ms": 0.0, "num_utterances": 0.0}

    for i in range(min(warmup, len(waveforms))):
        encode_fn(waveforms[i], sample_rate)

    start = time.perf_counter()
    for wave in waveforms:
        encode_fn(wave, sample_rate)
    elapsed = time.perf_counter() - start
    n = len(waveforms)
    return {
        "mean_ms": float(1000.0 * elapsed / n),
        "total_s": float(elapsed),
        "num_utterances": float(n),
    }


def score_embedding_pairs(
    enroll_embeddings: Sequence,
    test_embeddings: Sequence,
) -> list[float]:
    """Cosine-score paired enrollment/test embeddings."""
    if len(enroll_embeddings) != len(test_embeddings):
        raise ValueError("enroll_embeddings and test_embeddings length mismatch")
    return [
        cosine_similarity(e, t) for e, t in zip(enroll_embeddings, test_embeddings, strict=True)
    ]
