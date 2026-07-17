"""Verification pipeline with replay gate and quality-conditioned fusion."""

from __future__ import annotations

from dataclasses import dataclass

from voice_auth.common.audio import preprocess_audio
from voice_auth.common.interfaces import (
    EmbeddingFusion,
    QualityEstimator,
    ReplayDetector,
    SpeakerEncoder,
    SpeechEnhancer,
)
from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Embedding, Waveform
from voice_auth.speaker_verification.enrollment import EnrollmentTemplate
from voice_auth.speaker_verification.scoring import cosine_similarity, decide


@dataclass
class VerificationResult:
    """Outcome of a single verification attempt."""

    user_id: str
    accepted: bool
    score: float
    threshold: float
    replay_probability: float
    rejected_as_replay: bool
    alpha: float | None
    fused_embedding_dim: int | None = None


def run_verification(
    claimed_user_id: str,
    waveform: Waveform,
    sample_rate: int,
    template: EnrollmentTemplate,
    encoder: SpeakerEncoder,
    replay_detector: ReplayDetector,
    enhancer: SpeechEnhancer,
    quality_estimator: QualityEstimator,
    fusion: EmbeddingFusion,
    threshold: float,
    replay_threshold: float = 0.5,
) -> VerificationResult:
    """
    Verify a claimed identity against an enrollment template.

    Flow:
        1. Preprocess audio.
        2. Estimate replay probability; reject if suspicious.
        3. Encode original audio.
        4. Enhance audio and encode again.
        5. Extract quality vector and fuse embeddings.
        6. Cosine-score against the template and apply threshold.
    """
    if template.user_id != claimed_user_id:
        # Still score, but callers may treat ID mismatch separately.
        pass

    processed, sr = preprocess_audio(waveform, sample_rate, DEFAULT_SAMPLE_RATE)
    replay_prob = float(replay_detector.predict_replay_probability(processed, sr))
    if replay_prob >= replay_threshold:
        return VerificationResult(
            user_id=claimed_user_id,
            accepted=False,
            score=0.0,
            threshold=threshold,
            replay_probability=replay_prob,
            rejected_as_replay=True,
            alpha=None,
        )

    e_orig = encoder.encode(processed, sr)
    enhanced = enhancer.enhance(processed, sr)
    e_enh = encoder.encode(enhanced, sr)
    quality = quality_estimator.extract(processed, sr)
    fused, alpha = fusion.fuse(e_orig, e_enh, quality)

    enroll_emb: Embedding = template.as_array()
    score = cosine_similarity(enroll_emb, fused)
    accepted = decide(score, threshold)

    return VerificationResult(
        user_id=claimed_user_id,
        accepted=accepted,
        score=score,
        threshold=threshold,
        replay_probability=replay_prob,
        rejected_as_replay=False,
        alpha=float(alpha),
        fused_embedding_dim=int(fused.shape[0]),
    )
