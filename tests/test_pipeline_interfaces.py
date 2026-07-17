"""Tests for shared protocols and end-to-end pipeline wiring."""

from __future__ import annotations

import numpy as np

from voice_auth.common.interfaces import (
    EmbeddingFusion,
    QualityEstimator,
    ReplayDetector,
    SpeakerEncoder,
    SpeechEnhancer,
)
from voice_auth.pipeline.enrollment_pipeline import run_enrollment
from voice_auth.pipeline.verification_pipeline import run_verification
from voice_auth.quality_fusion.enhancement import IdentityEnhancer
from voice_auth.quality_fusion.fusion_gate import QualityConditionedFusion
from voice_auth.quality_fusion.quality_features import BasicQualityEstimator
from voice_auth.replay_detection.inference import ReplayDetectorInference
from voice_auth.speaker_verification.encoders import MockSpeakerEncoder


def test_protocol_compliance() -> None:
    encoder = MockSpeakerEncoder()
    detector = ReplayDetectorInference(n_mels=32, duration_sec=1.0)
    enhancer = IdentityEnhancer()
    quality = BasicQualityEstimator()
    fusion = QualityConditionedFusion()

    assert isinstance(encoder, SpeakerEncoder)
    assert isinstance(detector, ReplayDetector)
    assert isinstance(enhancer, SpeechEnhancer)
    assert isinstance(quality, QualityEstimator)
    assert isinstance(fusion, EmbeddingFusion)


def test_enrollment_and_verification_pipeline() -> None:
    encoder = MockSpeakerEncoder(embedding_dim=192, seed=7)
    waves = [np.random.randn(1, 8000).astype(np.float32) for _ in range(2)]
    rates = [16000, 16000]
    template = run_enrollment("u1", waves, rates, encoder, encoder_name="mock")
    assert template.user_id == "u1"

    # Low replay threshold disabled via very high threshold so path continues
    detector = ReplayDetectorInference(threshold=1.1, n_mels=32, duration_sec=0.5)
    result = run_verification(
        claimed_user_id="u1",
        waveform=np.random.randn(1, 8000).astype(np.float32),
        sample_rate=16000,
        template=template,
        encoder=encoder,
        replay_detector=detector,
        enhancer=IdentityEnhancer(),
        quality_estimator=BasicQualityEstimator(),
        fusion=QualityConditionedFusion(),
        threshold=-1.0,  # accept any cosine score for wiring test
        replay_threshold=1.1,
    )
    assert result.rejected_as_replay is False
    assert result.alpha is not None
    assert 0.0 <= result.alpha <= 1.0
    assert result.fused_embedding_dim == 192
    assert result.accepted is True
