"""Tests for cosine scoring and enrollment averaging."""

from __future__ import annotations

import numpy as np

from voice_auth.common.audio import l2_normalize
from voice_auth.speaker_verification.encoders import MockSpeakerEncoder
from voice_auth.speaker_verification.enrollment import create_enrollment_template
from voice_auth.speaker_verification.scoring import cosine_similarity, decide


def test_cosine_similarity_identical() -> None:
    a = l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-5


def test_decide_threshold() -> None:
    assert decide(0.8, 0.7) is True
    assert decide(0.6, 0.7) is False


def test_enrollment_template_shape() -> None:
    encoder = MockSpeakerEncoder(embedding_dim=192, seed=1)
    waves = [np.random.randn(1, 16000).astype(np.float32) for _ in range(3)]
    rates = [16000, 16000, 16000]
    template = create_enrollment_template("user_a", waves, rates, encoder, encoder_name="mock")
    assert template.user_id == "user_a"
    assert template.embedding_dim == 192
    assert template.num_utterances == 3
    emb = template.as_array()
    assert emb.shape == (192,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4
