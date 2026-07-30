"""Unit tests for enrollment template averaging and verify decisions."""

from __future__ import annotations

import numpy as np
import pytest

from ml_server.scoring import average_template, cosine_similarity, decide


def test_average_template_normalizes():
    embs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    template = average_template(embs)
    assert template.shape == (3,)
    assert abs(float(np.linalg.norm(template)) - 1.0) < 1e-5


def test_average_template_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        average_template(np.zeros((0, 3), dtype=np.float32))


def test_cosine_similarity_identical_vectors():
    a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_decide_accept_above_threshold():
    result = decide(0.8, threshold=0.25)
    assert result["accepted"] is True
    assert result["decision"] == "ACCEPT"
    assert result["threshold"] == 0.25
    assert result["score"] == 0.8


def test_decide_reject_below_threshold():
    result = decide(0.1, threshold=0.25)
    assert result["accepted"] is False
    assert result["decision"] == "REJECT"
