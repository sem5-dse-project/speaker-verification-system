"""Tests for the replay CNN skeleton."""

from __future__ import annotations

import numpy as np
import torch

from voice_auth.replay_detection.features import extract_logmel
from voice_auth.replay_detection.inference import ReplayDetectorInference
from voice_auth.replay_detection.model import ReplayCNN


def test_replay_cnn_forward_shape() -> None:
    model = ReplayCNN(n_mels=64, channels=(16, 32), dropout=0.1)
    x = torch.randn(4, 1, 64, 50)
    logits = model(x)
    assert logits.shape == (4,)
    probs = model.predict_proba(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_logmel_shape() -> None:
    wave = np.random.randn(1, 16000).astype(np.float32)
    feats = extract_logmel(wave, sample_rate=16000, n_mels=40, duration_sec=1.0)
    assert feats.ndim == 3
    assert feats.shape[0] == 1
    assert feats.shape[1] == 40


def test_replay_inference_probability_range() -> None:
    detector = ReplayDetectorInference(threshold=0.5, n_mels=32, duration_sec=1.0)
    wave = np.random.randn(1, 16000).astype(np.float32)
    prob = detector.predict_replay_probability(wave, 16000)
    assert 0.0 <= prob <= 1.0
