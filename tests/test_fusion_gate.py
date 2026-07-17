"""Tests for the Quality-Conditioned Fusion Gate."""

from __future__ import annotations

import numpy as np
import torch

from voice_auth.quality_fusion.fusion_gate import FusionGate, QualityConditionedFusion, build_fusion_gate_input


def test_fusion_gate_input_dim() -> None:
    e_o = torch.randn(2, 192)
    e_e = torch.randn(2, 192)
    q = torch.randn(2, 5)
    gate_in = build_fusion_gate_input(e_o, e_e, q)
    assert gate_in.shape == (2, 198)


def test_fusion_gate_shapes_and_alpha_range() -> None:
    model = FusionGate(embedding_dim=192, quality_dim=5, hidden_dim=64)
    e_o = torch.randn(3, 192)
    e_e = torch.randn(3, 192)
    q = torch.randn(3, 5)
    fused, alpha = model(e_o, e_e, q)
    assert fused.shape == (3, 192)
    assert alpha.shape == (3,)
    assert torch.all(alpha >= 0) and torch.all(alpha <= 1)


def test_fusion_gate_normalization() -> None:
    model = FusionGate()
    e_o = torch.randn(192)
    e_e = torch.randn(192)
    q = torch.randn(5)
    model.eval()
    with torch.no_grad():
        fused, alpha = model(e_o, e_e, q)
    assert fused.shape == (192,)
    assert alpha.ndim == 0
    assert 0.0 <= float(alpha) <= 1.0
    assert abs(float(fused.norm()) - 1.0) < 1e-4


def test_quality_conditioned_fusion_numpy() -> None:
    fusion = QualityConditionedFusion()
    e_o = np.random.randn(192).astype(np.float32)
    e_e = np.random.randn(192).astype(np.float32)
    q = np.random.randn(5).astype(np.float32)
    fused, alpha = fusion.fuse(e_o, e_e, q)
    assert fused.shape == (192,)
    assert 0.0 <= alpha <= 1.0
    assert abs(float(np.linalg.norm(fused)) - 1.0) < 1e-4
