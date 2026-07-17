"""Quality-Conditioned Fusion Gate (PyTorch skeleton)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from voice_auth.common.types import DEFAULT_EMBEDDING_DIM, DEFAULT_QUALITY_DIM, Embedding, QualityVector


def _l2_normalize_torch(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def build_fusion_gate_input(
    original_embedding: torch.Tensor,
    enhanced_embedding: torch.Tensor,
    quality_vector: torch.Tensor,
) -> torch.Tensor:
    """
    Build the FusionGate input vector.

    Components:
        - quality vector (Q)
        - cosine distance between embeddings (1)
        - absolute embedding difference (D)

    Total dim = Q + 1 + D (default 5 + 1 + 192 = 198).
    """
    e_o = _l2_normalize_torch(original_embedding)
    e_e = _l2_normalize_torch(enhanced_embedding)
    cosine_sim = (e_o * e_e).sum(dim=-1, keepdim=True)
    cosine_dist = 1.0 - cosine_sim
    abs_diff = (e_o - e_e).abs()
    return torch.cat([quality_vector, cosine_dist, abs_diff], dim=-1)


class FusionGate(nn.Module):
    """
    Quality-Conditioned Fusion Gate.

    Architecture: Linear(198, 64) → ReLU → Linear(64, 1) → Sigmoid → alpha.
    Fusion: ``e_fused = alpha * e_original + (1 - alpha) * e_enhanced``,
    with L2-normalization of inputs and output.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        quality_dim: int = DEFAULT_QUALITY_DIM,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.quality_dim = quality_dim
        input_dim = quality_dim + 1 + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        original_embedding: torch.Tensor,
        enhanced_embedding: torch.Tensor,
        quality_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            original_embedding: ``[B, D]`` or ``[D]``.
            enhanced_embedding: ``[B, D]`` or ``[D]``.
            quality_vector: ``[B, Q]`` or ``[Q]``.

        Returns:
            ``(fused_embedding, alpha)`` with matching batch layout.
        """
        squeeze = original_embedding.ndim == 1
        if squeeze:
            original_embedding = original_embedding.unsqueeze(0)
            enhanced_embedding = enhanced_embedding.unsqueeze(0)
            quality_vector = quality_vector.unsqueeze(0)

        e_o = _l2_normalize_torch(original_embedding.float())
        e_e = _l2_normalize_torch(enhanced_embedding.float())
        q = quality_vector.float()

        gate_in = build_fusion_gate_input(e_o, e_e, q)
        alpha = self.net(gate_in)  # [B, 1]
        fused = alpha * e_o + (1.0 - alpha) * e_e
        fused = _l2_normalize_torch(fused)

        if squeeze:
            return fused.squeeze(0), alpha.squeeze()
        return fused, alpha.squeeze(-1)


class QualityConditionedFusion:
    """
    NumPy-friendly wrapper implementing the EmbeddingFusion protocol.
    """

    def __init__(self, model: FusionGate | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model if model is not None else FusionGate()
        self.model.to(self.device)
        self.model.eval()

    def fuse(
        self,
        original_embedding: Embedding,
        enhanced_embedding: Embedding,
        quality_vector: QualityVector,
    ) -> tuple[Embedding, float]:
        """Fuse embeddings and return ``(fused_embedding, alpha)``."""
        e_o = torch.from_numpy(np.asarray(original_embedding, dtype=np.float32)).to(self.device)
        e_e = torch.from_numpy(np.asarray(enhanced_embedding, dtype=np.float32)).to(self.device)
        q = torch.from_numpy(np.asarray(quality_vector, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            fused, alpha = self.model(e_o, e_e, q)
        return fused.cpu().numpy().astype(np.float32), float(alpha.cpu().item())
