"""Fusion model registry – supports multiple architectures."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_server.config import DEVICE, FUSION_MODEL_TYPE


# ----------------------------------------------------------------------
# 1. PaperFusionMLP (original 3‑layer MLP)
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, embedding_dim: int = 192, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, noisy_emb, enhanced_emb, quality_vec=None):
        x = torch.cat([noisy_emb, enhanced_emb], dim=-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        fused = self.fc3(x)
        return F.normalize(fused, p=2, dim=-1)


# ----------------------------------------------------------------------
# 2. CrossAttentionFusion (your original cross‑attention gate)
# ----------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, embedding_dim: int = 192, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, noisy_emb, enhanced_emb, quality_vec=None):
        n = self.proj(noisy_emb)
        e = self.proj(enhanced_emb)
        x = torch.stack([n, e], dim=1)
        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm1(x + attn_out)
        pooled = x.mean(dim=1)
        mlp_out = self.mlp(pooled)
        out = self.norm2(pooled + mlp_out)
        fused = self.out_proj(out)
        return F.normalize(fused, p=2, dim=-1)


# ----------------------------------------------------------------------
# 3. NoiseAwareFusion (your latest model with noise extraction)
# ----------------------------------------------------------------------
class NoiseAwareFusion(nn.Module):
    def __init__(self, embedding_dim=192, num_heads=4, dropout=0.1, noise_bottleneck_dim=128):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim

        # Shared projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        # Estimate noise-related information
        self.noise_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 2, noise_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(noise_bottleneck_dim, embedding_dim),
        )

        # Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Noise gate
        self.noise_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.Sigmoid(),
        )

        # Transformer-style blocks
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, noisy_emb, enhanced_emb):
        noisy_emb = F.normalize(noisy_emb, p=2, dim=-1)
        enhanced_emb = F.normalize(enhanced_emb, p=2, dim=-1)
        n = self.proj(noisy_emb)
        e = self.proj(enhanced_emb)
        noise_est = self.noise_extractor(torch.cat([noisy_emb, enhanced_emb], dim=-1))
        x = torch.stack([n, e, noise_est], dim=1)  # Shape: B x 3 x D

        attn_out, attn_weights = self.cross_attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        token_importance = attn_weights.mean(dim=1)
        pooled = (x * token_importance.unsqueeze(-1)).sum(dim=1)
        gate = self.noise_gate(noise_est)
        pooled = gate * pooled + (1.0 - gate) * x.mean(dim=1)
        mlp_out = self.mlp(pooled)
        out = self.norm2(pooled + mlp_out)
        correction = self.out_proj(out)
        fused = noisy_emb + self.alpha * correction
        fused = F.normalize(fused, p=2, dim=-1)

        return fused, noise_est


# ----------------------------------------------------------------------
# Registry & Loader
# ----------------------------------------------------------------------
_MODEL_REGISTRY = {
    "mlp": MLP,
    "cross_attention": CrossAttentionFusion,
    "noise_aware": NoiseAwareFusion,
}


def load_fusion_model(
    checkpoint_path: str | Path,
    model_type: str = "noise_aware",
    device: str = DEVICE,
    **kwargs,
) -> nn.Module:
    """
    Load a fusion model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt file.
        model_type: One of ['mlp', 'cross_attention', 'noise_aware'].
        device: 'cpu' or 'cuda'.
        **kwargs: Additional arguments to pass to the model constructor.
    """
    model_cls = _MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown model_type: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    # Instantiate the model with default dims (override via kwargs)
    model = model_cls(**kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint may contain a key 'gate_state' or be the full state_dict
    if "gate_state" in ckpt:
        state = ckpt["gate_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
