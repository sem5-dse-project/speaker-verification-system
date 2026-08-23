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
# 2. SelfAttentionFusion (your original self‑attention gate)
# ----------------------------------------------------------------------


class SelfAttentionFusion(nn.Module):
    def __init__(self, embedding_dim=192, num_heads=4, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn = nn.MultiheadAttention(
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

    def forward(self, noisy_emb, enhanced_emb):
        n = self.proj(noisy_emb)
        e = self.proj(enhanced_emb)

        # FIX: Ensure contiguous memory layout for the attention backend
        x = torch.stack([n, e], dim=1).contiguous()

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        pooled = x.mean(dim=1)
        mlp_out = self.mlp(pooled)
        out = self.norm2(pooled + mlp_out)
        fused = self.out_proj(out)
        return F.normalize(fused, p=2, dim=-1)


# ----------------------------------------------------------------------
# MODEL – Cross-Attention
# ----------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, embedding_dim=192, num_heads=4, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, noisy_emb, enhanced_emb):
        B, D = noisy_emb.shape
        Q = self.q_proj(noisy_emb)
        K = self.k_proj(enhanced_emb)
        V = self.v_proj(enhanced_emb)

        Q = Q.view(B, self.num_heads, self.head_dim)
        K = K.view(B, self.num_heads, self.head_dim)
        V = V.view(B, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.contiguous().view(B, D)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(noisy_emb + attn_out)
        mlp_out = self.mlp(x)
        out = self.norm2(x + mlp_out)
        fused = self.fc_out(out)
        return F.normalize(fused, p=2, dim=-1)


# ----------------------------------------------------------------------
# Registry & Loader
# ----------------------------------------------------------------------
_MODEL_REGISTRY = {
    "mlp": MLP,
    "self_attention": SelfAttentionFusion,
    "cross_attention": CrossAttentionFusion,
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
