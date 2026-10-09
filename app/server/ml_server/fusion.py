"""Fusion model registry — v11 FrameGateFusion + legacy models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_server.config import DEVICE, FUSION_MODEL_TYPE


# ============================================================================
# 1. PaperFusionMLP (original 3-layer MLP) — unchanged
# ============================================================================
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


# ============================================================================
# 2. CrossAttentionFusion — unchanged
# ============================================================================
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


# ============================================================================
# 3. NoiseAwareFusion — unchanged
# ============================================================================
class NoiseAwareFusion(nn.Module):
    def __init__(self, embedding_dim=192, num_heads=4, dropout=0.1, noise_bottleneck_dim=128):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim

        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.noise_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 2, noise_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(noise_bottleneck_dim, embedding_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.noise_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.Sigmoid(),
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
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, noisy_emb, enhanced_emb):
        noisy_emb = F.normalize(noisy_emb, p=2, dim=-1)
        enhanced_emb = F.normalize(enhanced_emb, p=2, dim=-1)
        n = self.proj(noisy_emb)
        e = self.proj(enhanced_emb)
        noise_est = self.noise_extractor(torch.cat([noisy_emb, enhanced_emb], dim=-1))
        x = torch.stack([n, e, noise_est], dim=1)

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
        return F.normalize(fused, p=2, dim=-1), noise_est


# ============================================================================
# 4. SelfAttentionFusion (v6 — utterance-level) — kept for legacy checkpoints
# ============================================================================
class SelfAttentionFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2, embedding_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.gate_head = nn.Linear(embedding_dim, 1)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, noisy_emb, enhanced_emb, quality_vec=None, return_beta: bool = False):
        n = self.proj(noisy_emb)
        e = self.proj(enhanced_emb)
        x = torch.stack([n, e], dim=1).contiguous() + self.pos_emb
        h = self.encoder(x)
        pooled = self.norm(h.mean(dim=1))
        beta = torch.sigmoid(self.gate_head(pooled))
        delta = self.proj_out(pooled)
        blended = (1.0 - beta) * noisy_emb + beta * enhanced_emb
        out = F.normalize(blended + delta, dim=-1)
        if return_beta:
            return out, beta.squeeze(-1)
        return out


# ============================================================================
# 5. NEW: TrainableHead + FrameGateFusion — matches the v11 training script
#    byte-for-byte. The submodule names (`asp`, `asp_bn`, `fc` for head_net;
#    `norm_in`, `proj`, `mix`, `pos`, `emb_proj`, `tok_type`, `encoder`,
#    `out_norm`, `gate`, `delta_norm`, `delta` for the gate) MUST match the
#    checkpoint keys or load_state_dict will silently drop them.
# ============================================================================
class TrainableHead(nn.Module):
    """Trainable copy of ECAPA's pooling head (ASP -> BN -> FC), initialised
    from the frozen original. Must be built from an `EcapaFrames` instance."""

    def __init__(self, ecf):
        super().__init__()
        import copy

        self.asp = copy.deepcopy(ecf.em.asp)
        self.asp_bn = copy.deepcopy(ecf.em.asp_bn)
        self.fc = copy.deepcopy(ecf.em.fc)
        for p in self.parameters():
            p.requires_grad_(True)
        super().train(False)

    def train(self, mode: bool = True):  # BatchNorm stays in inference stats
        return super().train(False)

    def forward(self, h: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        x = self.fc(self.asp_bn(self.asp(h, lengths=wl))).transpose(1, 2)
        return F.normalize(x.squeeze(1).float(), dim=-1)


class FrameGateFusion(nn.Module):
    """Frame-level self-attention gate + embedding residual (v11 architecture).

    fused_frames[t] = (1 - g[t]) * hn[t] + g[t] * he[t]     (per-frame, per-group)
    final_emb       = L2_norm( head_net(fused_frames) + delta )

    `head_net` is attached *after* construction (see `load_fusion_model`), because
    it needs the production ECAPA instance to deep-copy.
    """

    def __init__(
        self,
        c_in: int = 3072,
        d: int = 192,
        n_heads: int = 4,
        n_layers: int = 2,
        pool: int = 4,
        dropout: float = 0.1,
        n_groups: int = 1,
        emb_dim: int = 192,
        use_delta: bool = True,
    ):
        super().__init__()
        assert c_in % n_groups == 0
        self.c_in, self.pool, self.G, self.use_delta, self.emb_dim = (
            c_in,
            pool,
            n_groups,
            use_delta,
            emb_dim,
        )
        self.norm_in = nn.LayerNorm(c_in)
        self.proj = nn.Linear(c_in, d)
        self.mix = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.pos = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.emb_proj = nn.Linear(emb_dim, d)
        self.tok_type = nn.Parameter(torch.zeros(2, d))
        layer = nn.TransformerEncoderLayer(
            d,
            n_heads,
            4 * d,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, n_groups)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.delta_norm = nn.LayerNorm(d)
        self.delta = nn.Linear(d, emb_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, hn, he, en, ee, wl):
        """hn, he: (B, C, T); en, ee: (B, emb_dim); wl: (B,) relative lengths.
        Returns (fused_frames (B, C, T), gate (B, G, T), delta (B, emb_dim))."""
        B, C, T = hn.shape
        fvalid = torch.arange(T, device=hn.device)[None] < wl[:, None] * T

        zn = self.proj(self.norm_in(hn.transpose(1, 2)))
        ze = self.proj(self.norm_in(he.transpose(1, 2)))
        x = self.mix(torch.cat([zn, ze, zn - ze], dim=-1))
        x = (x * fvalid.unsqueeze(-1)).transpose(1, 2)
        x = F.avg_pool1d(x, self.pool, self.pool, ceil_mode=True)
        Tp = x.shape[-1]
        tvalid = torch.arange(Tp, device=x.device)[None] < torch.ceil(
            wl[:, None] * T / self.pool
        ).clamp(min=1)
        x = x * tvalid.unsqueeze(1)
        x = (x + self.pos(x)).transpose(1, 2)

        et = torch.stack([self.emb_proj(en), self.emb_proj(ee)], dim=1) + self.tok_type
        pad = torch.cat([~tvalid, torch.zeros(B, 2, dtype=torch.bool, device=x.device)], dim=1)
        h = self.encoder(torch.cat([x, et], dim=1), src_key_padding_mask=pad)

        g = torch.sigmoid(self.gate(self.out_norm(h[:, :Tp])))
        g = F.interpolate(g.transpose(1, 2), size=T, mode="linear", align_corners=False)
        gc = g if self.G == 1 else g.repeat_interleave(C // self.G, dim=1)

        if self.use_delta:
            delta = self.delta(self.delta_norm(h[:, Tp:].mean(dim=1)))
        else:
            delta = torch.zeros(B, self.emb_dim, device=hn.device, dtype=hn.dtype)
        return hn + gc * (he - hn), g, delta


# ============================================================================
# Registry
# ============================================================================
_MODEL_REGISTRY = {
    "mlp": MLP,
    "cross_attention": CrossAttentionFusion,
    "noise_aware": NoiseAwareFusion,
    "self_attention": SelfAttentionFusion,
    "frame_gate": FrameGateFusion,  # v11
}


# ============================================================================
# Loader
# ============================================================================
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("fusion_state_dict", "gate_state", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # bare state dict
        return ckpt
    return ckpt


def load_fusion_model(
    checkpoint_path: str | Path,
    model_type: str = "frame_gate",
    device: str = DEVICE,
    *,
    ecapa_frames=None,  # REQUIRED for "frame_gate" — an EcapaFrames instance
    c_in: int | None = None,
    **kwargs,
) -> nn.Module:
    """Load a fusion checkpoint into the corresponding model class.

    For model_type="frame_gate" (v11), you MUST pass `ecapa_frames`, and `c_in`
    is auto-detected from the checkpoint if not provided.
    """
    model_cls = _MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown model_type: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # ---- FrameGateFusion: needs head_net attached before load ----
    if model_type == "frame_gate":
        if ecapa_frames is None:
            raise ValueError(
                "load_fusion_model(model_type='frame_gate') requires ecapa_frames=EcapaFrames(...)"
            )
        if c_in is None:
            # infer c_in from the first Linear layer inside `norm_in` or `proj`
            for key in ("norm_in.weight", "proj.weight"):
                if key in state:
                    c_in = int(state[key].shape[-1] if "norm_in" in key else state[key].shape[1])
                    break
            if c_in is None:
                c_in = 3072  # ECAPA-TDNN default frame width

        model = FrameGateFusion(c_in=c_in, **kwargs)
        model.head_net = TrainableHead(ecapa_frames)  # MUST be attached pre-load

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"FrameGateFusion missing {len(missing)} keys on load: {missing[:8]}"
            )
        if unexpected:
            print(
                f"[load_fusion_model] warning: {len(unexpected)} unexpected keys "
                f"(first 8: {unexpected[:8]})"
            )

        model.eval()
        model.to(device)
        model.head_net.eval()
        model.head_net.to(device)
        return model

    # ---- All other model types: original path ----
    model = model_cls(**kwargs)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
