"""WavLM-Base + ASP spoof detector for the ML server LA stage.

Architecture matches ``replay-cnn-baseline/experiments/wavlm_la2019``
(ICACS 2024 WavLM anti-spoofing: doi:10.1145/3708597.3708621).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_MODEL_ID = "microsoft/wavlm-base"


@dataclass
class WavLMAudioConfig:
    sample_rate: int = 16000
    seconds: float = 4.0

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.seconds)


class AttentiveStatisticsPooling(nn.Module):
    """Time-attentive mean + std pooling (x-vector ASP)."""

    def __init__(self, hidden_size: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.attention(hidden).squeeze(-1)
        if frame_mask is not None:
            scores = scores.masked_fill(~frame_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (hidden * weights).sum(dim=1)
        centered = hidden - mean.unsqueeze(1)
        var = (centered.pow(2) * weights).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class WavLMSpoofDetector(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        freeze_encoder: bool = True,
        dropout: float = 0.3,
        asp_bottleneck: int = 128,
        hidden_fc: int = 256,
    ) -> None:
        super().__init__()
        try:
            from transformers import WavLMModel
        except ImportError as exc:  # pragma: no cover
            # Transformers 5.x + SpeechBrain on Windows can raise ImportError
            # while resolving optional speechbrain.k2 during inspect(); re-raise
            # that case instead of claiming transformers is missing.
            msg = str(exc).lower()
            if "k2" in msg or "speechbrain" in msg:
                raise
            raise ImportError(
                "Install transformers for WavLM LA: pip install transformers"
            ) from exc

        self.model_id = model_id
        self.freeze_encoder = freeze_encoder
        self.encoder = WavLMModel.from_pretrained(model_id)
        hidden = int(self.encoder.config.hidden_size)

        self.pool = AttentiveStatisticsPooling(hidden, bottleneck=asp_bottleneck)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, hidden_fc // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc // 2, 1),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _frame_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        getter = getattr(self.encoder, "_get_feature_vector_attention_mask", None)
        if getter is None:
            return None
        return getter(hidden.shape[1], attention_mask)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(
                    input_values,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.encoder(
                input_values,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state
        frame_mask = self._frame_mask(hidden, attention_mask)
        pooled = self.pool(hidden, frame_mask)
        return self.head(pooled).squeeze(-1)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        device: torch.device | str = "cpu",
    ) -> tuple["WavLMSpoofDetector", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            model_id=ckpt.get("model_id", DEFAULT_MODEL_ID),
            freeze_encoder=ckpt.get("freeze_encoder", True),
        )
        model.pool.load_state_dict(ckpt["pool_state"])
        model.head.load_state_dict(ckpt["head_state"])
        model.to(device)
        model.eval()
        return model, ckpt
