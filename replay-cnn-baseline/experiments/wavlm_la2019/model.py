"""WavLM-Base encoder + attentive statistics pooling for LA anti-spoofing.

Follows ICACS 2024: “A lightweight end-to-end anti-spoofing voice model based on WavLM”
(https://doi.org/10.1145/3708597.3708621): frozen/light WavLM frontend, ASP, small FC head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
            raise ImportError("Install transformers: pip install transformers") from exc

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

    def trainable_parameters(self):
        if self.freeze_encoder:
            return list(self.pool.parameters()) + list(self.head.parameters())
        return self.parameters()

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

    def save_checkpoint(
        self,
        path: Path,
        *,
        audio_config: WavLMAudioConfig,
        threshold: float,
        val_eer: float,
        epoch: int,
        train_summary: dict,
        val_summary: dict,
        extra: dict | None = None,
    ) -> None:
        payload = {
            "experiment": "wavlm_la2019",
            "paper": "10.1145/3708597.3708621",
            "model_id": self.model_id,
            "freeze_encoder": self.freeze_encoder,
            "pool_state": self.pool.state_dict(),
            "head_state": self.head.state_dict(),
            "audio_config": asdict(audio_config),
            "threshold": threshold,
            "val_eer": val_eer,
            "epoch": epoch,
            "train_summary": train_summary,
            "val_summary": val_summary,
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

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
