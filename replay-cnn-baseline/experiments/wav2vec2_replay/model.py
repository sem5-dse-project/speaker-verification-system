"""Frozen Wav2Vec2 encoder + trainable MLP head for replay detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_MODEL_ID = "facebook/wav2vec2-base-960h"


@dataclass
class Wav2Vec2AudioConfig:
    sample_rate: int = 16000
    seconds: float = 4.0

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.seconds)


class Wav2Vec2ReplayDetector(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        freeze_encoder: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install transformers: pip install transformers"
            ) from exc

        self.model_id = model_id
        self.freeze_encoder = freeze_encoder
        self.encoder = Wav2Vec2Model.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def trainable_parameters(self):
        if self.freeze_encoder:
            return self.head.parameters()
        return self.parameters()

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
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)

        return self.head(pooled).squeeze(-1)

    def save_checkpoint(
        self,
        path: Path,
        *,
        audio_config: Wav2Vec2AudioConfig,
        threshold: float,
        val_eer: float,
        epoch: int,
        train_summary: dict,
        val_summary: dict,
        extra: dict | None = None,
    ) -> None:
        payload = {
            "experiment": "wav2vec2_mixed_2017_pa2019",
            "model_id": self.model_id,
            "freeze_encoder": self.freeze_encoder,
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
    ) -> tuple["Wav2Vec2ReplayDetector", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            model_id=ckpt.get("model_id", DEFAULT_MODEL_ID),
            freeze_encoder=ckpt.get("freeze_encoder", True),
        )
        model.head.load_state_dict(ckpt["head_state"])
        model.to(device)
        model.eval()
        return model, ckpt
