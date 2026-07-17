"""Log-Mel CNN skeleton for replay detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReplayCNN(nn.Module):
    """
    Small CNN over Log-Mel spectrograms.

    Configurable Mel bins, channels and dropout. Outputs a single logit
    (use sigmoid for replay probability).
    """

    def __init__(
        self,
        n_mels: int = 64,
        channels: tuple[int, ...] = (16, 32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout2d(p=dropout),
                ]
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.n_mels = n_mels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-Mel batch of shape ``[B, 1, n_mels, T]``.

        Returns:
            Logits of shape ``[B]``.
        """
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return replay probabilities in ``[0, 1]``, shape ``[B]``."""
        return torch.sigmoid(self.forward(x))
