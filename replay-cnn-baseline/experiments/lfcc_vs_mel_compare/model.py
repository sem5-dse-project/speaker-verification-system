"""Lightweight replay CNN used for Mel / inverted-Mel / LFCC comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from features import build_spectrogram_front_end


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    seconds: float = 4.0
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    n_lfcc: int = 60
    n_filters: int = 128
    feature_type: str = "lfcc"

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.seconds)


def fix_length(waveform: torch.Tensor, length: int, random_crop: bool) -> torch.Tensor:
    if waveform.numel() == 0:
        raise ValueError("Empty audio")
    if waveform.numel() < length:
        repeats = int(np.ceil(length / waveform.numel()))
        waveform = waveform.repeat(repeats)
    if waveform.numel() == length:
        return waveform
    max_start = waveform.numel() - length
    if random_crop:
        start = int(torch.randint(0, max_start + 1, (1,)).item())
    else:
        start = max_start // 2
    return waveform[start : start + length]


class ReplayCNN(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.front_end = build_spectrogram_front_end(
            feature_type=config.feature_type,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            n_lfcc=config.n_lfcc,
            n_filters=config.n_filters,
        )
        self.features = nn.Sequential(
            self.block(1, 16),
            self.block(16, 32),
            self.block(32, 64),
            self.block(64, 96),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 1),
        )

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.front_end(waveform)
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        spec = ((spec - mean) / std).unsqueeze(1)
        return self.classifier(self.features(spec)).squeeze(1)


def config_to_dict(config: AudioConfig) -> dict:
    return asdict(config)
