"""Mel, inverted-Mel, and LFCC front-ends for lightweight replay CNN.

LFCC = Linear Frequency Cepstral Coefficients (even Hz spacing).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio


def hz_to_mel(freq: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(freq, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    return 2595.0 * math.log10(1.0 + float(freq) / 700.0)


def mel_to_hz(mel: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(mel, torch.Tensor):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    return 700.0 * (10.0 ** (float(mel) / 2595.0) - 1.0)


def hz_to_inverted_mel(freq: torch.Tensor, f_max: float) -> torch.Tensor:
    return hz_to_mel(torch.tensor(f_max, dtype=freq.dtype, device=freq.device)) - hz_to_mel(
        f_max - freq
    )


def inverted_mel_to_hz(mel_inv: torch.Tensor, f_max: float) -> torch.Tensor:
    mel_fmax = hz_to_mel(torch.tensor(f_max, dtype=mel_inv.dtype, device=mel_inv.device))
    return f_max - mel_to_hz(mel_fmax - mel_inv)


def create_inverted_mel_filterbank(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    if f_max is None:
        f_max = sample_rate / 2.0
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_min = float(hz_to_inverted_mel(torch.tensor(f_min), f_max))
    mel_max = float(hz_to_inverted_mel(torch.tensor(f_max), f_max))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = inverted_mel_to_hz(mel_points, f_max)
    bins = torch.floor((n_freqs - 1) * hz_points / (sample_rate / 2.0)).long()
    bins = torch.clamp(bins, 0, n_freqs - 1)
    fbanks = torch.zeros(n_mels, n_freqs, dtype=torch.float32)
    for i in range(n_mels):
        left = int(bins[i].item())
        center = int(bins[i + 1].item())
        right = int(bins[i + 2].item())
        if center == left:
            center = min(left + 1, n_freqs - 1)
        if right == center:
            right = min(center + 1, n_freqs - 1)
        for j in range(left, center):
            fbanks[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fbanks[i, j] = (right - j) / max(right - center, 1)
    row_sums = fbanks.sum(dim=1, keepdim=True).clamp_min(1e-10)
    return fbanks / row_sums


def create_linear_filterbank(
    n_freqs: int,
    n_filters: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Triangular filterbank with linearly spaced center frequencies."""
    if f_max is None:
        f_max = sample_rate / 2.0
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    hz_points = torch.linspace(f_min, f_max, n_filters + 2)
    bins = torch.floor((n_freqs - 1) * hz_points / (sample_rate / 2.0)).long()
    bins = torch.clamp(bins, 0, n_freqs - 1)
    fbanks = torch.zeros(n_filters, n_freqs, dtype=torch.float32)
    for i in range(n_filters):
        left = int(bins[i].item())
        center = int(bins[i + 1].item())
        right = int(bins[i + 2].item())
        if center == left:
            center = min(left + 1, n_freqs - 1)
        if right == center:
            right = min(center + 1, n_freqs - 1)
        for j in range(left, center):
            fbanks[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fbanks[i, j] = (right - j) / max(right - center, 1)
    row_sums = fbanks.sum(dim=1, keepdim=True).clamp_min(1e-10)
    return fbanks / row_sums


def create_dct_matrix(n_filters: int, n_lfcc: int) -> torch.Tensor:
    """DCT-II matrix [n_lfcc, n_filters] (ortho-ish scaling)."""
    n = torch.arange(n_filters, dtype=torch.float32)
    k = torch.arange(n_lfcc, dtype=torch.float32).unsqueeze(1)
    dct = torch.cos(math.pi / n_filters * (n + 0.5) * k)
    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / n_filters)
    return dct


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.feature_name = "mel"

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform).clamp_min(1e-6).log()
        return mel.squeeze(0) if squeeze else mel


class InvertedLogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.feature_name = "inverted_mel"
        f_max = sample_rate / 2.0 if f_max is None else f_max
        n_freqs = n_fft // 2 + 1
        fbanks = create_inverted_mel_filterbank(
            n_freqs, n_mels, sample_rate, f_min=f_min, f_max=f_max
        )
        self.register_buffer("fbanks", fbanks)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        fbanks = self.fbanks.to(device=power.device, dtype=power.dtype)
        mel = torch.matmul(fbanks, power).clamp_min(1e-6).log()
        return mel.squeeze(0) if squeeze else mel


class LogLFCC(nn.Module):
    """Log Linear-Frequency Cepstral Coefficients.

    Output shape matches Mel front-ends: ``[batch, n_lfcc, frames]``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_filters: int = 128,
        n_lfcc: int = 60,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc
        self.feature_name = "lfcc"
        f_max = sample_rate / 2.0 if f_max is None else f_max
        n_freqs = n_fft // 2 + 1
        fbanks = create_linear_filterbank(
            n_freqs, n_filters, sample_rate, f_min=f_min, f_max=f_max
        )
        dct = create_dct_matrix(n_filters, n_lfcc)
        self.register_buffer("fbanks", fbanks)
        self.register_buffer("dct", dct)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)  # [B, F, T]
        fbanks = self.fbanks.to(device=power.device, dtype=power.dtype)
        filt = torch.matmul(fbanks, power).clamp_min(1e-6).log()  # [B, n_filters, T]
        dct = self.dct.to(device=filt.device, dtype=filt.dtype)
        lfcc = torch.matmul(dct, filt)  # [B, n_lfcc, T]
        return lfcc.squeeze(0) if squeeze else lfcc


def build_spectrogram_front_end(
    feature_type: str,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    n_lfcc: int = 60,
    n_filters: int = 128,
) -> nn.Module:
    key = feature_type.strip().lower().replace("-", "_")
    if key in {"mel", "log_mel", "logmel"}:
        return LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    if key in {"inverted_mel", "imel", "i_mel", "inverted"}:
        return InvertedLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    if key in {"lfcc", "log_lfcc", "linear_fcc"}:
        return LogLFCC(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_filters=n_filters,
            n_lfcc=n_lfcc,
        )
    raise ValueError(
        f"Unknown feature_type={feature_type!r}. Use 'mel', 'inverted_mel', or 'lfcc'."
    )
