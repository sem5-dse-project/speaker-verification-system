"""Replay detector inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Waveform
from voice_auth.replay_detection.features import extract_logmel
from voice_auth.replay_detection.model import ReplayCNN


class ReplayDetectorInference:
    """
    Implements the :class:`~voice_auth.common.interfaces.ReplayDetector` protocol.
    """

    def __init__(
        self,
        model: ReplayCNN | None = None,
        threshold: float = 0.5,
        n_mels: int = 64,
        duration_sec: float = 3.0,
        device: str = "cpu",
        weights_path: Path | None = None,
    ) -> None:
        self.threshold = threshold
        self.n_mels = n_mels
        self.duration_sec = duration_sec
        self.device = torch.device(device)
        self.model = model if model is not None else ReplayCNN(n_mels=n_mels)
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(Path(weights_path), map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_replay_probability(
        self,
        waveform: Waveform,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> float:
        """Return replay probability in ``[0, 1]``."""
        feats = extract_logmel(
            waveform,
            sample_rate=sample_rate,
            n_mels=self.n_mels,
            duration_sec=self.duration_sec,
        )
        tensor = torch.from_numpy(feats).unsqueeze(0).to(self.device)  # [1, 1, M, T]
        with torch.no_grad():
            prob = float(self.model.predict_proba(tensor).item())
        return float(np.clip(prob, 0.0, 1.0))

    def is_replay(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
        """Reject if replay probability >= threshold."""
        return self.predict_replay_probability(waveform, sample_rate) >= self.threshold
