"""Pre-trained speaker encoder adapters (placeholders)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_auth.common.audio import ensure_mono, l2_normalize
from voice_auth.common.types import DEFAULT_EMBEDDING_DIM, DEFAULT_SAMPLE_RATE, Embedding, Waveform


class WeightsUnavailableError(RuntimeError):
    """Raised when pre-trained encoder weights are missing."""


class MockSpeakerEncoder:
    """
    Deterministic mock encoder for offline tests without model weights.

    Produces a pseudo-embedding from waveform statistics so shapes and
    pipeline wiring can be validated without downloading checkpoints.
    """

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, seed: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed

    def encode(self, waveform: Waveform, sample_rate: int) -> Embedding:
        """Return a deterministic L2-normalized mock embedding."""
        wave = ensure_mono(waveform).astype(np.float32)
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate}")

        rng = np.random.default_rng(self.seed + int(wave.sum() * 1e3) % 10_000)
        # Mix energy cue with random projection for diversity across utterances
        energy = float(np.sqrt(np.mean(wave**2) + 1e-8))
        emb = rng.standard_normal(self.embedding_dim).astype(np.float32)
        emb = emb * (0.5 + energy)
        return l2_normalize(emb)


class XVectorEncoder:
    """
    Adapter for a pre-trained x-vector model.

    Weights are not bundled. Provide ``weights_path`` pointing to a local
    checkpoint, or use :class:`MockSpeakerEncoder` for development.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.embedding_dim = embedding_dim
        self._model = None

        if self.weights_path is None or not self.weights_path.exists():
            raise WeightsUnavailableError(
                "x-vector weights are unavailable. "
                "Place a checkpoint on disk and pass weights_path, "
                "or use MockSpeakerEncoder for scaffolding/tests. "
                f"Looked for: {self.weights_path}"
            )
        # Member 1: load SpeechBrain / sidekit / custom x-vector here.
        raise WeightsUnavailableError(
            f"x-vector weight loading is not implemented yet: {self.weights_path}"
        )

    def encode(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Embedding:
        """Extract an x-vector embedding (not yet implemented)."""
        raise WeightsUnavailableError("XVectorEncoder.encode requires loaded weights.")


class ECAPAEncoder:
    """
    Adapter for a pre-trained ECAPA-TDNN model.

    Weights are not bundled. Provide ``weights_path`` pointing to a local
    checkpoint, or use :class:`MockSpeakerEncoder` for development.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.embedding_dim = embedding_dim
        self._model = None

        if self.weights_path is None or not self.weights_path.exists():
            raise WeightsUnavailableError(
                "ECAPA-TDNN weights are unavailable. "
                "Place a checkpoint on disk and pass weights_path, "
                "or use MockSpeakerEncoder for scaffolding/tests. "
                f"Looked for: {self.weights_path}"
            )
        # Member 1: load SpeechBrain ECAPA-TDNN here.
        raise WeightsUnavailableError(
            f"ECAPA-TDNN weight loading is not implemented yet: {self.weights_path}"
        )

    def encode(self, waveform: Waveform, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Embedding:
        """Extract an ECAPA-TDNN embedding (not yet implemented)."""
        raise WeightsUnavailableError("ECAPAEncoder.encode requires loaded weights.")


def build_encoder(
    name: str,
    weights_path: Path | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    allow_mock: bool = True,
) -> MockSpeakerEncoder | XVectorEncoder | ECAPAEncoder:
    """
    Factory for speaker encoders.

    Args:
        name: ``"mock"``, ``"xvector"``, or ``"ecapa"``.
        weights_path: Optional local checkpoint path.
        embedding_dim: Embedding dimensionality.
        allow_mock: If True, fall back to mock when weights are missing.
    """
    key = name.strip().lower()
    if key == "mock":
        return MockSpeakerEncoder(embedding_dim=embedding_dim)

    try:
        if key in {"xvector", "x-vector"}:
            return XVectorEncoder(weights_path=weights_path, embedding_dim=embedding_dim)
        if key in {"ecapa", "ecapa-tdnn"}:
            return ECAPAEncoder(weights_path=weights_path, embedding_dim=embedding_dim)
    except WeightsUnavailableError:
        if allow_mock:
            return MockSpeakerEncoder(embedding_dim=embedding_dim)
        raise

    raise ValueError(f"Unknown encoder name: {name!r}. Use mock, xvector, or ecapa.")
