"""Abstract interfaces (Protocols) for pluggable system components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from voice_auth.common.types import Embedding, QualityVector, Waveform


@runtime_checkable
class SpeakerEncoder(Protocol):
    """Extract a fixed-dimensional speaker embedding from a waveform."""

    def encode(self, waveform: Waveform, sample_rate: int) -> Embedding:
        """
        Encode a mono waveform into a speaker embedding.

        Args:
            waveform: Audio tensor of shape ``[1, num_samples]``, float32.
            sample_rate: Sample rate in Hz (expected 16000).

        Returns:
            L2-normalizable embedding of shape ``[embedding_dim]``.
        """
        ...


@runtime_checkable
class ReplayDetector(Protocol):
    """Estimate the probability that audio is a replay attack."""

    def predict_replay_probability(self, waveform: Waveform, sample_rate: int) -> float:
        """
        Predict replay probability in ``[0, 1]``.

        Args:
            waveform: Audio tensor of shape ``[1, num_samples]``, float32.
            sample_rate: Sample rate in Hz (expected 16000).

        Returns:
            Probability that the audio is a replay (1.0 = certain replay).
        """
        ...


@runtime_checkable
class SpeechEnhancer(Protocol):
    """Enhance speech by suppressing background noise."""

    def enhance(self, waveform: Waveform, sample_rate: int) -> Waveform:
        """
        Produce an enhanced waveform.

        Args:
            waveform: Audio tensor of shape ``[1, num_samples]``, float32.
            sample_rate: Sample rate in Hz (expected 16000).

        Returns:
            Enhanced waveform of shape ``[1, num_samples]``, float32.
        """
        ...


@runtime_checkable
class QualityEstimator(Protocol):
    """Extract a low-dimensional audio-quality feature vector."""

    def extract(self, waveform: Waveform, sample_rate: int) -> QualityVector:
        """
        Extract a quality vector (default 5 dimensions).

        Args:
            waveform: Audio tensor of shape ``[1, num_samples]``, float32.
            sample_rate: Sample rate in Hz (expected 16000).

        Returns:
            Quality vector of shape ``[quality_dim]``.
        """
        ...


@runtime_checkable
class EmbeddingFusion(Protocol):
    """Fuse original and enhanced embeddings conditioned on audio quality."""

    def fuse(
        self,
        original_embedding: Embedding,
        enhanced_embedding: Embedding,
        quality_vector: QualityVector,
    ) -> tuple[Embedding, float]:
        """
        Fuse embeddings using a quality-conditioned gate.

        Both input embeddings and the fused result must be L2-normalized.
        Fusion: ``e_fused = alpha * e_original + (1 - alpha) * e_enhanced``.

        Args:
            original_embedding: Embedding from original audio, shape ``[D]``.
            enhanced_embedding: Embedding from enhanced audio, shape ``[D]``.
            quality_vector: Quality features, shape ``[Q]``.

        Returns:
            Tuple of ``(fused_embedding, alpha)`` where alpha is in ``[0, 1]``.
        """
        ...
