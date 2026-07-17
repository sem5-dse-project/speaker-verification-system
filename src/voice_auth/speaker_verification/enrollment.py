"""Enrollment template creation and storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from voice_auth.common.audio import l2_normalize, preprocess_audio
from voice_auth.common.interfaces import SpeakerEncoder
from voice_auth.common.types import DEFAULT_SAMPLE_RATE, Embedding, Waveform


@dataclass
class EnrollmentTemplate:
    """Stored speaker enrollment template."""

    user_id: str
    embedding: list[float]
    embedding_dim: int
    num_utterances: int
    encoder_name: str = "unknown"

    def as_array(self) -> Embedding:
        """Return the template embedding as a float32 NumPy array."""
        return np.asarray(self.embedding, dtype=np.float32)


def create_enrollment_template(
    user_id: str,
    waveforms: list[Waveform],
    sample_rates: list[int],
    encoder: SpeakerEncoder,
    encoder_name: str = "unknown",
) -> EnrollmentTemplate:
    """
    Build an enrollment template by averaging L2-normalized embeddings.

    Args:
        user_id: Claimed user identifier.
        waveforms: List of enrollment recordings.
        sample_rates: Sample rate for each recording.
        encoder: Speaker encoder implementing ``encode``.
        encoder_name: Label stored with the template.

    Returns:
        :class:`EnrollmentTemplate` with averaged, L2-normalized embedding.
    """
    if not waveforms:
        raise ValueError("At least one enrollment waveform is required")
    if len(waveforms) != len(sample_rates):
        raise ValueError("waveforms and sample_rates must have the same length")

    embeddings: list[np.ndarray] = []
    for wave, sr in zip(waveforms, sample_rates, strict=True):
        processed, target_sr = preprocess_audio(wave, sr, DEFAULT_SAMPLE_RATE)
        emb = np.asarray(encoder.encode(processed, target_sr), dtype=np.float32)
        embeddings.append(l2_normalize(emb))

    stacked = np.stack(embeddings, axis=0)
    mean_emb = l2_normalize(stacked.mean(axis=0))

    return EnrollmentTemplate(
        user_id=user_id,
        embedding=mean_emb.tolist(),
        embedding_dim=int(mean_emb.shape[0]),
        num_utterances=len(embeddings),
        encoder_name=encoder_name,
    )


def save_template(template: EnrollmentTemplate, path: Path) -> None:
    """Serialize an enrollment template to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(template), indent=2), encoding="utf-8")


def load_template(path: Path) -> EnrollmentTemplate:
    """Load an enrollment template from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EnrollmentTemplate(**data)
