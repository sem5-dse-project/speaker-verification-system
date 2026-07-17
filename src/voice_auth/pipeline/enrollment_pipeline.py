"""Enrollment pipeline: preprocess → encode → average → store template."""

from __future__ import annotations

from pathlib import Path

from voice_auth.common.interfaces import SpeakerEncoder
from voice_auth.common.types import Waveform
from voice_auth.speaker_verification.enrollment import (
    EnrollmentTemplate,
    create_enrollment_template,
    save_template,
)


def run_enrollment(
    user_id: str,
    waveforms: list[Waveform],
    sample_rates: list[int],
    encoder: SpeakerEncoder,
    output_path: Path | None = None,
    encoder_name: str = "ecapa",
) -> EnrollmentTemplate:
    """
    Create and optionally persist an enrollment template.

    Steps:
        1. Preprocess each enrollment recording (mono, 16 kHz, float32).
        2. Extract speaker embeddings.
        3. L2-normalize and average.
        4. Store the template with the user ID.
    """
    template = create_enrollment_template(
        user_id=user_id,
        waveforms=waveforms,
        sample_rates=sample_rates,
        encoder=encoder,
        encoder_name=encoder_name,
    )
    if output_path is not None:
        save_template(template, Path(output_path))
    return template
