"""Speaker-verification trial list helpers (placeholders)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Trial:
    """A single verification trial."""

    trial_id: str
    enrolled_user_id: str
    test_audio_path: Path
    label: int  # 1 = target, 0 = non-target


def load_trials(protocol_path: Path) -> list[Trial]:
    """
    Load trials from a CSV protocol file.

    Expected columns: ``trial_id,enrolled_user_id,test_audio_path,label``.

    Args:
        protocol_path: Path to the protocol CSV.

    Returns:
        List of :class:`Trial` objects.
    """
    protocol_path = Path(protocol_path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Trial protocol not found: {protocol_path}")

    trials: list[Trial] = []
    with protocol_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trials.append(
                Trial(
                    trial_id=row["trial_id"],
                    enrolled_user_id=row["enrolled_user_id"],
                    test_audio_path=Path(row["test_audio_path"]),
                    label=int(row["label"]),
                )
            )
    return trials


def write_trials(trials: list[Trial], protocol_path: Path) -> None:
    """Write trials to a CSV protocol file."""
    protocol_path = Path(protocol_path)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    with protocol_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_id", "enrolled_user_id", "test_audio_path", "label"],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "trial_id": trial.trial_id,
                    "enrolled_user_id": trial.enrolled_user_id,
                    "test_audio_path": str(trial.test_audio_path),
                    "label": trial.label,
                }
            )
