"""Replay CNN training entrypoint (placeholder — no training yet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_replay_detector(config: dict[str, Any], output_dir: Path) -> Path:
    """
    Train the replay CNN.

    Not implemented in the scaffold. Member 2 should implement the full loop
    (data loading, optimization, checkpointing, threshold tuning).

    Args:
        config: Loaded YAML configuration.
        output_dir: Directory for checkpoints and logs.

    Raises:
        NotImplementedError: Always in the scaffold.
    """
    output_dir = Path(output_dir)
    raise NotImplementedError(
        "Replay detector training is not implemented yet. "
        f"Config keys={list(config.keys())}, output_dir={output_dir}"
    )
