"""Fusion-gate training entrypoint (placeholder — no training yet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_fusion_gate(config: dict[str, Any], output_dir: Path) -> Path:
    """
    Train the Quality-Conditioned Fusion Gate.

    Not implemented in the scaffold. Member 3 should implement loss design,
    SNR-conditioned batches, and checkpoint export.

    Raises:
        NotImplementedError: Always in the scaffold.
    """
    output_dir = Path(output_dir)
    raise NotImplementedError(
        "Fusion gate training is not implemented yet. "
        f"Config keys={list(config.keys())}, output_dir={output_dir}"
    )
