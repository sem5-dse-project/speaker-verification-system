"""Fusion-training dataset placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FusionExample:
    """One precomputed fusion-training example."""

    utterance_id: str
    original_embedding: np.ndarray
    enhanced_embedding: np.ndarray
    quality_vector: np.ndarray
    snr_db: float | None = None


class FusionFeatureDataset:
    """
    Container for precomputed embeddings and quality vectors.

    Member 3 should extend this to load ``.npz`` shards produced by
    ``scripts/precompute_fusion_features.py``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else None
        self.examples: list[FusionExample] = []

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, example: FusionExample) -> None:
        """Append a single example."""
        self.examples.append(example)

    def load_numpy_dir(self, directory: Path) -> int:
        """
        Load examples from a directory of ``.npz`` files.

        Expected keys: ``utt_id``, ``e_orig``, ``e_enh``, ``quality``, optional ``snr_db``.
        Returns the number of loaded examples.
        """
        directory = Path(directory)
        if not directory.exists():
            return 0
        count = 0
        for path in sorted(directory.glob("*.npz")):
            data = np.load(path, allow_pickle=True)
            self.examples.append(
                FusionExample(
                    utterance_id=str(data["utt_id"]),
                    original_embedding=np.asarray(data["e_orig"], dtype=np.float32),
                    enhanced_embedding=np.asarray(data["e_enh"], dtype=np.float32),
                    quality_vector=np.asarray(data["quality"], dtype=np.float32),
                    snr_db=float(data["snr_db"]) if "snr_db" in data.files else None,
                )
            )
            count += 1
        return count
