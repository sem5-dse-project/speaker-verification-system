#!/usr/bin/env python3
"""Precompute embeddings and quality vectors for fusion training (placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute fusion features")
    parser.add_argument("--config", type=Path, default=Path("configs/quality_fusion.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fusion_features"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[placeholder] precompute_fusion_features.py\n"
        f"  config={args.config}\n"
        f"  output_dir={args.output_dir}\n"
        "Member 3: run enhancement + ECAPA encode + quality features; "
        "save .npz shards (do not commit them)."
    )


if __name__ == "__main__":
    main()
