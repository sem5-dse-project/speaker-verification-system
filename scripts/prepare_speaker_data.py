#!/usr/bin/env python3
"""Prepare speaker-verification data and trial lists (placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare speaker verification data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/speaker_verification.yaml"),
        help="Path to speaker verification YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/speaker"),
        help="Directory for prepared manifests (not committed)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[placeholder] prepare_speaker_data.py\n"
        f"  config={args.config}\n"
        f"  output_dir={args.output_dir}\n"
        "Member 1: implement OpenSLR SLR52 indexing, resampling manifests, "
        "and trial CSV generation."
    )


if __name__ == "__main__":
    main()
