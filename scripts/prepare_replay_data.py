#!/usr/bin/env python3
"""Prepare ASVspoof replay-detection data (placeholder — no downloads)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare replay detection data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/replay_detection.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/asvspoof2017"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[placeholder] prepare_replay_data.py\n"
        f"  config={args.config}\n"
        f"  output_dir={args.output_dir}\n"
        "Member 2: parse ASVspoof 2017 V2 protocols and build train/dev/eval lists."
    )


if __name__ == "__main__":
    main()
