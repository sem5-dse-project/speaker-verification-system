#!/usr/bin/env python3
"""Train the replay CNN (placeholder — training not implemented)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train replay detector")
    parser.add_argument("--config", type=Path, default=Path("configs/replay_detection.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/replay"))
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("PyYAML is required to load configs.", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from voice_auth.replay_detection.train import train_replay_detector

    try:
        train_replay_detector(config, args.output_dir)
    except NotImplementedError as exc:
        print(f"[scaffold] {exc}")
        sys.exit(0)


if __name__ == "__main__":
    main()
