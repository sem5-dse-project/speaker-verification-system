#!/usr/bin/env python3
"""Train the Quality-Conditioned Fusion Gate (placeholder)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion gate")
    parser.add_argument("--config", type=Path, default=Path("configs/quality_fusion.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/fusion"))
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("PyYAML is required to load configs.", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from voice_auth.quality_fusion.train import train_fusion_gate

    try:
        train_fusion_gate(config, args.output_dir)
    except NotImplementedError as exc:
        print(f"[scaffold] {exc}")
        sys.exit(0)


if __name__ == "__main__":
    main()
