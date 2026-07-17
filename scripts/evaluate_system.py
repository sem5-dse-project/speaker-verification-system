#!/usr/bin/env python3
"""Evaluate the full system / offline experiments (placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate speaker verification system")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[placeholder] evaluate_system.py\n"
        f"  config={args.config}\n"
        f"  output_dir={args.output_dir}\n"
        "Offline experiments to implement:\n"
        "  1) x-vector vs ECAPA-TDNN (EER, FAR, FRR, latency)\n"
        "  2) original / enhanced / 50-50 / quality-conditioned fusion\n"
        "  3) clean, 20, 10, 5, 0 dB conditions\n"
        "  4) replay detection (EER, precision, recall, F1, confusion matrix)"
    )


if __name__ == "__main__":
    main()
