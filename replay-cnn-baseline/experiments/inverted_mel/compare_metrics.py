"""Compare Mel vs inverted-Mel metric JSON files side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-metrics", type=Path, required=True)
    parser.add_argument("--inverted-mel-metrics", type=Path, required=True)
    args = parser.parse_args()

    mel = load_metrics(args.mel_metrics)
    imel = load_metrics(args.inverted_mel_metrics)

    keys = [
        "feature_type",
        "split",
        "eer_percent",
        "f1_replay",
        "accuracy",
        "precision_replay",
        "recall_replay",
        "threshold",
    ]
    print(f"{'metric':<22} {'mel':>12} {'inverted_mel':>14}")
    print("-" * 50)
    for key in keys:
        left = mel.get(key, "—")
        right = imel.get(key, "—")
        if isinstance(left, float):
            left = f"{left:.4f}"
        if isinstance(right, float):
            right = f"{right:.4f}"
        print(f"{key:<22} {str(left):>12} {str(right):>14}")


if __name__ == "__main__":
    main()
