"""CLI: evaluate Wav2Vec2 replay checkpoint on 2017 and PA dev splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_lib import eval_wav2vec2_replay
from experiment_lib import CACHE_DIR, DEFAULT_ASV17, DEFAULT_CKPT, DEFAULT_PA, RUNS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asv17-root", type=Path, default=DEFAULT_ASV17)
    parser.add_argument("--pa-root", type=Path, default=DEFAULT_PA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--corpus",
        choices=["both", "asvspoof2017", "pa2019"],
        default="both",
    )
    parser.add_argument("--asv17-split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument("--pa-split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "eval_wav2vec2_both_dev",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    eval_wav2vec2_replay(
        checkpoint=args.checkpoint,
        asv17_root=args.asv17_root,
        pa_root=args.pa_root,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        corpus=args.corpus,
        asv17_split=args.asv17_split,
        pa_split=args.pa_split,
        batch_size=args.batch_size,
        workers=args.workers,
        force_cpu=args.cpu,
        refresh_cache=args.refresh_cache,
    )
