"""CLI: evaluate WavLM LA checkpoint on ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_lib import eval_wavlm_la
from experiment_lib import CACHE_DIR, DEFAULT_CKPT, DEFAULT_LA, RUNS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-root", type=Path, default=DEFAULT_LA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-utts", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = args.output or (RUNS_DIR / f"eval_wavlm_la_{args.split}")
    eval_wavlm_la(
        checkpoint=args.checkpoint,
        la_root=args.la_root,
        output_dir=output,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        max_utts=args.max_utts,
        force_cpu=args.cpu,
        refresh_cache=args.refresh_cache,
    )
