"""CLI: train WavLM-Base + ASP on ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_lib import CACHE_DIR, DEFAULT_LA, RUNS_DIR
from model import DEFAULT_MODEL_ID
from train_lib import train_wavlm_la


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-root", type=Path, default=DEFAULT_LA)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "wavlm_la")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Fine-tune full WavLM (slow; default freezes encoder)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_wavlm_la(
        la_root=args.la_root,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        freeze_encoder=not args.unfreeze_encoder,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seconds=args.seconds,
        validation_fraction=args.validation_fraction,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
        force_cpu=args.cpu,
        refresh_cache=args.refresh_cache,
    )
