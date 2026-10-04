"""Score the mixed inverted-Mel checkpoint on an ASVspoof 2019 LA split.

Threshold stays at the value stored in the checkpoint. The reported equal
error rate is the sweep on this split, which does not move that threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve().parent
_LA_DIR = _THIS.parent / "inverted_mel_mixed_on_la2019"
_IMEL_DIR = _THIS.parent / "inverted_mel"
sys.path.insert(0, str(_LA_DIR))
sys.path.insert(0, str(_IMEL_DIR))
sys.path.insert(0, str(_THIS))

from common import la_root, paper_runs  # noqa: E402
from inverted_mel_cnn import AudioConfig, ReplayCNN, calculate_eer, fix_length  # noqa: E402
from la_data import (  # noqa: E402
    LAWaveformDataset,
    filter_readable_records,
    read_la_cm_protocol,
    PROTOCOL_BY_SPLIT,
)


def _lfcc_network():
    """Load the comparison CNN without reusing the Mel-only ``features`` module."""
    import importlib

    compare_dir = _THIS.parent / "lfcc_vs_mel_compare"
    sys.path.insert(0, str(compare_dir))
    saved = {
        name: sys.modules.pop(name)
        for name in ("features", "model")
        if name in sys.modules
    }
    try:
        network = importlib.import_module("model")
        return network.AudioConfig, network.ReplayCNN
    finally:
        sys.modules.pop("model", None)
        sys.modules.pop("features", None)
        sys.modules.update(saved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="eval")
    parser.add_argument("--la-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.la_root) if args.la_root else la_root()
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        _THIS.parent
        / "inverted_mel_mixed_2017_pa2019"
        / "runs"
        / "inverted_mel_mixed"
        / "best_inverted_mel_mixed_2017_pa2019.pt"
    )
    output = Path(args.output) if args.output else paper_runs() / f"mixed_imel_on_la_{args.split}"
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg:
        cfg["feature_type"] = ckpt.get("feature_type", "inverted_mel")
    feature_key = str(cfg.get("feature_type", "")).strip().lower().replace("-", "_")
    config_cls, network_cls = AudioConfig, ReplayCNN
    if feature_key in {"lfcc", "log_lfcc", "linear_fcc"}:
        config_cls, network_cls = _lfcc_network()
    allowed = set(config_cls.__dataclass_fields__)
    config = config_cls(**{k: v for k, v in cfg.items() if k in allowed})
    model = network_cls(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    train_thr = float(ckpt["threshold"])
    print(f"feature={config.feature_type} device={device} train_thr={train_thr:.4f}")

    protocol = root / PROTOCOL_BY_SPLIT[args.split]
    records = read_la_cm_protocol(protocol)
    print(f"LA {args.split} protocol: {len(records)}")
    cache = _LA_DIR / "cache" / f"la2019_{args.split}_readable.json"
    readable, skipped = filter_readable_records(
        root,
        args.split,
        records,
        cache_path=cache,
        force_refresh=args.refresh_cache,
    )
    print(f"Readable {len(readable)}  skipped {len(skipped)}")
    if skipped:
        (output / "skipped_utts.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")

    dataset = LAWaveformDataset(
        root,
        args.split,
        readable,
        config.sample_rate,
        config.samples,
        fix_length_fn=fix_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels, scores = [], []
    with torch.inference_mode():
        for waves, labs, _ids in loader:
            logits = model(waves.to(device, non_blocking=True))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(labs.tolist())
    labels_np = np.asarray(labels, dtype=int)
    scores_np = np.asarray(scores, dtype=float)
    eer, eer_thr = calculate_eer(labels_np, scores_np)
    result = {
        "experiment": "mixed_inverted_mel_on_la",
        "split": args.split,
        "checkpoint": str(checkpoint.resolve()),
        "num_protocol_files": len(records),
        "num_scored_files": int(len(labels_np)),
        "num_skipped_corrupt": len(skipped),
        "threshold_from_checkpoint": train_thr,
        "eer_percent": float(eer * 100.0),
        "eer_threshold_on_this_split": float(eer_thr),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if len(records) != len(labels_np):
        print(
            f"WARNING: scored {len(labels_np)} of {len(records)} protocol files."
        )


if __name__ == "__main__":
    main()
