"""Evaluate a PA2019-trained inverted-Mel checkpoint on ASVspoof 2019 PA dev."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "PA").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_INVERTED_MEL_DIR))
sys.path.insert(0, str(_THIS_DIR))

from inverted_mel_cnn import AudioConfig, ReplayCNN, fix_length  # noqa: E402
from pa_data import (  # noqa: E402
    PROTOCOL_BY_SPLIT,
    PA2019WaveformDataset,
    filter_readable_records,
    limit_records_stratified,
    read_pa_cm_protocol,
)

_DEFAULT_PA_ROOT = _REPO_ROOT / "data" / "PA"
_DEFAULT_CKPT = _THIS_DIR / "runs" / "inverted_mel_pa2019" / "best_inverted_mel_pa2019.pt"


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    live = scores[labels == 0]
    spoof = scores[labels == 1]
    if live.size == 0 or spoof.size == 0:
        raise ValueError("EER requires both classes")
    if float(live.max()) < float(spoof.min()):
        thr = (float(live.max()) + float(spoof.min())) / 2.0
        return 0.0, thr
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    miss = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - miss)))
    return float((fpr[idx] + miss[idx]) / 2.0), float(thresholds[idx])


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    eer, eer_thr = calculate_eer(labels, scores)
    return {
        "threshold": float(threshold),
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold_on_this_split": eer_thr,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_replay": float(precision_score(labels, preds, zero_division=0)),
        "recall_replay": float(recall_score(labels, preds, zero_division=0)),
        "f1_replay": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix_live_replay": confusion_matrix(
            labels, preds, labels=[0, 1]
        ).tolist(),
    }


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg and "feature_type" in ckpt:
        cfg["feature_type"] = ckpt["feature_type"]
    config = AudioConfig(**cfg)
    model = ReplayCNN(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, float(ckpt["threshold"]), config, ckpt


@torch.inference_mode()
def collect_scores(model, loader, device):
    model.eval()
    labels, scores, ids = [], [], []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring PA dev"):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pa_root = Path(args.pa_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_thr, config, ckpt = load_checkpoint(Path(args.checkpoint), device)
    print(
        f"Loaded {args.checkpoint}\n"
        f"feature={config.feature_type}; device={device}; "
        f"ckpt_threshold={ckpt_thr:.6f}; train_epoch={ckpt.get('epoch')}"
    )

    protocol = pa_root / PROTOCOL_BY_SPLIT[args.split]
    records = read_pa_cm_protocol(protocol)
    random.seed(args.seed)
    random.shuffle(records)
    records, skipped = filter_readable_records(
        pa_root,
        args.split,
        records,
        cache_path=cache_dir / f"pa2019_{args.split}_readable.json",
        force_refresh=args.refresh_cache,
        target_count=args.max_files if args.max_files > 0 else 0,
    )
    print(
        f"PA {args.split}: readable_pool={len(records)}, skipped_corrupt={len(skipped)}"
    )
    if skipped:
        (output_dir / f"{args.split}_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    records = limit_records_stratified(records, args.max_files, args.seed)
    print(f"PA {args.split}: scoring {len(records)} files")
    dataset = PA2019WaveformDataset(
        pa_root,
        args.split,
        records,
        config.sample_rate,
        config.samples,
        training=False,
        fix_length_fn=fix_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels, scores, utt_ids = collect_scores(model, loader, device)

    transfer = metrics_at_threshold(labels, scores, ckpt_thr)
    eer, eer_thr = calculate_eer(labels, scores)
    oracle = metrics_at_threshold(labels, scores, eer_thr)

    result = {
        "experiment": "inverted_mel_trained_on_pa2019",
        "dataset": "ASVspoof2019_PA",
        "split": args.split,
        "feature_type": config.feature_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_scored_files": int(len(labels)),
        "num_skipped_corrupt": len(skipped),
        "num_bonafide": int((labels == 0).sum()),
        "num_spoof": int((labels == 1).sum()),
        "metrics_at_train_val_threshold": transfer,
        "metrics_at_oracle_eer_threshold": oracle,
    }
    prefix = f"pa2019_{args.split}"
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["utt_id", "true_label", "replay_probability", "pred_ckpt_thr", "pred_eer_thr"]
        )
        pred_t = (scores >= ckpt_thr).astype(int)
        pred_e = (scores >= eer_thr).astype(int)
        for utt_id, lab, score, pt, pe in zip(utt_ids, labels, scores, pred_t, pred_e):
            writer.writerow(
                [
                    utt_id,
                    "spoof" if lab else "bonafide",
                    score,
                    "spoof" if pt else "bonafide",
                    "spoof" if pe else "bonafide",
                ]
            )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pa-root", type=Path, default=_DEFAULT_PA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--split", choices=["dev", "eval", "train"], default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "runs" / "eval_pa2019_dev",
    )
    parser.add_argument("--cache-dir", type=Path, default=_THIS_DIR / "cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


if __name__ == "__main__":
    run_eval(build_parser().parse_args())
