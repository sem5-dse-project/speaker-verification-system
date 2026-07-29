"""Zero-shot: PA2019-trained inverted-Mel CNN → ASVspoof 2017.

Loads the checkpoint from inverted_mel_pa2019_train and scores ASVspoof 2017
dev (or eval) under replay-cnn-baseline/data.

New files only — does not modify PA training or the 2017 inverted_mel scripts.
"""

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
_PA_TRAIN_DIR = _THIS_DIR.parent / "inverted_mel_pa2019_train"
_DEFAULT_DATA_ROOT = _THIS_DIR.parents[1] / "data"
_DEFAULT_CKPT = (
    _PA_TRAIN_DIR / "runs" / "inverted_mel_pa2019" / "best_inverted_mel_pa2019.pt"
)

sys.path.insert(0, str(_INVERTED_MEL_DIR))
from inverted_mel_cnn import (  # noqa: E402
    AudioConfig,
    ReplayCNN,
    ReplayDataset,
    limit_records,
    read_protocol,
    resolve_dataset_root,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def metrics_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
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
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring ASVspoof2017"):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train PA2019 first, or pass --checkpoint to your .pt file."
        )

    model, ckpt_thr, config, ckpt = load_checkpoint(ckpt_path, device)
    print(
        f"Loaded {ckpt_path}\n"
        f"feature={config.feature_type}; device={device}; "
        f"ckpt_threshold={ckpt_thr:.6f}; train_epoch={ckpt.get('epoch')}; "
        f"dataset={ckpt.get('dataset', 'unknown')}"
    )

    dataset_root = resolve_dataset_root(Path(args.data_root))
    records = read_protocol(dataset_root, args.split)
    records = limit_records(records, args.max_files, args.seed)
    print(
        f"ASVspoof2017 {args.split}: {len(records)} files "
        f"(genuine={sum(r.label == 0 for r in records)}, "
        f"spoof={sum(r.label == 1 for r in records)})"
    )

    dataset = ReplayDataset(
        dataset_root, args.split, records, config, training=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels, scores, file_ids = collect_scores(model, loader, device)

    transfer = metrics_at_threshold(labels, scores, ckpt_thr)
    eer, eer_thr = calculate_eer(labels, scores)
    oracle = metrics_at_threshold(labels, scores, eer_thr)

    result = {
        "experiment": "pa2019_inverted_mel_zero_shot_on_asvspoof2017",
        "train_dataset": ckpt.get("dataset", "ASVspoof2019_PA"),
        "eval_dataset": "ASVspoof2017_V2",
        "split": args.split,
        "feature_type": config.feature_type,
        "checkpoint": str(ckpt_path.resolve()),
        "num_scored_files": int(len(labels)),
        "num_genuine": int((labels == 0).sum()),
        "num_spoof": int((labels == 1).sum()),
        "metrics_at_pa_train_val_threshold": transfer,
        "metrics_at_oracle_eer_threshold": oracle,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"asvspoof2017_{args.split}"
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_id",
                "true_label",
                "replay_probability",
                "pred_pa_thr",
                "pred_eer_thr",
            ]
        )
        pred_t = (scores >= ckpt_thr).astype(int)
        pred_e = (scores >= eer_thr).astype(int)
        for file_id, lab, score, pt, pe in zip(
            file_ids, labels, scores, pred_t, pred_e
        ):
            writer.writerow(
                [
                    file_id,
                    "spoof" if lab else "genuine",
                    score,
                    "spoof" if pt else "genuine",
                    "spoof" if pe else "genuine",
                ]
            )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "runs" / "eval_asvspoof2017_dev",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run_eval(build_parser().parse_args())
