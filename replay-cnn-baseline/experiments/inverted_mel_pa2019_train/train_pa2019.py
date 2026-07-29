"""Train inverted-Mel CNN from scratch on ASVspoof 2019 PA train.

- Train on readable PA train files
- Early-stop / threshold on speaker-disjoint PA train validation
- Final evaluation is done separately with eval_pa2019.py on PA dev

Does not modify the ASVspoof 2017 inverted_mel experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
    split_by_speaker,
)

_DEFAULT_PA_ROOT = _REPO_ROOT / "data" / "PA"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate_scores(
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


def save_artifacts(
    labels: np.ndarray,
    scores: np.ndarray,
    utt_ids: list[str],
    threshold: float,
    output_dir: Path,
    prefix: str,
    extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_scores(labels, scores, threshold)
    if extra:
        metrics = {**extra, **metrics}
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    preds = (scores >= threshold).astype(int)
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "true_label", "replay_probability", "prediction"])
        for utt_id, lab, score, pred in zip(utt_ids, labels, scores, preds):
            writer.writerow(
                [
                    utt_id,
                    "spoof" if lab else "bonafide",
                    score,
                    "spoof" if pred else "bonafide",
                ]
            )

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"EER = {metrics['eer_percent']:.2f}%")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(
        xlabel="False live rejection rate",
        ylabel="Replay detection rate",
        title="ROC curve",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix_live_replay"]),
        display_labels=["Live", "Replay"],
    ).plot(ax=axis, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


@torch.inference_mode()
def collect_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    utt_ids: list[str] = []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring", leave=False):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        utt_ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), utt_ids


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pa_root = Path(args.pa_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = AudioConfig(seconds=args.seconds, feature_type=args.feature_type)
    print(f"Feature={config.feature_type}; device={device}")

    train_protocol = pa_root / PROTOCOL_BY_SPLIT["train"]
    all_train = read_pa_cm_protocol(train_protocol)
    print(f"PA train protocol: {len(all_train)} files")

    # Speaker-disjoint split on full protocol first, then only scan audio we need.
    train_pool, val_pool = split_by_speaker(
        all_train, args.validation_fraction, args.seed
    )
    # Shuffle pools so early-exit readability checks are not speaker-biased.
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    readable_train, skipped_train = filter_readable_records(
        pa_root,
        "train",
        train_pool,
        cache_path=cache_dir / "pa2019_train_readable.json",
        force_refresh=args.refresh_cache,
        target_count=args.max_train if args.max_train > 0 else 0,
    )
    readable_val, skipped_val = filter_readable_records(
        pa_root,
        "train",
        val_pool,
        cache_path=cache_dir / "pa2019_train_readable.json",
        force_refresh=False,
        target_count=args.max_val if args.max_val > 0 else 0,
    )
    skipped_all = sorted(set(skipped_train) | set(skipped_val))
    print(
        f"Readable train-pool: {len(readable_train)} "
        f"(skipped {len(skipped_train)}); "
        f"val-pool: {len(readable_val)} (skipped {len(skipped_val)})"
    )
    if skipped_all:
        (output_dir / "train_skipped_utts.txt").write_text(
            "\n".join(skipped_all) + "\n", encoding="utf-8"
        )

    train_records = limit_records_stratified(readable_train, args.max_train, args.seed)
    val_records = limit_records_stratified(readable_val, args.max_val, args.seed + 1)

    n_train_spk = len({r.speaker_id for r in train_records})
    n_val_spk = len({r.speaker_id for r in val_records})
    print(
        f"Train: {len(train_records)} files / {n_train_spk} speakers "
        f"(bona={sum(r.label == 0 for r in train_records)}, "
        f"spoof={sum(r.label == 1 for r in train_records)})"
    )
    print(
        f"Val:   {len(val_records)} files / {n_val_spk} speakers "
        f"(bona={sum(r.label == 0 for r in val_records)}, "
        f"spoof={sum(r.label == 1 for r in val_records)})"
    )

    train_ds = PA2019WaveformDataset(
        pa_root,
        "train",
        train_records,
        config.sample_rate,
        config.samples,
        training=True,
        fix_length_fn=fix_length,
    )
    val_ds = PA2019WaveformDataset(
        pa_root,
        "train",
        val_records,
        config.sample_rate,
        config.samples,
        training=False,
        fix_length_fn=fix_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = ReplayCNN(config).to(device)
    spoof_count = sum(r.label for r in train_records)
    live_count = len(train_records) - spoof_count
    if spoof_count == 0 or live_count == 0:
        raise ValueError("Training set must contain both bonafide and spoof")
    pos_weight = torch.tensor([live_count / spoof_count], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = float("inf")
    patience_left = args.patience
    ckpt_path = output_dir / "best_inverted_mel_pa2019.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waveforms, labels, _ in progress:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(waveforms)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_labels, val_scores, val_ids = collect_scores(model, val_loader, device)
        val_eer, val_thr = calculate_eer(val_labels, val_scores)
        mean_loss = running / max(len(train_ds), 1)
        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} "
            f"val_eer={val_eer * 100:.2f}% thr={val_thr:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_eer": val_eer,
                "val_threshold": val_thr,
            }
        )

        if val_eer < best_eer:
            best_eer = val_eer
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_config": asdict(config),
                    "feature_type": config.feature_type,
                    "threshold": val_thr,
                    "val_eer": val_eer,
                    "epoch": epoch,
                    "dataset": "ASVspoof2019_PA",
                    "train_speakers": sorted({r.speaker_id for r in train_records}),
                    "val_speakers": sorted({r.speaker_id for r in val_records}),
                    "num_train": len(train_records),
                    "num_val": len(val_records),
                },
                ckpt_path,
            )
            save_artifacts(
                val_labels,
                val_scores,
                val_ids,
                val_thr,
                output_dir,
                "best_val",
                extra={
                    "feature_type": config.feature_type,
                    "split": "pa2019_train_speaker_val",
                    "epoch": epoch,
                },
            )
            print(f"Saved best checkpoint -> {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Done. Best val EER={best_eer * 100:.2f}%. Checkpoint: {ckpt_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pa-root", type=Path, default=_DEFAULT_PA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "runs" / "inverted_mel_pa2019",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_THIS_DIR / "cache",
    )
    parser.add_argument(
        "--feature-type",
        choices=["mel", "inverted_mel"],
        default="inverted_mel",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of PA-train speakers held out for validation",
    )
    parser.add_argument("--max-train", type=int, default=0, help="0 = all readable train")
    parser.add_argument("--max-val", type=int, default=0, help="0 = all readable val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-scan FLAC readability instead of using cache",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
