"""Train / eval LFCC CNN on ASVspoof 2019 LA."""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_LFCC_DIR = _THIS_DIR.parent / "lfcc_vs_mel_compare"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "LA").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_LFCC_DIR))
sys.path.insert(0, str(_THIS_DIR))

from model import AudioConfig, ReplayCNN, fix_length  # noqa: E402

from la_data import (  # noqa: E402
    PROTOCOL_BY_SPLIT,
    LAWaveformDataset,
    count_summary,
    filter_readable_records,
    limit_records_stratified,
    read_la_cm_protocol,
    split_by_speaker,
)

DEFAULT_LA = _REPO_ROOT / "data" / "LA"
RUNS_DIR = _THIS_DIR / "runs"
CACHE_DIR = _THIS_DIR / "cache"
DEFAULT_CKPT = RUNS_DIR / "lfcc_la" / "best_lfcc_la2019.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "precision_spoof": float(precision_score(labels, preds, zero_division=0)),
        "recall_spoof": float(recall_score(labels, preds, zero_division=0)),
        "f1_spoof": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix_bona_spoof": confusion_matrix(
            labels, preds, labels=[0, 1]
        ).tolist(),
    }


@torch.inference_mode()
def collect_scores(model, loader, device):
    model.eval()
    labels, scores, ids = [], [], []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring", leave=False):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


def load_la_train_val(
    la_root: Path | None = None,
    cache_dir: Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
    max_train: int = 0,
    max_val: int = 0,
    refresh_cache: bool = False,
):
    la_root = Path(la_root or DEFAULT_LA)
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_train = read_la_cm_protocol(la_root / PROTOCOL_BY_SPLIT["train"])
    train_pool, val_pool = split_by_speaker(all_train, validation_fraction, seed)

    train_readable, skipped_train = filter_readable_records(
        la_root,
        "train",
        train_pool,
        cache_path=cache_dir / "la2019_train_readable.json",
        force_refresh=refresh_cache,
    )
    val_readable, skipped_val = filter_readable_records(
        la_root,
        "train",
        val_pool,
        cache_path=cache_dir / "la2019_train_readable.json",
        force_refresh=False,
    )
    train_readable = limit_records_stratified(train_readable, max_train, seed)
    val_readable = limit_records_stratified(val_readable, max_val, seed + 1)
    skipped = sorted(set(skipped_train) | set(skipped_val))
    return train_readable, val_readable, skipped


def train_lfcc_la(
    *,
    la_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    epochs: int = 15,
    patience: int = 4,
    batch_size: int = 8,
    workers: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seconds: float = 4.0,
    validation_fraction: float = 0.2,
    max_train: int = 0,
    max_val: int = 0,
    seed: int = 42,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> dict:
    seed_everything(seed)
    device = pick_device(force_cpu)
    output_dir = Path(output_dir or (RUNS_DIR / "lfcc_la"))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AudioConfig(seconds=seconds, feature_type="lfcc")
    print(f"Train LFCC on LA2019 | device={device}")

    train_records, val_records, skipped = load_la_train_val(
        la_root=la_root,
        cache_dir=cache_dir,
        validation_fraction=validation_fraction,
        seed=seed,
        max_train=max_train,
        max_val=max_val,
        refresh_cache=refresh_cache,
    )
    if skipped:
        (output_dir / "la_train_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    train_summary = count_summary(train_records)
    val_summary = count_summary(val_records)
    print(f"Train: {json.dumps(train_summary)}")
    print(f"Val:   {json.dumps(val_summary)}")
    (output_dir / "data_summary.json").write_text(
        json.dumps({"train": train_summary, "val": val_summary}, indent=2),
        encoding="utf-8",
    )

    la_root = Path(la_root or DEFAULT_LA)
    train_ds = LAWaveformDataset(
        la_root,
        "train",
        train_records,
        config.sample_rate,
        config.samples,
        training=True,
        fix_length_fn=fix_length,
    )
    val_ds = LAWaveformDataset(
        la_root,
        "train",
        val_records,
        config.sample_rate,
        config.samples,
        training=False,
        fix_length_fn=fix_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = ReplayCNN(config).to(device)
    spoof_count = sum(r.label for r in train_records)
    live_count = len(train_records) - spoof_count
    if spoof_count == 0 or live_count == 0:
        raise ValueError("Training set must contain both classes")
    pos_weight = torch.tensor([live_count / spoof_count], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = float("inf")
    patience_left = patience
    ckpt_path = output_dir / "best_lfcc_la2019.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"LFCC-LA epoch {epoch}/{epochs}")
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

        val_labels, val_scores, _ = collect_scores(model, val_loader, device)
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
            patience_left = patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_config": asdict(config),
                    "feature_type": "lfcc",
                    "threshold": val_thr,
                    "val_eer": val_eer,
                    "epoch": epoch,
                    "experiment": "lfcc_la2019",
                    "train_summary": train_summary,
                    "val_summary": val_summary,
                },
                ckpt_path,
            )
            print(f"Saved best -> {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "feature_type": "lfcc",
        "device": str(device),
        "best_val_eer": best_eer,
        "best_val_eer_percent": best_eer * 100.0,
        "checkpoint": str(ckpt_path),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "zero_shot_baseline_la_dev_eer_percent": 41.12,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Done. Best speaker-val EER={best_eer * 100:.2f}%")
    return summary


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg and "feature_type" in ckpt:
        cfg["feature_type"] = ckpt["feature_type"]
    allowed = set(AudioConfig.__dataclass_fields__)
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    config = AudioConfig(**cfg)
    model = ReplayCNN(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, float(ckpt["threshold"]), config, ckpt


def eval_lfcc_la(
    checkpoint: Path | None = None,
    *,
    la_root: Path | None = None,
    split: str = "dev",
    cache_dir: Path | None = None,
    batch_size: int = 8,
    workers: int = 0,
    max_utts: int = 0,
    force_cpu: bool = False,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict:
    device = pick_device(force_cpu)
    la_root = Path(la_root or DEFAULT_LA)
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint or DEFAULT_CKPT)
    model, train_thr, config, _ = load_checkpoint(checkpoint, device)

    records = read_la_cm_protocol(la_root / PROTOCOL_BY_SPLIT[split])
    if max_utts > 0 and max_utts < len(records):
        records = limit_records_stratified(records, max_utts, seed=42)

    readable, skipped = filter_readable_records(
        la_root,
        split,
        records,
        cache_path=cache_dir / f"la2019_{split}_readable.json",
        force_refresh=refresh_cache,
    )

    ds = LAWaveformDataset(
        la_root,
        split,
        readable,
        config.sample_rate,
        config.samples,
        training=False,
        fix_length_fn=fix_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    labels, scores, ids = collect_scores(model, loader, device)
    eer, eer_thr = calculate_eer(labels, scores)
    at_train = metrics_at_threshold(labels, scores, train_thr)
    at_oracle = metrics_at_threshold(labels, scores, eer_thr)

    results = {
        "experiment": "lfcc_la2019_eval",
        "checkpoint": str(checkpoint),
        "feature_type": config.feature_type,
        "split": split,
        "n": int(len(labels)),
        "n_bonafide": int((labels == 0).sum()),
        "n_spoof": int((labels == 1).sum()),
        "device": str(device),
        "train_threshold": train_thr,
        "oracle_eer_percent": eer * 100.0,
        "metrics_at_train_threshold": at_train,
        "metrics_at_oracle_eer_threshold": at_oracle,
        "zero_shot_mixed_imel_la_dev_eer_percent": 41.12,
        "summary": count_summary(readable),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"la2019_{split}_metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        with (out / f"la2019_{split}_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["utt_id", "true_label", "spoof_probability", "pred_at_train_thr"]
            )
            preds = (scores >= train_thr).astype(int)
            for uid, lab, sc, pr in zip(ids, labels, scores, preds):
                writer.writerow(
                    [
                        uid,
                        "spoof" if lab else "bonafide",
                        sc,
                        "spoof" if pr else "bonafide",
                    ]
                )
        if skipped:
            (out / f"la2019_{split}_skipped_utts.txt").write_text(
                "\n".join(skipped) + "\n", encoding="utf-8"
            )
    return results
