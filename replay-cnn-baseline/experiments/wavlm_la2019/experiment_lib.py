"""Shared paths, metrics, and loaders for WavLM LA anti-spoofing."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_LFCC_LA_DIR = _THIS_DIR.parent / "lfcc_la2019"
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "LA").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_LFCC_LA_DIR))
sys.path.insert(0, str(_INVERTED_MEL_DIR))
sys.path.insert(0, str(_THIS_DIR))

from inverted_mel_cnn import fix_length  # noqa: E402
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
DEFAULT_CKPT = RUNS_DIR / "wavlm_la" / "best_wavlm_la2019.pt"
PAPER_LA_EER_PERCENT = 0.45
LFCC_LA_BASELINE_NOTE = "Compare to LFCC-LA and the paper's 0.45% EER on ASVspoof2019 LA (not replay EER)."


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


def build_la_loaders(
    la_root: Path,
    train_records,
    val_records,
    *,
    sample_rate: int,
    num_samples: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    train_ds = LAWaveformDataset(
        la_root,
        "train",
        train_records,
        sample_rate,
        num_samples,
        training=True,
        fix_length_fn=fix_length,
    )
    val_ds = LAWaveformDataset(
        la_root,
        "train",
        val_records,
        sample_rate,
        num_samples,
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
    return train_loader, val_loader


def forward_batch(model, waveforms, device):
    input_values = waveforms.to(device, non_blocking=True)
    attention_mask = torch.ones(
        input_values.shape[0],
        input_values.shape[1],
        dtype=torch.long,
        device=device,
    )
    return model(input_values, attention_mask=attention_mask)


@torch.inference_mode()
def collect_scores(model, loader, device):
    model.eval()
    labels, scores, ids = [], [], []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring", leave=False):
        logits = forward_batch(model, waveforms, device)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


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
    metrics = metrics_at_threshold(labels, scores, threshold)
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
        writer.writerow(["utt_id", "true_label", "spoof_probability", "prediction"])
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
    axis.set(xlabel="False bona fide rejection", ylabel="Spoof detection", title="ROC")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix_bona_spoof"]),
        display_labels=["Bona fide", "Spoof"],
    ).plot(ax=axis, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics
