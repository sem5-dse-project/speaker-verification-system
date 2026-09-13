"""Shared paths, metrics, and data helpers for Wav2Vec2 replay experiments."""

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
_MIXED_DIR = _THIS_DIR.parent / "inverted_mel_mixed_2017_pa2019"
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "PA").exists() or (
        _REPO_ROOT / "replay-cnn-baseline" / "data"
    ).exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_INVERTED_MEL_DIR))
sys.path.insert(0, str(_MIXED_DIR))
sys.path.insert(0, str(_THIS_DIR))

from inverted_mel_cnn import fix_length  # noqa: E402
from mixed_data import MixedWaveformDataset  # noqa: E402

DEFAULT_ASV17 = _REPO_ROOT / "replay-cnn-baseline" / "data"
if not (DEFAULT_ASV17 / "ASVspoof2017_V2_train").exists():
    DEFAULT_ASV17 = _REPO_ROOT / "data"
DEFAULT_PA = _REPO_ROOT / "data" / "PA"
RUNS_DIR = _THIS_DIR / "runs"
CACHE_DIR = _THIS_DIR / "cache"
DEFAULT_CKPT = RUNS_DIR / "wav2vec2_mixed" / "best_wav2vec2_mixed_2017_pa2019.pt"


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
        "precision_replay": float(precision_score(labels, preds, zero_division=0)),
        "recall_replay": float(recall_score(labels, preds, zero_division=0)),
        "f1_replay": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix_live_replay": confusion_matrix(
            labels, preds, labels=[0, 1]
        ).tolist(),
    }


def build_waveform_loaders(
    train_records,
    val_records,
    *,
    sample_rate: int,
    num_samples: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    train_ds = MixedWaveformDataset(
        train_records,
        sample_rate,
        num_samples,
        training=True,
        fix_length_fn=fix_length,
    )
    val_ds = MixedWaveformDataset(
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
    return model(input_values, attention_mask=None)


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
    axis.set(xlabel="False live rejection", ylabel="Replay detection", title="ROC")
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
