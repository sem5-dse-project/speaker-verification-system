"""Shared train/eval helpers for Mel vs inverted-Mel vs LFCC comparison.

Reuses mixed 2017+PA loaders from ``inverted_mel_mixed_2017_pa2019``.
CUDA is used automatically when available.
"""

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
_MIXED_DIR = _THIS_DIR.parent / "inverted_mel_mixed_2017_pa2019"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "PA").exists() or (
        _REPO_ROOT / "replay-cnn-baseline" / "data"
    ).exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_MIXED_DIR))

from model import AudioConfig, ReplayCNN, fix_length  # noqa: E402

from build_mixed import (  # noqa: E402
    load_asv17_split,
    load_asv17_train_val,
    load_pa_split,
    load_pa_train_val,
)
from mixed_data import (  # noqa: E402
    MixedWaveformDataset,
    balance_by_corpus,
    count_summary,
    limit_mixed_stratified,
)

DEFAULT_ASV17 = _REPO_ROOT / "replay-cnn-baseline" / "data"
if not (DEFAULT_ASV17 / "ASVspoof2017_V2_train").exists():
    DEFAULT_ASV17 = _REPO_ROOT / "data"
DEFAULT_PA = _REPO_ROOT / "data" / "PA"
RUNS_DIR = _THIS_DIR / "runs"
CACHE_DIR = _THIS_DIR / "cache"
FEATURE_TYPES = ("mel", "inverted_mel", "lfcc")


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


def load_mixed_train_val(
    asv17_root: Path | None = None,
    pa_root: Path | None = None,
    cache_dir: Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
    max_train: int = 0,
    max_val: int = 0,
    balance_corpora: bool = True,
    refresh_cache: bool = False,
):
    asv17_root = Path(asv17_root or DEFAULT_ASV17)
    pa_root = Path(pa_root or DEFAULT_PA)
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    asv_train, asv_val = load_asv17_train_val(asv17_root, validation_fraction, seed)
    pa_train, pa_val, skipped = load_pa_train_val(
        pa_root, cache_dir, validation_fraction, seed + 1, refresh_cache
    )
    train_records = asv_train + pa_train
    val_records = asv_val + pa_val
    train_records = limit_mixed_stratified(train_records, max_train, seed)
    val_records = limit_mixed_stratified(val_records, max_val, seed + 2)
    if balance_corpora:
        train_records = balance_by_corpus(train_records, seed)
    return train_records, val_records, skipped


def train_feature(
    feature_type: str,
    *,
    asv17_root: Path | None = None,
    pa_root: Path | None = None,
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
    balance_corpora: bool = True,
    seed: int = 42,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> dict:
    """Train one front-end on mixed 2017+PA. Returns summary dict."""
    feature_type = feature_type.strip().lower()
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type must be one of {FEATURE_TYPES}")

    seed_everything(seed)
    device = pick_device(force_cpu)
    output_dir = Path(output_dir or (RUNS_DIR / feature_type))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AudioConfig(seconds=seconds, feature_type=feature_type)
    print(f"Train {feature_type} | device={device} | amp={device.type == 'cuda'}")

    train_records, val_records, skipped = load_mixed_train_val(
        asv17_root=asv17_root,
        pa_root=pa_root,
        cache_dir=cache_dir,
        validation_fraction=validation_fraction,
        seed=seed,
        max_train=max_train,
        max_val=max_val,
        balance_corpora=balance_corpora,
        refresh_cache=refresh_cache,
    )
    if skipped:
        (output_dir / "pa_train_skipped_utts.txt").write_text(
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

    train_ds = MixedWaveformDataset(
        train_records,
        config.sample_rate,
        config.samples,
        training=True,
        fix_length_fn=fix_length,
    )
    val_ds = MixedWaveformDataset(
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
    ckpt_path = output_dir / f"best_{feature_type}_mixed_2017_pa2019.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"{feature_type} epoch {epoch}/{epochs}")
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
                    "feature_type": config.feature_type,
                    "threshold": val_thr,
                    "val_eer": val_eer,
                    "epoch": epoch,
                    "experiment": "lfcc_vs_mel_compare",
                    "train_summary": train_summary,
                    "val_summary": val_summary,
                    "balance_corpora": balance_corpora,
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
        "feature_type": feature_type,
        "device": str(device),
        "best_val_eer": best_eer,
        "best_val_eer_percent": best_eer * 100.0,
        "checkpoint": str(ckpt_path),
        "train_summary": train_summary,
        "val_summary": val_summary,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Done {feature_type}. Best mixed-val EER={best_eer * 100:.2f}%")
    return summary


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["audio_config"])
    if "feature_type" not in cfg and "feature_type" in ckpt:
        cfg["feature_type"] = ckpt["feature_type"]
    # Drop unknown keys if older ckpts lack n_lfcc etc — AudioConfig has defaults
    allowed = set(AudioConfig.__dataclass_fields__)
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    config = AudioConfig(**cfg)
    model = ReplayCNN(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, float(ckpt["threshold"]), config, ckpt


def eval_checkpoint_on_corpora(
    checkpoint: Path,
    *,
    asv17_root: Path | None = None,
    pa_root: Path | None = None,
    cache_dir: Path | None = None,
    asv17_split: str = "dev",
    pa_split: str = "dev",
    batch_size: int = 8,
    workers: int = 0,
    force_cpu: bool = False,
    output_dir: Path | None = None,
    refresh_cache: bool = False,
) -> dict:
    """Score one checkpoint on ASVspoof2017 and PA2019; return per-corpus EER."""
    device = pick_device(force_cpu)
    asv17_root = Path(asv17_root or DEFAULT_ASV17)
    pa_root = Path(pa_root or DEFAULT_PA)
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    model, thr, config, ckpt = load_checkpoint(checkpoint, device)

    asv_records = load_asv17_split(asv17_root, asv17_split)
    pa_records, skipped = load_pa_split(
        pa_root, pa_split, cache_dir, refresh_cache=refresh_cache
    )

    results = {
        "feature_type": config.feature_type,
        "checkpoint": str(checkpoint),
        "threshold_from_train": thr,
        "device": str(device),
    }

    for name, records in (
        ("asvspoof2017", asv_records),
        ("pa2019", pa_records),
    ):
        if not records:
            results[name] = {"error": "no records"}
            continue
        ds = MixedWaveformDataset(
            records,
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
        # Report EER on this split (standard comparison); also metrics at train thr
        metrics = metrics_at_threshold(labels, scores, thr)
        results[name] = {
            **metrics,
            "n": int(len(labels)),
            "summary": count_summary(records),
        }
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with (out / f"{name}_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["utt_id", "true_label", "replay_probability", "prediction"]
                )
                preds = (scores >= thr).astype(int)
                for utt_id, lab, score, pred in zip(ids, labels, scores, preds):
                    writer.writerow(
                        [
                            utt_id,
                            "spoof" if lab else "bonafide",
                            score,
                            "spoof" if pred else "bonafide",
                        ]
                    )

    if skipped and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "pa_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def build_comparison_table(all_results: list[dict]) -> list[dict]:
    """Flatten eval results into Mel / I-Mel / LFCC × 2017 / PA EER rows."""
    rows = []
    for res in all_results:
        ft = res.get("feature_type", "?")
        row = {"feature": ft}
        for corpus, key in (("asvspoof2017", "eer_2017"), ("pa2019", "eer_pa")):
            block = res.get(corpus, {})
            if "eer_percent" in block:
                row[key] = round(float(block["eer_percent"]), 2)
                row[f"n_{corpus}"] = block.get("n")
            else:
                row[key] = None
        rows.append(row)
    return rows


def markdown_eer_table(rows: list[dict]) -> str:
    lines = [
        "| Feature | EER ASVspoof2017 (%) | EER PA2019 (%) |",
        "|---------|----------------------|----------------|",
    ]
    for row in rows:
        e17 = row.get("eer_2017")
        epa = row.get("eer_pa")
        s17 = f"{e17:.2f}" if isinstance(e17, (int, float)) else "—"
        spa = f"{epa:.2f}" if isinstance(epa, (int, float)) else "—"
        lines.append(f"| {row.get('feature', '?')} | {s17} | {spa} |")
    return "\n".join(lines)


def default_checkpoint(feature_type: str) -> Path:
    return RUNS_DIR / feature_type / f"best_{feature_type}_mixed_2017_pa2019.pt"
