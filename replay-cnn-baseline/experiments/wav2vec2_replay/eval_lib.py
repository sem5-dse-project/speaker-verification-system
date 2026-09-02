"""Evaluate Wav2Vec2 replay checkpoint on ASVspoof 2017 and PA2019 separately."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

_MIXED_DIR = Path(__file__).resolve().parent.parent / "inverted_mel_mixed_2017_pa2019"
_INVERTED_MEL_DIR = Path(__file__).resolve().parent.parent / "inverted_mel"
sys.path.insert(0, str(_MIXED_DIR))
sys.path.insert(0, str(_INVERTED_MEL_DIR))

from build_mixed import load_asv17_split, load_pa_split
from experiment_lib import (
    CACHE_DIR,
    DEFAULT_ASV17,
    DEFAULT_CKPT,
    DEFAULT_PA,
    RUNS_DIR,
    calculate_eer,
    collect_scores,
    metrics_at_threshold,
    pick_device,
)
from inverted_mel_cnn import fix_length
from mixed_data import MixedWaveformDataset, count_summary
from model import Wav2Vec2AudioConfig, Wav2Vec2ReplayDetector


def score_records(model, records, audio_config, device, batch_size, workers):
    ds = MixedWaveformDataset(
        records,
        audio_config.sample_rate,
        audio_config.samples,
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
    return collect_scores(model, loader, device)


def eval_one(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    utt_ids: list[str],
    ckpt_thr: float,
    output_dir: Path,
    extra: dict,
) -> dict:
    transfer = metrics_at_threshold(labels, scores, ckpt_thr)
    eer, eer_thr = calculate_eer(labels, scores)
    oracle = metrics_at_threshold(labels, scores, eer_thr)
    result = {
        **extra,
        "corpus": name,
        "num_scored_files": int(len(labels)),
        "num_bonafide": int((labels == 0).sum()),
        "num_spoof": int((labels == 1).sum()),
        "metrics_at_train_val_threshold": transfer,
        "metrics_at_oracle_eer_threshold": oracle,
    }
    prefix = f"{name}_{extra.get('split', 'eval')}"
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

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"EER = {transfer['eer_percent']:.2f}%")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False live rejection", ylabel="Replay detection", title=prefix)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    preds = (scores >= ckpt_thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Live", "Replay"]).plot(
        ax=axis, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return result


def eval_wav2vec2_replay(
    *,
    checkpoint: Path | None = None,
    asv17_root: Path | None = None,
    pa_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    corpus: str = "both",
    asv17_split: str = "dev",
    pa_split: str = "dev",
    batch_size: int = 8,
    workers: int = 0,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> dict:
    checkpoint = Path(checkpoint or DEFAULT_CKPT)
    asv17_root = Path(asv17_root or DEFAULT_ASV17)
    pa_root = Path(pa_root or DEFAULT_PA)
    output_dir = Path(output_dir or (RUNS_DIR / "eval_wav2vec2_both_dev"))
    cache_dir = Path(cache_dir or CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(force_cpu)
    model, ckpt = Wav2Vec2ReplayDetector.load_checkpoint(checkpoint, device)
    ckpt_thr = float(ckpt["threshold"])
    audio_cfg = ckpt.get("audio_config") or {}
    audio_config = Wav2Vec2AudioConfig(
        sample_rate=int(audio_cfg.get("sample_rate", 16000)),
        seconds=float(audio_cfg.get("seconds", 4.0)),
    )

    print(
        f"Loaded {checkpoint}\n"
        f"model={ckpt.get('model_id')}; device={device}; thr={ckpt_thr:.4f}; "
        f"epoch={ckpt.get('epoch')}"
    )

    results = {
        "experiment": "wav2vec2_mixed_2017_pa2019",
        "checkpoint": str(checkpoint.resolve()),
        "model_id": ckpt.get("model_id"),
        "ckpt_threshold": ckpt_thr,
    }

    if corpus in ("asvspoof2017", "both"):
        asv_records = load_asv17_split(asv17_root, asv17_split)
        print(f"ASVspoof2017 {asv17_split}: {count_summary(asv_records)}")
        labels, scores, ids = score_records(
            model, asv_records, audio_config, device, batch_size, workers
        )
        results["asvspoof2017"] = eval_one(
            "asvspoof2017",
            labels,
            scores,
            ids,
            ckpt_thr,
            output_dir,
            {"split": asv17_split, "model_id": ckpt.get("model_id")},
        )

    if corpus in ("pa2019", "both"):
        pa_records, skipped = load_pa_split(
            pa_root,
            pa_split,
            cache_dir,
            refresh_cache,
        )
        if skipped:
            (output_dir / f"pa_{pa_split}_skipped_utts.txt").write_text(
                "\n".join(skipped) + "\n", encoding="utf-8"
            )
        print(
            f"PA2019 {pa_split}: {count_summary(pa_records)} "
            f"(skipped_corrupt={len(skipped)})"
        )
        labels, scores, ids = score_records(
            model, pa_records, audio_config, device, batch_size, workers
        )
        results["pa2019"] = eval_one(
            "pa2019",
            labels,
            scores,
            ids,
            ckpt_thr,
            output_dir,
            {
                "split": pa_split,
                "model_id": ckpt.get("model_id"),
                "num_skipped_corrupt": len(skipped),
            },
        )

    summary_rows = []
    for key in ("asvspoof2017", "pa2019"):
        if key not in results:
            continue
        block = results[key]
        summary_rows.append(
            {
                "corpus": key,
                "split": block["split"],
                "num_files": block["num_scored_files"],
                "eer_percent_ckpt_thr": block["metrics_at_train_val_threshold"][
                    "eer_percent"
                ],
                "eer_percent_oracle": block["metrics_at_oracle_eer_threshold"][
                    "eer_percent"
                ],
                "f1_replay_ckpt_thr": block["metrics_at_train_val_threshold"][
                    "f1_replay"
                ],
            }
        )
    results["comparison"] = summary_rows
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({"comparison": summary_rows}, indent=2))
    return results
