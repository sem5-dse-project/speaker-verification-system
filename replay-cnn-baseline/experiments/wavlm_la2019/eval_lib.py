"""Evaluate WavLM LA checkpoint on ASVspoof 2019 LA splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from experiment_lib import (
    CACHE_DIR,
    DEFAULT_CKPT,
    DEFAULT_LA,
    LFCC_LA_BASELINE_NOTE,
    PAPER_LA_EER_PERCENT,
    RUNS_DIR,
    calculate_eer,
    collect_scores,
    count_summary,
    filter_readable_records,
    limit_records_stratified,
    metrics_at_threshold,
    pick_device,
    read_la_cm_protocol,
    PROTOCOL_BY_SPLIT,
)
from inverted_mel_cnn import fix_length
from la_data import LAWaveformDataset
from model import WavLMAudioConfig, WavLMSpoofDetector


def eval_wavlm_la(
    *,
    checkpoint: Path | None = None,
    la_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    split: str = "dev",
    batch_size: int = 4,
    workers: int = 0,
    max_utts: int = 0,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> dict:
    checkpoint = Path(checkpoint or DEFAULT_CKPT)
    la_root = Path(la_root or DEFAULT_LA)
    output_dir = Path(output_dir or (RUNS_DIR / f"eval_wavlm_la_{split}"))
    cache_dir = Path(cache_dir or CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(force_cpu)
    model, ckpt = WavLMSpoofDetector.load_checkpoint(checkpoint, device)
    ckpt_thr = float(ckpt["threshold"])
    audio_cfg = ckpt.get("audio_config") or {}
    audio_config = WavLMAudioConfig(
        sample_rate=int(audio_cfg.get("sample_rate", 16000)),
        seconds=float(audio_cfg.get("seconds", 4.0)),
    )

    print(
        f"Loaded {checkpoint}\n"
        f"model={ckpt.get('model_id')}; device={device}; thr={ckpt_thr:.4f}; "
        f"epoch={ckpt.get('epoch')}"
    )

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
    if skipped:
        (output_dir / f"la2019_{split}_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    print(f"LA2019 {split}: {count_summary(readable)} (skipped={len(skipped)})")

    ds = LAWaveformDataset(
        la_root,
        split,
        readable,
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
    labels, scores, ids = collect_scores(model, loader, device)
    eer, eer_thr = calculate_eer(labels, scores)
    at_train = metrics_at_threshold(labels, scores, ckpt_thr)
    at_oracle = metrics_at_threshold(labels, scores, eer_thr)

    results = {
        "experiment": "wavlm_la2019_eval",
        "paper": "10.1145/3708597.3708621",
        "paper_la_eer_percent": PAPER_LA_EER_PERCENT,
        "note": LFCC_LA_BASELINE_NOTE,
        "checkpoint": str(checkpoint.resolve()),
        "model_id": ckpt.get("model_id"),
        "split": split,
        "num_scored_files": int(len(labels)),
        "num_bonafide": int((labels == 0).sum()),
        "num_spoof": int((labels == 1).sum()),
        "ckpt_threshold": ckpt_thr,
        "metrics_at_train_val_threshold": at_train,
        "metrics_at_oracle_eer_threshold": at_oracle,
        "summary": count_summary(readable),
        "num_skipped_corrupt": len(skipped),
    }

    prefix = f"la2019_{split}"
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["utt_id", "true_label", "spoof_probability", "pred_ckpt_thr", "pred_eer_thr"]
        )
        pred_t = (scores >= ckpt_thr).astype(int)
        pred_e = (scores >= eer_thr).astype(int)
        for utt_id, lab, score, pt, pe in zip(ids, labels, scores, pred_t, pred_e):
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
    axis.plot(fpr, tpr, label=f"oracle EER = {at_oracle['eer_percent']:.2f}%")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False bona fide rejection", ylabel="Spoof detection", title=prefix)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    preds = (scores >= ckpt_thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bona fide", "Spoof"]).plot(
        ax=axis, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(
        json.dumps(
            {
                "split": split,
                "n": results["num_scored_files"],
                "oracle_eer_percent": at_oracle["eer_percent"],
                "ckpt_thr_eer_percent": at_train["eer_percent"],
                "paper_la_eer_percent": PAPER_LA_EER_PERCENT,
            },
            indent=2,
        )
    )
    return results
