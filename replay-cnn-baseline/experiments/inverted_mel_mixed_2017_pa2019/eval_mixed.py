"""Evaluate a mixed 2017+PA2019 checkpoint on each corpus separately.

Reports ASVspoof2017 and PA2019 metrics side-by-side so generalization
is visible (do not rely on a single pooled EER).
"""

from __future__ import annotations

import argparse
import csv
import json
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
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "PA").exists() or (
        _REPO_ROOT / "replay-cnn-baseline" / "data"
    ).exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_INVERTED_MEL_DIR))
sys.path.insert(0, str(_THIS_DIR))

from inverted_mel_cnn import AudioConfig, ReplayCNN, fix_length  # noqa: E402

from build_mixed import load_asv17_split, load_pa_split  # noqa: E402
from mixed_data import MixedWaveformDataset, count_summary  # noqa: E402

_DEFAULT_ASV17 = _REPO_ROOT / "replay-cnn-baseline" / "data"
if not (_DEFAULT_ASV17 / "ASVspoof2017_V2_train").exists():
    _DEFAULT_ASV17 = _REPO_ROOT / "data"
_DEFAULT_PA = _REPO_ROOT / "data" / "PA"
_DEFAULT_CKPT = (
    _THIS_DIR / "runs" / "inverted_mel_mixed" / "best_inverted_mel_mixed_2017_pa2019.pt"
)


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
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring"):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


def score_records(model, records, config, device, batch_size, workers):
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
    return collect_scores(model, loader, device)


def save_plots(labels, scores, threshold, output_dir, prefix, eer_percent):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"EER = {eer_percent:.2f}%")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False live rejection", ylabel="Replay detection", title=prefix)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Live", "Replay"]).plot(
        ax=axis, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)


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
    save_plots(labels, scores, ckpt_thr, output_dir, prefix, transfer["eer_percent"])
    return result


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_thr, config, ckpt = load_checkpoint(Path(args.checkpoint), device)
    print(
        f"Loaded {args.checkpoint}\n"
        f"feature={config.feature_type}; device={device}; thr={ckpt_thr:.4f}; "
        f"epoch={ckpt.get('epoch')}"
    )

    results = {
        "experiment": "inverted_mel_mixed_2017_pa2019",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "feature_type": config.feature_type,
        "ckpt_threshold": ckpt_thr,
    }

    if args.corpus in ("asvspoof2017", "both"):
        asv_records = load_asv17_split(Path(args.asv17_root), args.asv17_split)
        print(f"ASVspoof2017 {args.asv17_split}: {count_summary(asv_records)}")
        labels, scores, ids = score_records(
            model, asv_records, config, device, args.batch_size, args.workers
        )
        results["asvspoof2017"] = eval_one(
            "asvspoof2017",
            labels,
            scores,
            ids,
            ckpt_thr,
            output_dir,
            {"split": args.asv17_split, "feature_type": config.feature_type},
        )

    if args.corpus in ("pa2019", "both"):
        pa_records, skipped = load_pa_split(
            Path(args.pa_root),
            args.pa_split,
            cache_dir,
            args.refresh_cache,
        )
        if skipped:
            (output_dir / f"pa_{args.pa_split}_skipped_utts.txt").write_text(
                "\n".join(skipped) + "\n", encoding="utf-8"
            )
        print(
            f"PA2019 {args.pa_split}: {count_summary(pa_records)} "
            f"(skipped_corrupt={len(skipped)})"
        )
        labels, scores, ids = score_records(
            model, pa_records, config, device, args.batch_size, args.workers
        )
        results["pa2019"] = eval_one(
            "pa2019",
            labels,
            scores,
            ids,
            ckpt_thr,
            output_dir,
            {
                "split": args.pa_split,
                "feature_type": config.feature_type,
                "num_skipped_corrupt": len(skipped),
            },
        )

    # Compact comparison table for logbook / README
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asv17-root", type=Path, default=_DEFAULT_ASV17)
    parser.add_argument("--pa-root", type=Path, default=_DEFAULT_PA)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument(
        "--corpus",
        choices=["both", "asvspoof2017", "pa2019"],
        default="both",
    )
    parser.add_argument(
        "--asv17-split",
        choices=["train", "dev", "eval"],
        default="dev",
    )
    parser.add_argument(
        "--pa-split",
        choices=["train", "dev", "eval"],
        default="dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "runs" / "eval_mixed_both_dev",
    )
    parser.add_argument("--cache-dir", type=Path, default=_THIS_DIR / "cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


if __name__ == "__main__":
    run_eval(build_parser().parse_args())
