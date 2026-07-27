"""Evaluate inverted-Mel (or Mel) CNN zero-shot on ASVspoof 2019 PA.

Uses your trained checkpoint from experiments/inverted_mel (ASVspoof 2017)
and scores ASVspoof 2019 Physical Access CM protocols under data/PA.

New files only — does not modify the inverted_mel training scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "PA").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

_DEFAULT_PA_ROOT = _REPO_ROOT / "data" / "PA"
_DEFAULT_CKPT = (
    _INVERTED_MEL_DIR / "runs" / "inverted_mel" / "best_replay_cnn.pt"
)

sys.path.insert(0, str(_INVERTED_MEL_DIR))
from inverted_mel_cnn import AudioConfig, ReplayCNN, fix_length  # noqa: E402


PROTOCOL_BY_SPLIT = {
    "train": "ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt",
    "dev": "ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
    "eval": "ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt",
}

AUDIO_DIR_BY_SPLIT = {
    "train": "ASVspoof2019_PA_train/flac",
    "dev": "ASVspoof2019_PA_dev/flac",
    "eval": "ASVspoof2019_PA_eval/flac",
}


@dataclass(frozen=True)
class PARecord:
    speaker_id: str
    utt_id: str
    label: int  # 0 bonafide, 1 spoof/replay
    attack: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_pa_cm_protocol(protocol_path: Path) -> list[PARecord]:
    """
    ASVspoof 2019 PA CM protocol lines, e.g.:
      PA_0069 PA_D_0000001 aaa - bonafide
      PA_0069 PA_D_0005401 aaa AA spoof
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")

    records: list[PARecord] = []
    with protocol_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"Bad protocol line {line_number}: {line!r}")
            speaker_id, utt_id = fields[0], fields[1]
            label_text = fields[-1].lower()
            attack = fields[-2]
            if label_text not in {"bonafide", "spoof"}:
                raise ValueError(f"Unexpected label {label_text!r} at line {line_number}")
            records.append(
                PARecord(
                    speaker_id=speaker_id,
                    utt_id=utt_id,
                    label=int(label_text == "spoof"),
                    attack=attack,
                )
            )
    return records


def resolve_audio_path(pa_root: Path, split: str, utt_id: str) -> Path:
    audio_dir = pa_root / AUDIO_DIR_BY_SPLIT[split]
    flac = audio_dir / f"{utt_id}.flac"
    wav = audio_dir / f"{utt_id}.wav"
    if flac.exists():
        return flac
    if wav.exists():
        return wav
    raise FileNotFoundError(f"Audio not found for {utt_id} under {audio_dir}")


class PA2019Dataset(Dataset):
    def __init__(
        self,
        pa_root: Path,
        split: str,
        records: list[PARecord],
        audio_config: AudioConfig,
    ) -> None:
        self.pa_root = pa_root
        self.split = split
        self.config = audio_config
        self.records, self.skipped = self._filter_readable(records)
        if not self.records:
            raise RuntimeError(
                "No readable PA audio files found. "
                "Many FLACs may be corrupt — re-download ASVspoof2019_PA_dev."
            )

    def _filter_readable(
        self, records: list[PARecord]
    ) -> tuple[list[PARecord], list[str]]:
        kept: list[PARecord] = []
        skipped: list[str] = []
        for record in tqdm(records, desc="Checking audio readability"):
            path = resolve_audio_path(self.pa_root, self.split, record.utt_id)
            try:
                # Full decode check — partial reads can miss corrupt frames later in the file
                sf.read(str(path), dtype="float32")
                kept.append(record)
            except Exception:
                skipped.append(record.utt_id)
        return kept, skipped

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        path = resolve_audio_path(self.pa_root, self.split, record.utt_id)
        try:
            waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            # Should be rare after pre-filter; return silence + skip marker label unused
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc
        waveform = torch.from_numpy(waveform).mean(dim=1)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )
        waveform = fix_length(waveform, self.config.samples, random_crop=False)
        return waveform, torch.tensor(record.label, dtype=torch.float32), record.utt_id


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[ReplayCNN, float, AudioConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["audio_config"])
    if "feature_type" not in cfg and "feature_type" in checkpoint:
        cfg["feature_type"] = checkpoint["feature_type"]
    config = AudioConfig(**cfg)
    model = ReplayCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["threshold"]), config, checkpoint


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


@torch.inference_mode()
def collect_scores(
    model: ReplayCNN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    utt_ids: list[str] = []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring PA2019"):
        logits = model(waveforms.to(device, non_blocking=True))
        probs = torch.sigmoid(logits).cpu().tolist()
        scores.extend(probs)
        labels.extend(batch_labels.tolist())
        utt_ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), utt_ids


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pa_root = Path(args.pa_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_threshold, config, checkpoint = load_checkpoint(
        Path(args.checkpoint), device
    )
    print(
        f"Checkpoint: {args.checkpoint}\n"
        f"Feature: {config.feature_type}; device={device}; "
        f"ckpt_threshold={ckpt_threshold:.6f}"
    )

    protocol = pa_root / PROTOCOL_BY_SPLIT[args.split]
    records = read_pa_cm_protocol(protocol)
    if args.max_files > 0 and args.max_files < len(records):
        rng = random.Random(args.seed)
        # stratified-ish: keep ratio approx by sampling from each class
        bona = [r for r in records if r.label == 0]
        spoof = [r for r in records if r.label == 1]
        n_bona = max(1, int(round(args.max_files * len(bona) / len(records))))
        n_spoof = max(1, args.max_files - n_bona)
        n_bona = min(n_bona, len(bona))
        n_spoof = min(n_spoof, len(spoof))
        records = rng.sample(bona, n_bona) + rng.sample(spoof, n_spoof)
        rng.shuffle(records)

    n_bona = sum(r.label == 0 for r in records)
    n_spoof = sum(r.label == 1 for r in records)
    print(
        f"PA2019 split={args.split}; files={len(records)} "
        f"(bonafide={n_bona}, spoof={n_spoof})"
    )

    dataset = PA2019Dataset(pa_root, args.split, records, config)
    if dataset.skipped:
        print(
            f"WARNING: skipped {len(dataset.skipped)} unreadable FLAC files "
            f"(likely corrupt/incomplete download). "
            f"Using {len(dataset)} readable files."
        )
        (output_dir / f"pa2019_{args.split}_skipped_utts.txt").write_text(
            "\n".join(dataset.skipped) + "\n", encoding="utf-8"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels, scores, utt_ids = collect_scores(model, loader, device)

    n_bona = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    # 1) Transfer operating point: threshold from 2017 checkpoint
    transfer = metrics_at_threshold(labels, scores, ckpt_threshold)
    # 2) Oracle EER threshold on this PA split
    eer, eer_thr = calculate_eer(labels, scores)
    oracle = metrics_at_threshold(labels, scores, eer_thr)

    result = {
        "experiment": "inverted_mel_zero_shot_asvspoof2019_pa",
        "dataset": "ASVspoof2019_PA",
        "split": args.split,
        "feature_type": config.feature_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_protocol_files": len(records),
        "num_scored_files": len(dataset),
        "num_skipped_corrupt": len(dataset.skipped),
        "num_bonafide": n_bona,
        "num_spoof": n_spoof,
        "transfer_from_asvspoof2017_threshold": transfer,
        "oracle_eer_on_pa2019": oracle,
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
            [
                "utt_id",
                "true_label",
                "replay_probability",
                "pred_transfer_thr",
                "pred_eer_thr",
            ]
        )
        pred_t = (scores >= ckpt_threshold).astype(int)
        pred_e = (scores >= eer_thr).astype(int)
        for utt_id, lab, score, pt, pe in zip(
            utt_ids, labels, scores, pred_t, pred_e
        ):
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
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "runs" / "pa2019_dev",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_eval(args)
