"""Train and test a lightweight CNN for replay-attack detection.

Dataset: ASVspoof 2017 Version 2
Labels: 0 = bona fide/live, 1 = spoof/replay

This is a research baseline, not a guarantee of real-world liveness.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
#Fixed audio settings used everywhere in the code
class AudioConfig:
    sample_rate: int = 16000
    seconds: float = 4.0
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.seconds)


@dataclass(frozen=True)
# One row of metadata for one file
class Record:
    file_id: str
    label: int  # 0 = bona fide/live, 1 = spoof/replay
    speaker_id: str


# Protocol files for each split
PROTOCOL_FILES = {
    "train": "ASVspoof2017_V2_train.trn.txt",
    "dev": "ASVspoof2017_V2_dev.trl.txt",
    "eval": "ASVspoof2017_V2_eval.trl.txt",
}

# Directories for each split
SPLIT_DIRS = {
    "train": "ASVspoof2017_V2_train",
    "dev": "ASVspoof2017_V2_dev",
    "eval": "ASVspoof2017_V2_eval",
}

# Sets random seeds so runs are more repeatable.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Resolves the dataset root directory.
def resolve_dataset_root(path: Path) -> Path:
    """Accept the directory containing the extracted 2017 V2 folders."""
    path = path.expanduser().resolve()
    candidates = [
        path,
        path / "ASVspoof2017_V2",
        path / "ASVspoof2017",
    ]
    for candidate in candidates:
        if (candidate / SPLIT_DIRS["train"]).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find ASVspoof2017_V2_train. Expected a structure such as "
        f"{path / 'ASVspoof2017_V2_train'}"
    )

# Reads the protocol file for a given split and returns a list of Record objects.
def read_protocol(dataset_root: Path, split: str) -> list[Record]:
    protocol = dataset_root / "protocol_V2" / PROTOCOL_FILES[split]
    if not protocol.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol}")

    records: list[Record] = []
    with protocol.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue    
            if len(fields) != 7:
                raise ValueError(
                    f"Expected 7 fields at {protocol}:{line_number}, got {len(fields)}"
                )
            file_id = fields[0]
            label_text = fields[1].lower()
            speaker_id = fields[2]
            if label_text not in {"genuine", "spoof"}:
                raise ValueError(f"Unexpected label {label_text!r} at line {line_number}")
            records.append(Record(file_id, int(label_text == "spoof"), speaker_id))
    return records

# Creates a speaker-disjoint validation set from the official train set.
def split_by_speaker(
    records: list[Record], validation_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    """Create a speaker-disjoint validation set from the official train set."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    speakers = sorted({record.speaker_id for record in records})
    if len(speakers) < 2:
        raise ValueError("At least two speakers are needed for a speaker-disjoint split")
    rng = random.Random(seed)
    rng.shuffle(speakers)
    validation_count = min(
        len(speakers) - 1,
        max(1, round(len(speakers) * validation_fraction)),
    )
    validation_speakers = set(speakers[:validation_count])
    train_records = [r for r in records if r.speaker_id not in validation_speakers]
    validation_records = [r for r in records if r.speaker_id in validation_speakers]
    return train_records, validation_records


def limit_records(records: list[Record], maximum: int, seed: int) -> list[Record]:
    if maximum <= 0 or maximum >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, maximum)


def fix_length(waveform: torch.Tensor, length: int, random_crop: bool) -> torch.Tensor:
    if waveform.numel() == 0:
        raise ValueError("Empty audio file")
    if waveform.numel() < length:
        repeats = int(np.ceil(length / waveform.numel()))
        waveform = waveform.repeat(repeats)
    if waveform.numel() == length:
        return waveform
    max_start = waveform.numel() - length
    start = random.randint(0, max_start) if random_crop else max_start // 2
    return waveform[start : start + length]

# A dataset loader class for the replay-attack detection task.
class ReplayDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        records: list[Record],
        audio_config: AudioConfig,
        training: bool,
    ) -> None:
        self.audio_dir = dataset_root / SPLIT_DIRS[split]
        self.records = records
        self.config = audio_config
        self.training = training
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        # ASVspoof 2017 V2 protocol IDs normally already end in .wav.
        # Also support protocol variants that contain the stem only.
        filename = (
            record.file_id
            if Path(record.file_id).suffix.lower() == ".wav"
            else f"{record.file_id}.wav"
        )
        path = self.audio_dir / filename
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform).mean(dim=1)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )
        waveform = fix_length(waveform, self.config.samples, self.training)
        return waveform, torch.tensor(record.label, dtype=torch.float32), record.file_id

# A CNN model for the replay-attack detection task.
*** **
Step A - Mel spectrogram
Raw waveform → Log-Mel image. Then it normalizes each example

Step B - CNN blocks
A series of convolutional layers that extract features from the Mel spectrogram.

Step C - Classifier
A fully connected layer that outputs a single score for the replay-attack detection task.

** ***
class ReplayCNN(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )
        self.features = nn.Sequential(
            self.block(1, 16),
            self.block(16, 32),
            self.block(32, 64),
            self.block(64, 96),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 1),
        )

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform).clamp_min(1e-6).log()
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        mel = ((mel - mean) / std).unsqueeze(1)
        return self.classifier(self.features(mel)).squeeze(1)

# Wraps the dataset into batches
def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

# Collects scores for a given model and dataset.
@torch.inference_mode()
def collect_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    file_ids: list[str] = []
    for waveforms, batch_labels, batch_ids in tqdm(loader, desc="Scoring", leave=False):
        logits = model(waveforms.to(device, non_blocking=True))
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
        file_ids.extend(batch_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores), file_ids

# Calculates the Equal Error Rate (EER) and threshold for a given model and dataset.
def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return EER and threshold; higher score means more likely replay."""
    live_scores = scores[labels == 0]
    replay_scores = scores[labels == 1]
    if live_scores.size == 0 or replay_scores.size == 0:
        raise ValueError("EER requires both genuine and replay samples")
    # When the classes are perfectly separated, use the middle of the empty
    # score gap. This is more stable than selecting one class boundary.
    if float(live_scores.max()) < float(replay_scores.min()):
        threshold = (float(live_scores.max()) + float(replay_scores.min())) / 2.0
        return 0.0, threshold
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    replay_miss_rate = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - replay_miss_rate)))
    eer = float((fpr[index] + replay_miss_rate[index]) / 2.0)
    return eer, float(thresholds[index])

# Evaluates the performance of a given model and dataset.
def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    eer, eer_threshold = calculate_eer(labels, scores)
    return {
        "threshold": float(threshold),
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold_on_this_split": eer_threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_replay": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_replay": float(recall_score(labels, predictions, zero_division=0)),
        "f1_replay": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix_live_replay": matrix.tolist(),
    }

# Saves the metrics, predictions, and confusion matrix to the output directory.
def save_artifacts(
    labels: np.ndarray,
    scores: np.ndarray,
    file_ids: list[str],
    threshold: float,
    output_dir: Path,
    prefix: str,
) -> dict[str, float | list[list[int]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_scores(labels, scores, threshold)
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = (scores >= threshold).astype(int)
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "true_label", "replay_probability", "prediction"])
        for file_id, label, score, prediction in zip(
            file_ids, labels, scores, predictions
        ):
            writer.writerow(
                [file_id, "replay" if label else "live", score, "replay" if prediction else "live"]
            )

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"EER = {metrics['eer_percent']:.2f}%")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False live rejection rate", ylabel="Replay detection rate", title="ROC curve")
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

# Trains the model on the training set and calibrates it on the development set.
def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_root = resolve_dataset_root(Path(args.data_root))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AudioConfig(seconds=args.seconds)

    train_records = read_protocol(dataset_root, "train")
    all_dev_records = read_protocol(dataset_root, "dev")
    heldout_test_records, calibration_records = split_by_speaker(
        all_dev_records, args.calibration_fraction, args.seed
    )
    train_records = limit_records(train_records, args.max_train, args.seed)
    calibration_records = limit_records(
        calibration_records, args.max_val, args.seed + 1
    )
    train_speakers = len({record.speaker_id for record in train_records})
    calibration_speakers = sorted(
        {record.speaker_id for record in calibration_records}
    )
    heldout_test_speakers = sorted(
        {record.speaker_id for record in heldout_test_records}
    )
    print(
        f"Device: {device}; train files: {len(train_records)} "
        f"({train_speakers} speakers); calibration files: {len(calibration_records)} "
        f"({len(calibration_speakers)} dev speakers); held-out test files: "
        f"{len(heldout_test_records)} ({len(heldout_test_speakers)} dev speakers)"
    )

    train_dataset = ReplayDataset(
        dataset_root, "train", train_records, config, training=True
    )
    calibration_dataset = ReplayDataset(
        dataset_root, "dev", calibration_records, config, training=False
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True, device)
    calibration_loader = make_loader(
        calibration_dataset, args.batch_size, args.workers, False, device
    )

    model = ReplayCNN(config).to(device)
    replay_count = sum(record.label for record in train_records)
    live_count = len(train_records) - replay_count
    if replay_count == 0 or live_count == 0:
        raise ValueError("Training subset must contain both live and replay samples")
    pos_weight = torch.tensor([live_count / replay_count], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_replay_cnn.pt"

    # Trains the model for a given number of epochs.
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waveforms, labels, _ in progress:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(waveforms)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        calibration_labels, calibration_scores, calibration_ids = collect_scores(
            model, calibration_loader, device
        )
        calibration_eer, calibration_threshold = calculate_eer(
            calibration_labels, calibration_scores
        )
        mean_loss = running_loss / len(train_dataset)
        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} "
            f"calibration_eer={calibration_eer * 100:.2f}% "
            f"threshold={calibration_threshold:.4f}"
        )

        if calibration_eer < best_eer:
            best_eer = calibration_eer
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_config": asdict(config),
                    "threshold": calibration_threshold,
                    "calibration_eer": calibration_eer,
                    "calibration_dev_speakers": calibration_speakers,
                    "heldout_dev_test_speakers": heldout_test_speakers,
                    "calibration_fraction": args.calibration_fraction,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            save_artifacts(
                calibration_labels,
                calibration_scores,
                calibration_ids,
                calibration_threshold,
                output_dir,
                "best_calibration",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    print(f"Saved best checkpoint to {checkpoint_path}")

# Loads a checkpoint from a given path.
def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[ReplayCNN, float, AudioConfig, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = AudioConfig(**checkpoint["audio_config"])
    model = ReplayCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, float(checkpoint["threshold"]), config, checkpoint

# Evaluates the model on the official development or evaluation split.
def evaluate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, threshold, config, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    dataset_root = resolve_dataset_root(Path(args.data_root))
    records = read_protocol(dataset_root, args.split)
    if args.split == "dev" and not args.use_full_split:
        heldout_speakers = set(checkpoint.get("heldout_dev_test_speakers", []))
        if not heldout_speakers:
            raise ValueError(
                "Checkpoint does not contain held-out development speakers. "
                "Retrain with the updated script or pass --use-full-split."
            )
        records = [r for r in records if r.speaker_id in heldout_speakers]
        print(
            f"Testing only on {len(records)} held-out development files from "
            f"{len(heldout_speakers)} speakers"
        )
    records = limit_records(records, args.max_files, args.seed)
    dataset = ReplayDataset(
        dataset_root, args.split, records, config, training=False
    )
    loader = make_loader(dataset, args.batch_size, args.workers, False, device)
    labels, scores, file_ids = collect_scores(model, loader, device)
    metrics = save_artifacts(
        labels, scores, file_ids, threshold, Path(args.output), args.split
    )
    print(json.dumps(metrics, indent=2))

# Loads an external audio file and returns a tensor.
def load_external_audio(path: Path, config: AudioConfig) -> torch.Tensor:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).mean(dim=1)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    return fix_length(waveform, config.samples, random_crop=False).unsqueeze(0)

# Predicts the replay probability for a given audio file.
@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, default_threshold, config, _ = load_checkpoint(Path(args.checkpoint), device)
    threshold = default_threshold if args.threshold is None else args.threshold
    waveform = load_external_audio(Path(args.audio), config).to(device)
    probability = torch.sigmoid(model(waveform)).item()
    result = {
        "audio": str(Path(args.audio)),
        "replay_probability": probability,
        "threshold": threshold,
        "decision": "replay/suspicious" if probability >= threshold else "likely live",
    }
    print(json.dumps(result, indent=2))

# Builds the argument parser.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train and calibrate with separate development speakers"
    )
    train_parser.add_argument("--data-root", required=True)
    train_parser.add_argument("--output", default="runs/replay_cnn")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seconds", type=float, default=4.0)
    train_parser.add_argument("--max-train", type=int, default=0)
    train_parser.add_argument("--max-val", type=int, default=0)
    train_parser.add_argument(
        "--calibration-fraction",
        "--validation-fraction",
        dest="calibration_fraction",
        type=float,
        default=0.5,
        help="Fraction of development speakers used for calibration (default: 0.5)",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser(
        "eval", help="Test on the official development or evaluation split"
    )
    eval_parser.add_argument("--data-root", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output", default="runs/replay_cnn_eval")
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--workers", type=int, default=0)
    eval_parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    eval_parser.add_argument("--max-files", type=int, default=0)
    eval_parser.add_argument(
        "--use-full-split",
        action="store_true",
        help="Evaluate the entire split, including calibration speakers",
    )
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--cpu", action="store_true")
    eval_parser.set_defaults(func=evaluate)

    predict_parser = subparsers.add_parser("predict", help="Test one WAV/FLAC recording")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--audio", required=True)
    predict_parser.add_argument("--threshold", type=float)
    predict_parser.add_argument("--cpu", action="store_true")
    predict_parser.set_defaults(func=predict)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)