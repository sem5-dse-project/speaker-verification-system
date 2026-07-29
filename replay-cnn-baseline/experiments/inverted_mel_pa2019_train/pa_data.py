"""Shared ASVspoof 2019 PA data helpers for inverted-Mel training/eval."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

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


def read_pa_cm_protocol(protocol_path: Path) -> list[PARecord]:
    """Parse ASVspoof 2019 PA CM protocol (train/dev/eval)."""
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


def is_readable_audio(path: Path) -> bool:
    try:
        sf.read(str(path), dtype="float32")
        return True
    except Exception:
        return False


def _load_cache(cache_path: Path | None) -> dict:
    if cache_path is None or not cache_path.exists():
        return {"readable": [], "skipped": []}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cache(
    cache_path: Path | None,
    split: str,
    readable_ids: set[str],
    skipped_ids: set[str],
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prev = _load_cache(cache_path)
    readable = sorted(set(prev.get("readable", [])) | readable_ids)
    skipped = sorted(set(prev.get("skipped", [])) | skipped_ids)
    cache_path.write_text(
        json.dumps(
            {
                "split": split,
                "readable": readable,
                "skipped": skipped,
                "num_readable": len(readable),
                "num_skipped": len(skipped),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def filter_readable_records(
    pa_root: Path,
    split: str,
    records: list[PARecord],
    cache_path: Path | None = None,
    force_refresh: bool = False,
    target_count: int = 0,
) -> tuple[list[PARecord], list[str]]:
    """
    Keep only fully decodable audio files.

    Caches readable/skipped utt IDs. If ``target_count`` > 0, stop once enough
    readable files are found (uses cache first).
    """
    cached = {} if force_refresh else _load_cache(cache_path)
    readable_ids = set(cached.get("readable", []))
    skipped_ids = set(cached.get("skipped", []))

    kept: list[PARecord] = []
    skipped: list[str] = []
    unknown: list[PARecord] = []

    for record in records:
        if not force_refresh and record.utt_id in readable_ids:
            kept.append(record)
        elif not force_refresh and record.utt_id in skipped_ids:
            skipped.append(record.utt_id)
        else:
            unknown.append(record)

    if target_count > 0 and len(kept) >= target_count and not force_refresh:
        return kept, skipped

    newly_kept: list[PARecord] = []
    newly_skipped: list[str] = []
    for record in tqdm(unknown, desc=f"Checking {split} audio"):
        path = resolve_audio_path(pa_root, split, record.utt_id)
        if is_readable_audio(path):
            newly_kept.append(record)
            kept.append(record)
        else:
            newly_skipped.append(record.utt_id)
            skipped.append(record.utt_id)
        if target_count > 0 and len(kept) >= target_count:
            break

    _save_cache(
        cache_path,
        split,
        {r.utt_id for r in newly_kept} | {r.utt_id for r in kept},
        set(newly_skipped) | set(skipped),
    )
    return kept, skipped


def split_by_speaker(
    records: list[PARecord], validation_fraction: float, seed: int
) -> tuple[list[PARecord], list[PARecord]]:
    """Speaker-disjoint split within a protocol list (e.g. PA train)."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    speakers = sorted({r.speaker_id for r in records})
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers for a speaker-disjoint split")
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = min(len(speakers) - 1, max(1, round(len(speakers) * validation_fraction)))
    val_speakers = set(speakers[:n_val])
    train = [r for r in records if r.speaker_id not in val_speakers]
    val = [r for r in records if r.speaker_id in val_speakers]
    return train, val


def limit_records(records: list[PARecord], maximum: int, seed: int) -> list[PARecord]:
    if maximum <= 0 or maximum >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, maximum)


def limit_records_stratified(
    records: list[PARecord], maximum: int, seed: int
) -> list[PARecord]:
    """Keep approximate class ratio when subsampling."""
    if maximum <= 0 or maximum >= len(records):
        return records
    bona = [r for r in records if r.label == 0]
    spoof = [r for r in records if r.label == 1]
    rng = random.Random(seed)
    n_bona = max(1, int(round(maximum * len(bona) / max(len(records), 1))))
    n_bona = min(n_bona, len(bona), maximum - 1)
    n_spoof = min(len(spoof), maximum - n_bona)
    out = rng.sample(bona, n_bona) + rng.sample(spoof, n_spoof)
    rng.shuffle(out)
    return out


class PA2019WaveformDataset(Dataset):
    """Load PA FLACs as fixed-length mono waveforms for ReplayCNN."""

    def __init__(
        self,
        pa_root: Path,
        split: str,
        records: list[PARecord],
        sample_rate: int,
        num_samples: int,
        training: bool,
        fix_length_fn,
    ) -> None:
        self.pa_root = pa_root
        self.split = split
        self.records = records
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.training = training
        self.fix_length_fn = fix_length_fn
        if not records:
            raise ValueError(f"Empty dataset for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        path = resolve_audio_path(self.pa_root, self.split, record.utt_id)
        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform).mean(dim=1)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = self.fix_length_fn(waveform, self.num_samples, self.training)
        return waveform, torch.tensor(record.label, dtype=torch.float32), record.utt_id
