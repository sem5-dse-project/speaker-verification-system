"""ASVspoof 2019 LA CM helpers for LFCC training/eval."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm.auto import tqdm

PROTOCOL_BY_SPLIT = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

AUDIO_DIR_BY_SPLIT = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
}


@dataclass(frozen=True)
class LARecord:
    speaker_id: str
    utt_id: str
    label: int  # 0 bonafide, 1 spoof
    attack: str


def read_la_cm_protocol(protocol_path: Path) -> list[LARecord]:
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")
    records: list[LARecord] = []
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
                LARecord(
                    speaker_id=speaker_id,
                    utt_id=utt_id,
                    label=int(label_text == "spoof"),
                    attack=attack,
                )
            )
    return records


def resolve_audio_path(la_root: Path, split: str, utt_id: str) -> Path:
    audio_dir = la_root / AUDIO_DIR_BY_SPLIT[split]
    flac = audio_dir / f"{utt_id}.flac"
    wav = audio_dir / f"{utt_id}.wav"
    if flac.exists():
        return flac
    if wav.exists():
        return wav
    raise FileNotFoundError(f"Audio not found for {utt_id} under {audio_dir}")


def filter_readable_records(
    la_root: Path,
    split: str,
    records: list[LARecord],
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> tuple[list[LARecord], list[str]]:
    cached_ok: set[str] = set()
    cached_bad: set[str] = set()
    if cache_path is not None and cache_path.exists() and not force_refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_ok = set(data.get("readable", []))
        cached_bad = set(data.get("skipped", []))

    kept: list[LARecord] = []
    skipped: list[str] = []
    for record in tqdm(records, desc=f"LA {split} readability", leave=False):
        if record.utt_id in cached_bad:
            skipped.append(record.utt_id)
            continue
        if record.utt_id in cached_ok:
            kept.append(record)
            continue
        path = resolve_audio_path(la_root, split, record.utt_id)
        try:
            sf.read(str(path), dtype="float32")
            kept.append(record)
            cached_ok.add(record.utt_id)
        except Exception:
            skipped.append(record.utt_id)
            cached_bad.add(record.utt_id)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "readable": sorted(cached_ok),
                    "skipped": sorted(cached_bad),
                    "num_readable": len(cached_ok),
                    "num_skipped": len(cached_bad),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return kept, skipped


def split_by_speaker(
    records: list[LARecord], validation_fraction: float, seed: int
) -> tuple[list[LARecord], list[LARecord]]:
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


def limit_records_stratified(
    records: list[LARecord], maximum: int, seed: int
) -> list[LARecord]:
    if maximum <= 0 or maximum >= len(records):
        return records
    bona = [r for r in records if r.label == 0]
    spoof = [r for r in records if r.label == 1]
    rng = random.Random(seed)
    n_bona = max(1, int(round(maximum * len(bona) / max(len(records), 1))))
    n_bona = min(n_bona, len(bona), maximum - 1) if spoof else min(n_bona, len(bona))
    n_spoof = min(len(spoof), maximum - n_bona)
    out = rng.sample(bona, n_bona) + rng.sample(spoof, n_spoof)
    rng.shuffle(out)
    return out


def count_summary(records: list[LARecord]) -> dict:
    return {
        "total": len(records),
        "bonafide": sum(r.label == 0 for r in records),
        "spoof": sum(r.label == 1 for r in records),
        "speakers": len({r.speaker_id for r in records}),
    }


class LAWaveformDataset(Dataset):
    def __init__(
        self,
        la_root: Path,
        split: str,
        records: list[LARecord],
        sample_rate: int,
        num_samples: int,
        training: bool,
        fix_length_fn,
    ) -> None:
        self.la_root = la_root
        self.split = split
        self.records = records
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.training = training
        self.fix_length_fn = fix_length_fn
        if not records:
            raise ValueError("Empty LA dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        path = resolve_audio_path(self.la_root, self.split, record.utt_id)
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform).mean(dim=1)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = self.fix_length_fn(waveform, self.num_samples, self.training)
        return waveform, torch.tensor(record.label, dtype=torch.float32), record.utt_id
