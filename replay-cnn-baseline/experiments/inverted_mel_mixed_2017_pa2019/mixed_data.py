"""Unified sample records for mixed ASVspoof 2017 + PA2019 training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MixedRecord:
    """Corpus-tagged utterance used by MixedWaveformDataset."""

    corpus: str  # "asvspoof2017" | "pa2019"
    utt_id: str
    speaker_id: str
    label: int  # 0 bonafide/genuine, 1 spoof/replay
    split: str  # audio split folder key (train/dev/eval)
    path: Path


def balance_by_corpus(
    records: list[MixedRecord], seed: int
) -> list[MixedRecord]:
    """Upsample the smaller corpus so both appear equally often per epoch."""
    by = {"asvspoof2017": [], "pa2019": []}
    for record in records:
        by[record.corpus].append(record)
    a = by["asvspoof2017"]
    b = by["pa2019"]
    if not a or not b:
        return list(records)
    rng = random.Random(seed)
    target = max(len(a), len(b))

    def upsample(pool: list[MixedRecord], n: int) -> list[MixedRecord]:
        if len(pool) >= n:
            return list(pool)
        out = list(pool)
        while len(out) < n:
            out.extend(rng.sample(pool, min(len(pool), n - len(out))))
        return out

    mixed = upsample(a, target) + upsample(b, target)
    rng.shuffle(mixed)
    return mixed


def limit_mixed_stratified(
    records: list[MixedRecord], maximum: int, seed: int
) -> list[MixedRecord]:
    """Optional cap keeping corpus and class roughly balanced."""
    if maximum <= 0 or maximum >= len(records):
        return records
    rng = random.Random(seed)
    half = maximum // 2
    out: list[MixedRecord] = []
    for corpus in ("asvspoof2017", "pa2019"):
        pool = [r for r in records if r.corpus == corpus]
        if not pool:
            continue
        take = min(len(pool), half if out else maximum - len(out))
        bona = [r for r in pool if r.label == 0]
        spoof = [r for r in pool if r.label == 1]
        n_bona = max(1, int(round(take * len(bona) / max(len(pool), 1))))
        n_bona = min(n_bona, len(bona), take - 1) if spoof else min(n_bona, len(bona))
        n_spoof = min(len(spoof), take - n_bona)
        picked = rng.sample(bona, n_bona) + rng.sample(spoof, n_spoof)
        out.extend(picked)
    rng.shuffle(out)
    return out[:maximum]


def count_summary(records: list[MixedRecord]) -> dict:
    return {
        "total": len(records),
        "asvspoof2017": sum(r.corpus == "asvspoof2017" for r in records),
        "pa2019": sum(r.corpus == "pa2019" for r in records),
        "bonafide": sum(r.label == 0 for r in records),
        "spoof": sum(r.label == 1 for r in records),
        "speakers": len({(r.corpus, r.speaker_id) for r in records}),
    }


class MixedWaveformDataset(Dataset):
    """Load 2017 WAV / PA FLAC paths as fixed-length mono waveforms."""

    def __init__(
        self,
        records: list[MixedRecord],
        sample_rate: int,
        num_samples: int,
        training: bool,
        fix_length_fn,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.training = training
        self.fix_length_fn = fix_length_fn
        if not records:
            raise ValueError("Empty mixed dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        waveform, sr = sf.read(str(record.path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform).mean(dim=1)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = self.fix_length_fn(waveform, self.num_samples, self.training)
        # id encodes corpus so eval CSVs stay unambiguous
        sample_id = f"{record.corpus}:{record.utt_id}"
        return waveform, torch.tensor(record.label, dtype=torch.float32), sample_id
