"""Build MixedRecord lists from ASVspoof 2017 V2 and ASVspoof 2019 PA."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_INVERTED_MEL_DIR = _THIS_DIR.parent / "inverted_mel"
_PA_TRAIN_DIR = _THIS_DIR.parent / "inverted_mel_pa2019_train"
sys.path.insert(0, str(_INVERTED_MEL_DIR))
sys.path.insert(0, str(_PA_TRAIN_DIR))

from inverted_mel_cnn import (  # noqa: E402
    PROTOCOL_FILES,
    SPLIT_DIRS,
    read_protocol,
    resolve_dataset_root,
)
from pa_data import (  # noqa: E402
    PROTOCOL_BY_SPLIT,
    filter_readable_records,
    read_pa_cm_protocol,
    resolve_audio_path,
    split_by_speaker as pa_split_by_speaker,
)

from mixed_data import MixedRecord  # noqa: E402


def resolve_asv17_root(path: Path) -> Path:
    return resolve_dataset_root(path)


def load_asv17_train_val(
    asv17_root: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[list[MixedRecord], list[MixedRecord]]:
    root = resolve_asv17_root(asv17_root)
    all_train = read_protocol(root, "train")
    # Speaker-disjoint within 2017 train
    from inverted_mel_cnn import split_by_speaker as asv_split

    train_recs, val_recs = asv_split(all_train, validation_fraction, seed)

    def to_mixed(recs, split: str) -> list[MixedRecord]:
        out: list[MixedRecord] = []
        audio_dir = root / SPLIT_DIRS[split]
        for r in recs:
            filename = (
                r.file_id
                if Path(r.file_id).suffix.lower() == ".wav"
                else f"{r.file_id}.wav"
            )
            path = audio_dir / filename
            if not path.exists():
                continue
            out.append(
                MixedRecord(
                    corpus="asvspoof2017",
                    utt_id=r.file_id,
                    speaker_id=r.speaker_id,
                    label=r.label,
                    split=split,
                    path=path,
                )
            )
        return out

    return to_mixed(train_recs, "train"), to_mixed(val_recs, "train")


def load_asv17_split(asv17_root: Path, split: str) -> list[MixedRecord]:
    root = resolve_asv17_root(asv17_root)
    recs = read_protocol(root, split)
    audio_dir = root / SPLIT_DIRS[split]
    out: list[MixedRecord] = []
    for r in recs:
        filename = (
            r.file_id
            if Path(r.file_id).suffix.lower() == ".wav"
            else f"{r.file_id}.wav"
        )
        path = audio_dir / filename
        if not path.exists():
            continue
        out.append(
            MixedRecord(
                corpus="asvspoof2017",
                utt_id=r.file_id,
                speaker_id=r.speaker_id,
                label=r.label,
                split=split,
                path=path,
            )
        )
    return out


def load_pa_train_val(
    pa_root: Path,
    cache_dir: Path,
    validation_fraction: float,
    seed: int,
    refresh_cache: bool,
) -> tuple[list[MixedRecord], list[MixedRecord], list[str]]:
    protocol = pa_root / PROTOCOL_BY_SPLIT["train"]
    all_train = read_pa_cm_protocol(protocol)
    train_pool, val_pool = pa_split_by_speaker(all_train, validation_fraction, seed)

    import random

    rng = random.Random(seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    readable_train, skipped_train = filter_readable_records(
        pa_root,
        "train",
        train_pool,
        cache_path=cache_dir / "pa2019_train_readable.json",
        force_refresh=refresh_cache,
        target_count=0,
    )
    readable_val, skipped_val = filter_readable_records(
        pa_root,
        "train",
        val_pool,
        cache_path=cache_dir / "pa2019_train_readable.json",
        force_refresh=False,
        target_count=0,
    )
    skipped = sorted(set(skipped_train) | set(skipped_val))

    def to_mixed(recs) -> list[MixedRecord]:
        out: list[MixedRecord] = []
        for r in recs:
            path = resolve_audio_path(pa_root, "train", r.utt_id)
            out.append(
                MixedRecord(
                    corpus="pa2019",
                    utt_id=r.utt_id,
                    speaker_id=r.speaker_id,
                    label=r.label,
                    split="train",
                    path=path,
                )
            )
        return out

    return to_mixed(readable_train), to_mixed(readable_val), skipped


def load_pa_split(
    pa_root: Path,
    split: str,
    cache_dir: Path,
    refresh_cache: bool,
) -> tuple[list[MixedRecord], list[str]]:
    protocol = pa_root / PROTOCOL_BY_SPLIT[split]
    records = read_pa_cm_protocol(protocol)
    readable, skipped = filter_readable_records(
        pa_root,
        split,
        records,
        cache_path=cache_dir / f"pa2019_{split}_readable.json",
        force_refresh=refresh_cache,
        target_count=0,
    )
    out: list[MixedRecord] = []
    for r in readable:
        path = resolve_audio_path(pa_root, split, r.utt_id)
        out.append(
            MixedRecord(
                corpus="pa2019",
                utt_id=r.utt_id,
                speaker_id=r.speaker_id,
                label=r.label,
                split=split,
                path=path,
            )
        )
    return out, skipped
