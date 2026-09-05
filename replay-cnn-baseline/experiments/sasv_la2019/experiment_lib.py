"""Paths and helpers for SASV-style eval on ASVspoof 2019 LA."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR
for _ in range(6):
    if (_REPO_ROOT / "data" / "LA").exists() and (_REPO_ROOT / "SASVC2022_Baseline").exists():
        break
    if (_REPO_ROOT / "data" / "LA").exists() and (_REPO_ROOT / "app" / "server").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

DEFAULT_LA = _REPO_ROOT / "data" / "LA"
DEFAULT_SASV = _REPO_ROOT / "SASVC2022_Baseline"
DEFAULT_SERVER = _REPO_ROOT / "app" / "server"
RUNS_DIR = _THIS_DIR / "runs"
CACHE_DIR = _THIS_DIR / "cache"

AUDIO_DIR = {
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
    "train": "ASVspoof2019_LA_train/flac",
}

TRIAL_PROTOCOL = {
    "dev": "protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt",
}

ENROLL_PROTOCOLS = {
    "dev": [
        "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
        "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt",
    ],
    "eval": [
        "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
        "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt",
    ],
}


@dataclass(frozen=True)
class SasvTrial:
    speaker_id: str
    test_utt: str
    attack_or_type: str
    key: str  # target | nontarget | spoof


def ensure_sasv_on_path(sasv_root: Path | None = None) -> Path:
    root = Path(sasv_root or DEFAULT_SASV)
    if not root.exists():
        raise FileNotFoundError(
            f"SASVC2022_Baseline not found at {root}. "
            "Clone https://github.com/sasv-challenge/SASVC2022_Baseline"
        )
    sys.path.insert(0, str(root.resolve()))
    return root


def ensure_server_on_path(server_root: Path | None = None) -> Path:
    root = Path(server_root or DEFAULT_SERVER)
    if not (root / "ml_server").exists():
        raise FileNotFoundError(f"ML server not found at {root}")
    sys.path.insert(0, str(root.resolve()))
    return root


def resolve_audio_path(la_root: Path, split: str, utt_id: str) -> Path:
    flac = Path(la_root) / AUDIO_DIR[split] / f"{utt_id}.flac"
    wav = flac.with_suffix(".wav")
    if flac.exists():
        return flac
    if wav.exists():
        return wav
    raise FileNotFoundError(f"Audio not found for {utt_id}: tried {flac}")


def load_waveform(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load mono float32 waveform (full clip, no 4s crop)."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(wave).mean(dim=1)
    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)
    if wave.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    return wave


def read_enroll_map(la_root: Path, split: str) -> dict[str, list[str]]:
    """speaker_id -> list of enrolment utterance ids."""
    mapping: dict[str, list[str]] = {}
    for rel in ENROLL_PROTOCOLS[split]:
        path = Path(la_root) / rel
        if not path.exists():
            raise FileNotFoundError(f"Enrolment protocol missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                spk, utts = parts[0], parts[1].split(",")
                mapping.setdefault(spk, []).extend(utts)
    return mapping


def read_trials(
    sasv_root: Path,
    split: str,
    max_trials: int = 0,
    seed: int = 42,
) -> list[SasvTrial]:
    """Load SASV trials.

    If ``max_trials > 0``, take a **stratified** subset (target / nontarget / spoof).
    Taking the first N lines is wrong: the file is grouped and starts with targets only.
    """
    import random

    path = Path(sasv_root) / TRIAL_PROTOCOL[split]
    if not path.exists():
        raise FileNotFoundError(f"Trial protocol missing: {path}")
    trials: list[SasvTrial] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            key = parts[-1].lower()
            if key not in {"target", "nontarget", "spoof"}:
                raise ValueError(f"Unexpected key {key!r} in {path}")
            trials.append(
                SasvTrial(
                    speaker_id=parts[0],
                    test_utt=parts[1],
                    attack_or_type=parts[2],
                    key=key,
                )
            )

    if max_trials <= 0 or max_trials >= len(trials):
        return trials

    by_key: dict[str, list[SasvTrial]] = {"target": [], "nontarget": [], "spoof": []}
    for trial in trials:
        by_key[trial.key].append(trial)

    rng = random.Random(seed)
    # Proportional allocation, at least 1 per class when available
    total = len(trials)
    planned: dict[str, int] = {}
    for key, group in by_key.items():
        if not group:
            planned[key] = 0
            continue
        n = max(1, int(round(max_trials * len(group) / total)))
        planned[key] = min(len(group), n)

    # Fix rounding so we hit max_trials as closely as possible
    while sum(planned.values()) > max_trials:
        for key in ("spoof", "nontarget", "target"):
            if planned[key] > 1 and sum(planned.values()) > max_trials:
                planned[key] -= 1
    while sum(planned.values()) < max_trials:
        progressed = False
        for key in ("spoof", "nontarget", "target"):
            if planned[key] < len(by_key[key]) and sum(planned.values()) < max_trials:
                planned[key] += 1
                progressed = True
        if not progressed:
            break

    out: list[SasvTrial] = []
    for key, group in by_key.items():
        out.extend(rng.sample(group, planned[key]))
    rng.shuffle(out)
    return out


def trial_key_counts(trials: list[SasvTrial]) -> dict[str, int]:
    counts = {"target": 0, "nontarget": 0, "spoof": 0, "total": len(trials)}
    for trial in trials:
        counts[trial.key] = counts.get(trial.key, 0) + 1
    return counts


def load_spk_meta(sasv_root: Path, split: str) -> dict:
    name = {"dev": "spk_meta_dev.pk", "eval": "spk_meta_eval.pk", "train": "spk_meta_trn.pk"}
    path = Path(sasv_root) / "spk_meta" / name[split]
    with path.open("rb") as handle:
        return pickle.load(handle)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
