"""ASVspoof-style dataset placeholder (no downloads)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_auth.common.types import REPLAY_LABEL_BONA_FIDE, REPLAY_LABEL_REPLAY


@dataclass(frozen=True)
class ReplaySample:
    """One labeled audio sample for replay detection."""

    utt_id: str
    audio_path: Path
    label: int  # 0 = bona fide, 1 = replay
    split: str  # train / dev / eval


class ReplayDatasetPlaceholder:
    """
    Lightweight dataset stub for ASVspoof 2017 V2 protocol files.

    Does not load audio. Member 2 should wire actual FLAC/WAV reading and
    protocol parsing once data is available under ``data/``.
    """

    def __init__(self, root: Path, protocol_file: Path | None = None) -> None:
        self.root = Path(root)
        self.protocol_file = Path(protocol_file) if protocol_file else None
        self.samples: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def load_protocol(self) -> list[ReplaySample]:
        """
        Parse a protocol file if present; otherwise return an empty list.

        Expected whitespace-separated columns (ASVspoof-like):
        ``utt_id path label`` with label in {bonafide, replay} or {0, 1}.
        """
        if self.protocol_file is None or not self.protocol_file.exists():
            self.samples = []
            return self.samples

        samples: list[ReplaySample] = []
        for line in self.protocol_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            utt_id, rel_path, raw_label = parts[0], parts[1], parts[2]
            label = _parse_label(raw_label)
            samples.append(
                ReplaySample(
                    utt_id=utt_id,
                    audio_path=self.root / rel_path,
                    label=label,
                    split="unknown",
                )
            )
        self.samples = samples
        return samples


def _parse_label(raw: str) -> int:
    key = raw.strip().lower()
    if key in {"0", "bonafide", "bona-fide", "genuine"}:
        return REPLAY_LABEL_BONA_FIDE
    if key in {"1", "replay", "spoof"}:
        return REPLAY_LABEL_REPLAY
    raise ValueError(f"Unrecognized replay label: {raw!r}")
