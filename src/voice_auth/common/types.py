"""Shared type aliases and constants for the voice authentication system."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Audio conventions
DEFAULT_SAMPLE_RATE: int = 16_000
AUDIO_DTYPE = np.float32

# Embedding / quality defaults
DEFAULT_EMBEDDING_DIM: int = 192
DEFAULT_QUALITY_DIM: int = 5

# Replay labels: 0 = bona fide, 1 = replay
REPLAY_LABEL_BONA_FIDE: int = 0
REPLAY_LABEL_REPLAY: int = 1

Waveform: TypeAlias = NDArray[np.floating]
Embedding: TypeAlias = NDArray[np.floating]
QualityVector: TypeAlias = NDArray[np.floating]
