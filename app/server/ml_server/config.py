"""Configuration for the ECAPA Python ML server."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "4.0"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.25"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cpu")  # cpu | cuda
ECAPA_SOURCE = os.getenv("ECAPA_SOURCE", "speechbrain/spkrec-ecapa-voxceleb")
ECAPA_SAVEDIR = Path(
    os.getenv(
        "ECAPA_SAVEDIR",
        str(_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    )
)
