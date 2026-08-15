"""Configuration for the ECAPA Python ML server."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "4.0"))
# Cosine ACCEPT threshold (raised so silence/noise rarely clears speaker verify)
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.45"))
# Silero VAD preprocessing (before replay + ECAPA)
VAD_ENABLED = os.getenv("VAD_ENABLED", "true").lower() in {"1", "true", "yes"}
VAD_USE_ONNX = os.getenv("VAD_USE_ONNX", "true").lower() in {"1", "true", "yes"}
VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.50"))
VAD_MIN_AUDIO_MS = float(os.getenv("VAD_MIN_AUDIO_MS", "250"))
VAD_MIN_SPEECH_MS = float(os.getenv("VAD_MIN_SPEECH_MS", "120"))
VAD_MIN_SILENCE_MS = float(os.getenv("VAD_MIN_SILENCE_MS", "120"))
VAD_SPEECH_PAD_MS = float(os.getenv("VAD_SPEECH_PAD_MS", "30"))
VAD_MIN_TOTAL_SPEECH_MS = float(os.getenv("VAD_MIN_TOTAL_SPEECH_MS", "300"))
# Frame-level speech gate — require sustained louder frames (rejects ambient mic)
# Defaults tuned on laptop verify WAVs that false-ACCEPTed without speech.
MIN_SPEECH_FRAME_RMS = float(os.getenv("MIN_SPEECH_FRAME_RMS", "0.08"))
MIN_SPEECH_MS = float(os.getenv("MIN_SPEECH_MS", "350"))
MIN_SPEECH_STRONG_FRAME_RMS = float(os.getenv("MIN_SPEECH_STRONG_FRAME_RMS", "0.10"))
MIN_SPEECH_STRONG_MS = float(os.getenv("MIN_SPEECH_STRONG_MS", "200"))
MIN_SPEECH_PEAK = float(os.getenv("MIN_SPEECH_PEAK", "0.40"))
# Legacy whole-clip floor (very quiet clips fail fast)
MIN_SPEECH_RMS = float(os.getenv("MIN_SPEECH_RMS", "0.01"))
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

# Inverted-Mel replay CNN (mixed ASVspoof 2017 + PA2019).
# Prefer this for the app: LFCC wins ASVspoof tables but over-fires REPLAY on browser mics.
_REPO_ROOT = _ROOT.parent.parent  # app/server -> repo root
_DEFAULT_REPLAY_CKPT = (
    _REPO_ROOT
    / "replay-cnn-baseline"
    / "experiments"
    / "inverted_mel_mixed_2017_pa2019"
    / "runs"
    / "inverted_mel_mixed"
    / "best_inverted_mel_mixed_2017_pa2019.pt"
)
REPLAY_ENABLED = os.getenv("REPLAY_ENABLED", "true").lower() in {"1", "true", "yes"}
REPLAY_CHECKPOINT = Path(
    os.getenv("REPLAY_CHECKPOINT", str(_DEFAULT_REPLAY_CKPT))
)
_raw_replay_thr = os.getenv("REPLAY_THRESHOLD", "").strip()
REPLAY_THRESHOLD = float(_raw_replay_thr) if _raw_replay_thr else None

# Dual-threshold band: LIVE < t_low <= UNCERTAIN < t_high <= REPLAY
# Default: center at checkpoint EER thr ± REPLAY_MARGIN
_raw_margin = os.getenv("REPLAY_MARGIN", "0.10").strip()
REPLAY_MARGIN = float(_raw_margin) if _raw_margin else 0.10
_raw_t_low = os.getenv("REPLAY_T_LOW", "").strip()
_raw_t_high = os.getenv("REPLAY_T_HIGH", "").strip()
REPLAY_T_LOW = float(_raw_t_low) if _raw_t_low else None
REPLAY_T_HIGH = float(_raw_t_high) if _raw_t_high else None

# ASVspoof 2019 LA (synthetic) LFCC CNN — second stage after inverted-Mel replay.
_DEFAULT_LA_CKPT = (
    _REPO_ROOT
    / "replay-cnn-baseline"
    / "experiments"
    / "lfcc_la2019"
    / "runs"
    / "lfcc_la"
    / "best_lfcc_la2019.pt"
)
# Off by default: LFCC-LA is strong on ASVspoof LA but scores ~1.0 on browser-mic
# speech (domain mismatch), same failure mode as mixed-LFCC replay earlier.
LA_ENABLED = os.getenv("LA_ENABLED", "false").lower() in {"1", "true", "yes"}
# When enabled, scores are always returned; only hard-block if this is true
# (use for lab ASVspoof files — unsafe for laptop/browser verify).
LA_HARD_GATE = os.getenv("LA_HARD_GATE", "false").lower() in {"1", "true", "yes"}
LA_CHECKPOINT = Path(os.getenv("LA_CHECKPOINT", str(_DEFAULT_LA_CKPT)))
_raw_la_thr = os.getenv("LA_THRESHOLD", "").strip()
LA_THRESHOLD = float(_raw_la_thr) if _raw_la_thr else None
_raw_la_margin = os.getenv("LA_MARGIN", "0.10").strip()
LA_MARGIN = float(_raw_la_margin) if _raw_la_margin else 0.10
_raw_la_t_low = os.getenv("LA_T_LOW", "").strip()
_raw_la_t_high = os.getenv("LA_T_HIGH", "").strip()
LA_T_LOW = float(_raw_la_t_low) if _raw_la_t_low else None
LA_T_HIGH = float(_raw_la_t_high) if _raw_la_t_high else None
