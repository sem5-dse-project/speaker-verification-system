"""
Noise-aware gated SASV helpers.

Builds on ``sasv_la2019`` (protocol, ECAPA, metrics) and adds:
  - test-side additive noise at fixed SNRs
  - raw / enhanced ECAPA fusion (linear SNR gate or always-enhance)
  - CSV scoring harness for B0 / B1 / P2 / P3 style systems

Paper rule: tune on **dev**, lock and report **eval** once.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch

_THIS = Path(__file__).resolve().parent
_SASV_LA = _THIS.parent / "sasv_la2019"
if str(_SASV_LA) not in sys.path:
    sys.path.insert(0, str(_SASV_LA))

from experiment_lib import (  # noqa: E402
    DEFAULT_LA,
    DEFAULT_SASV,
    DEFAULT_SERVER,
    RUNS_DIR as SASV_RUNS,
    cosine,
    ensure_sasv_on_path,
    ensure_server_on_path,
    l2_normalize,
    load_waveform,
    patch_speechbrain_windows_lazy_import,
    read_enroll_map,
    read_trials,
    resolve_audio_path,
    trial_key_counts,
)
from score_lib import (  # noqa: E402
    build_speaker_models,
    embed_utt,
    load_app_ecapa,
)

RUNS_DIR = _THIS / "runs"
CACHE_DIR = _THIS / "cache"
NOISY_DIR = CACHE_DIR / "noisy_wavs"

DEFAULT_SNRS_DB = (None, 15, 10, 5, 0)  # None = clean
DEFAULT_ALPHA_AASIST = 0.30  # locked from sasv_la2019 notebook 11


def snr_tag(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"snr{int(snr_db)}db"


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_DIR.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def add_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix ``noise`` into ``clean`` at target SNR (dB). Both 1-D float arrays."""
    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if noise.size < clean.size:
        reps = int(np.ceil(clean.size / max(noise.size, 1)))
        noise = np.tile(noise, reps)
    noise = noise[: clean.size]
    c_rms = rms(clean)
    n_rms = rms(noise)
    if n_rms < 1e-8:
        return clean.copy()
    target_n = c_rms / (10 ** (float(snr_db) / 20.0))
    mixed = clean + noise * (target_n / n_rms)
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def load_noise_bank(
    noise_root: Path | None,
    *,
    max_files: int = 64,
    sr: int = 16000,
) -> list[np.ndarray]:
    """Load a small bank of noise waveforms (MUSAN/RIRS/etc.)."""
    if noise_root is None or not Path(noise_root).exists():
        return []
    root = Path(noise_root)
    paths = sorted(
        list(root.rglob("*.wav")) + list(root.rglob("*.flac")) + list(root.rglob("*.ogg"))
    )[:max_files]
    bank: list[np.ndarray] = []
    for path in paths:
        try:
            audio, file_sr = sf.read(path, always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = np.mean(audio, axis=-1)
            audio = np.asarray(audio, dtype=np.float32)
            if int(file_sr) != sr:
                # Lightweight resample via torchaudio if available
                wav = torch.from_numpy(audio).unsqueeze(0)
                wav = torchaudio_resample(wav, int(file_sr), sr)
                audio = wav.squeeze(0).numpy().astype(np.float32)
            if audio.size > sr:  # keep at least 1s
                bank.append(audio)
        except Exception:
            continue
    return bank


def torchaudio_resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    import torchaudio

    if orig_sr == new_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, new_sr)


def synthetic_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fallback white noise if MUSAN is not on disk."""
    return rng.standard_normal(n).astype(np.float32)


def maybe_noise_waveform(
    clean: torch.Tensor,
    *,
    snr_db: float | None,
    noise_bank: list[np.ndarray],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Return clean or noisy 1-D waveform tensor @ 16 kHz."""
    if snr_db is None:
        return clean
    x = clean.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if noise_bank:
        noise = noise_bank[int(rng.integers(0, len(noise_bank)))]
    else:
        noise = synthetic_noise(x.size, rng)
    y = add_noise_at_snr(x, noise, float(snr_db))
    return torch.from_numpy(y)


def snr_gate_weight(snr_db: float | None, *, mid_db: float = 8.0, sharp: float = 0.35) -> float:
    """
    Scalar gate in (0, 1): higher → trust **enhanced** more.

    At low SNR, prefer enhancement; at high SNR / clean, prefer raw.
    ``w_enhanced = sigmoid(sharp * (mid_db - snr))`` with clean treated as +30 dB.
    """
    snr = 30.0 if snr_db is None else float(snr_db)
    w_enh = 1.0 / (1.0 + np.exp(-sharp * (mid_db - snr)))
    return float(np.clip(w_enh, 0.05, 0.95))


def fuse_embeddings(
    emb_raw: np.ndarray,
    emb_enh: np.ndarray,
    *,
    mode: str,
    snr_db: float | None,
) -> np.ndarray:
    """
    mode:
      - ``gated``: SNR sigmoid blend (P2)
      - ``always_enhance``: use enhanced only (P3)
      - ``raw``: use raw only
    """
    mode = (mode or "gated").lower()
    if mode == "raw":
        return l2_normalize(emb_raw)
    if mode == "always_enhance":
        return l2_normalize(emb_enh)
    w = snr_gate_weight(snr_db)
    fused = (1.0 - w) * emb_raw + w * emb_enh
    return l2_normalize(fused)


def write_score_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def eers_from_preds(preds: list[float], keys: list[str], sasv_root: Path | None = None) -> dict:
    ensure_sasv_on_path(sasv_root)
    from metrics import get_all_EERs

    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    return {
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }


def save_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_noise_root() -> Path | None:
    """Try common MUSAN / noise locations (override with NOISE_ROOT env)."""
    import os

    env = os.environ.get("NOISE_ROOT", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    repo = Path(__file__).resolve().parents[3]  # speaker-verification-system
    candidates.extend(
        [
            Path(r"D:/downloads/musan/musan/noise"),
            Path(r"D:/downloads/musan/musan"),
            repo / "data" / "musan" / "noise",
            repo / "data" / "musan",
            repo / "data" / "rirs_noises" / "pointsource_noises",
            Path(r"D:/data/musan/noise"),
            Path(r"D:/musan/noise"),
            Path(r"D:/datasets/musan/noise"),
            Path.home() / "data" / "musan" / "noise",
            Path.home() / "musan" / "noise",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_dir():
            # Prefer a folder that actually contains audio
            if any(path.rglob("*.wav")) or any(path.rglob("*.flac")):
                return path
    return None


def load_enhancer(device: str = "cpu"):
    """Load Wave-U-Net enhancer from app/server if checkpoint exists; else None."""
    import os

    ensure_server_on_path()
    server_root = Path(DEFAULT_SERVER)
    try:
        from dotenv import load_dotenv

        load_dotenv(server_root / ".env", override=False)
    except Exception:
        pass

    ckpt = Path(
        os.environ.get(
            "WAVEUNET_CHECKPOINT",
            str(server_root / "checkpoints" / "waveunet_finetuned_v4_best.pt"),
        )
    )
    if not ckpt.is_absolute():
        ckpt = (server_root / ckpt).resolve()
    if not ckpt.exists():
        print(f"[noise_gated_lib] WAVEUNET checkpoint missing: {ckpt}")
        return None

    try:
        from ml_server.enhancement import get_enhancer

        enhancer = get_enhancer(
            mode="waveunet",
            checkpoint_path=str(ckpt),
            device=device,
        )
        print(f"[noise_gated_lib] enhancer ok: {type(enhancer).__name__} ckpt={ckpt}")
        return enhancer
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[noise_gated_lib] enhancer unavailable: {exc}")
        print(
            "[noise_gated_lib] hint: pip install denoisers  "
            "(in app/server/.venv) and ensure WAVEUNET checkpoint exists"
        )
        return None


def resolve_noise_bank(noise_root: Path | str | None = None, *, max_files: int = 64):
    """Load noise bank from explicit path or auto-discovery; print what was used."""
    root: Path | None
    if noise_root is not None and str(noise_root).strip():
        root = Path(noise_root)
    else:
        root = discover_noise_root()
    bank = load_noise_bank(root, max_files=max_files) if root else []
    if root and bank:
        print(f"[noise] using {len(bank)} clips from {root}")
    else:
        print("[noise] MUSAN not found — using seeded white-noise fallback")
    return bank, root



@torch.inference_mode()
def embed_waveform(classifier, wave: torch.Tensor, device: str) -> np.ndarray:
    from ml_server.ecapa import encode_waveforms

    batch = wave.reshape(1, -1).to(device)
    emb = encode_waveforms(classifier, batch).squeeze(0).cpu().numpy().astype(np.float32)
    return emb


def protocol_summary() -> dict:
    return {
        "corpus": "ASVspoof 2019 LA",
        "trials": "SASV 2022 official GI trial lists",
        "tune_on": "dev",
        "report_on": "eval (once, locked)",
        "metrics": ["SASV-EER", "SV-EER", "SPF-EER"],
        "systems": {
            "B0": "ECAPA cosine only",
            "B1": f"ECAPA + AASIST weighted α={DEFAULT_ALPHA_AASIST}",
            "B2": "ECAPA + LFCC score-sum (optional)",
            "P2": "Wave-U-Net + SNR-gated emb fusion + CM",
            "P3": "Wave-U-Net always-enhance + CM",
        },
        "snrs_db": [snr_tag(s) for s in DEFAULT_SNRS_DB],
        "paths": {
            "la": str(DEFAULT_LA),
            "sasv": str(DEFAULT_SASV),
            "server": str(DEFAULT_SERVER),
            "runs": str(RUNS_DIR),
            "sasv_la2019_runs": str(SASV_RUNS),
        },
    }


__all__ = [
    "CACHE_DIR",
    "DEFAULT_ALPHA_AASIST",
    "DEFAULT_LA",
    "DEFAULT_SASV",
    "DEFAULT_SNRS_DB",
    "NOISY_DIR",
    "RUNS_DIR",
    "add_noise_at_snr",
    "build_speaker_models",
    "cosine",
    "eers_from_preds",
    "embed_utt",
    "embed_waveform",
    "ensure_dirs",
    "ensure_sasv_on_path",
    "ensure_server_on_path",
    "fuse_embeddings",
    "load_app_ecapa",
    "load_enhancer",
    "load_noise_bank",
    "discover_noise_root",
    "resolve_noise_bank",
    "load_waveform",
    "maybe_noise_waveform",
    "patch_speechbrain_windows_lazy_import",
    "protocol_summary",
    "read_enroll_map",
    "read_trials",
    "resolve_audio_path",
    "save_json",
    "snr_gate_weight",
    "snr_tag",
    "trial_key_counts",
    "write_score_csv",
]
