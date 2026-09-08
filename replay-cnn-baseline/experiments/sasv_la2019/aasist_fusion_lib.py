"""ECAPA + AASIST score-sum fusion for SASV trials.

Reuses saved ECAPA ``s_asv`` from LFCC fusion CSVs when present, and scores
AASIST only on unique test utterances (cached).
"""

from __future__ import annotations

import csv
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from experiment_lib import (
    DEFAULT_LA,
    DEFAULT_SASV,
    RUNS_DIR,
    _REPO_ROOT,
    ensure_sasv_on_path,
    resolve_audio_path,
)

DEFAULT_AASIST = _REPO_ROOT / "aasist"
AASIST_CUT = 64600  # ~4 s @ 16 kHz (official AASIST)


def ensure_aasist_on_path(aasist_root: Path | None = None) -> Path:
    root = Path(aasist_root or DEFAULT_AASIST)
    if not (root / "models" / "AASIST.py").exists():
        raise FileNotFoundError(
            f"AASIST repo not found at {root}. "
            "Clone https://github.com/clovaai/aasist"
        )
    sys.path.insert(0, str(root.resolve()))
    return root


def pad_aasist(x: np.ndarray, max_len: int = AASIST_CUT) -> np.ndarray:
    x_len = int(x.shape[0])
    if x_len >= max_len:
        return x[:max_len]
    n_rep = int(max_len / max(x_len, 1)) + 1
    return np.tile(x, n_rep)[:max_len]


def load_aasist_model(
    *,
    aasist_root: Path | None = None,
    config_path: Path | None = None,
    device: str = "cuda",
):
    """Load pretrained AASIST (bonafide logit = class index 1)."""
    root = ensure_aasist_on_path(aasist_root)
    cfg_path = Path(config_path or (root / "config" / "AASIST.conf"))
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_cfg = config["model_config"]
    weight = root / config["model_path"]
    if not weight.exists():
        weight = root / "models" / "weights" / "AASIST.pth"
    if not weight.exists():
        raise FileNotFoundError(f"AASIST weights missing: {weight}")

    module = import_module(f"models.{model_cfg['architecture']}")
    model = module.Model(model_cfg)
    state = torch.load(str(weight), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


@torch.inference_mode()
def score_aasist_utt(
    model,
    wave_1d: np.ndarray,
    device: str,
) -> tuple[float, float]:
    """Return ``(s_cm, p_spoof)`` with ``s_cm = P(bonafide)`` via softmax."""
    x = pad_aasist(np.asarray(wave_1d, dtype=np.float32))
    batch = torch.from_numpy(x).float().unsqueeze(0).to(device)
    _, logits = model(batch)
    probs = F.softmax(logits, dim=1)
    p_bona = float(probs[0, 1].item())
    p_spoof = float(probs[0, 0].item())
    return p_bona, p_spoof


def load_ecapa_asv_csv(path: Path) -> list[dict]:
    """Load rows with at least ``s_asv``, ``test_utt``, ``key``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ECAPA score CSV: {path}. Run notebook 03/04 (LFCC) first "
            "so s_asv is available, or set RECOMPUTE_ECAPA=True in the notebook."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "speaker_id": row.get("speaker_id", ""),
                    "test_utt": row["test_utt"],
                    "key": str(row["key"]).lower(),
                    "s_asv": float(row["s_asv"]),
                }
            )
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def stratified_rows(rows: list[dict], max_trials: int, seed: int = 42) -> list[dict]:
    if max_trials <= 0 or max_trials >= len(rows):
        return rows
    import random

    by_key: dict[str, list[dict]] = {"target": [], "nontarget": [], "spoof": []}
    for row in rows:
        by_key.setdefault(row["key"], []).append(row)
    rng = random.Random(seed)
    for key in by_key:
        rng.shuffle(by_key[key])
    keys = [k for k in ("target", "nontarget", "spoof") if by_key.get(k)]
    if not keys:
        return rows[:max_trials]
    per = max(1, max_trials // len(keys))
    out: list[dict] = []
    for key in keys:
        out.extend(by_key[key][:per])
    rng.shuffle(out)
    return out[:max_trials]


def score_ecapa_aasist_fusion(
    *,
    split: str = "dev",
    max_trials: int = 0,
    device: str = "cuda",
    force_cpu: bool = False,
    la_root: Path | None = None,
    sasv_root: Path | None = None,
    aasist_root: Path | None = None,
    ecapa_csv: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Fuse reused ECAPA cosine with AASIST P(bonafide)."""
    la_root = Path(la_root or DEFAULT_LA)
    sasv_root = ensure_sasv_on_path(sasv_root or DEFAULT_SASV)
    from metrics import get_all_EERs

    if force_cpu:
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    csv_path = Path(
        ecapa_csv
        or (RUNS_DIR / f"ecapa_plus_lfcc_{split}" / f"scores_{split}.csv")
    )
    rows = load_ecapa_asv_csv(csv_path)
    rows = stratified_rows(rows, max_trials)
    counts = {"target": 0, "nontarget": 0, "spoof": 0, "total": len(rows)}
    for row in rows:
        counts[row["key"]] = counts.get(row["key"], 0) + 1
    print(f"Trials: {counts} | reuse s_asv from {csv_path.name}")

    model, _cfg = load_aasist_model(aasist_root=aasist_root, device=device)
    print(f"AASIST on {device}")

    utt_ids = sorted({row["test_utt"] for row in rows})
    cm_cache: dict[str, tuple[float, float]] = {}
    for utt in tqdm(utt_ids, desc="AASIST utts"):
        path = resolve_audio_path(la_root, split, utt)
        wave, _sr = sf.read(str(path))
        if getattr(wave, "ndim", 1) > 1:
            wave = wave.mean(axis=1)
        s_cm, p_spoof = score_aasist_utt(model, wave, device)
        cm_cache[utt] = (s_cm, p_spoof)

    preds: list[float] = []
    keys: list[str] = []
    out_rows: list[dict] = []
    for row in rows:
        s_asv = float(row["s_asv"])
        s_cm, p_spoof = cm_cache[row["test_utt"]]
        score = s_asv + s_cm  # B1-v2-ish: cosine + bounded CM
        preds.append(score)
        keys.append(row["key"])
        out_rows.append(
            {
                "speaker_id": row["speaker_id"],
                "test_utt": row["test_utt"],
                "key": row["key"],
                "s_asv": s_asv,
                "s_cm": s_cm,
                "p_spoof": p_spoof,
                "score": score,
            }
        )

    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    summary = {
        "system": "ecapa_plus_aasist_sum",
        "split": split,
        "max_trials": max_trials,
        "num_scored": len(preds),
        "key_counts": counts,
        "device": device,
        "cm_backend": "aasist",
        "fusion": "s_asv + P_bona(AASIST softmax)",
        "ecapa_csv": str(csv_path),
        "num_unique_test_utts": len(utt_ids),
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }

    output_dir = Path(output_dir or (RUNS_DIR / f"ecapa_plus_aasist_{split}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"scores_{split}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker_id",
                "test_utt",
                "key",
                "s_asv",
                "s_cm",
                "p_spoof",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary
