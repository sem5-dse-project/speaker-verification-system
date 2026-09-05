"""Weighted / calibrated fusion helpers for SASV score CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiment_lib import ensure_sasv_on_path, RUNS_DIR


def load_score_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load ``s_asv``, ``s_cm``, ``key`` columns from a fused scores CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing scores CSV: {path}. Run notebook 03/04 first (LFCC fusion)."
        )
    s_asv: list[float] = []
    s_cm: list[float] = []
    keys: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            s_asv.append(float(row["s_asv"]))
            s_cm.append(float(row["s_cm"]))
            keys.append(str(row["key"]).lower())
    if not keys:
        raise ValueError(f"Empty scores CSV: {path}")
    return np.asarray(s_asv, dtype=np.float64), np.asarray(s_cm, dtype=np.float64), keys


def weighted_scores(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """``score = α * s_asv + (1 - α) * s_cm``."""
    a = float(alpha)
    return a * s_asv + (1.0 - a) * s_cm


def eers_for_alpha(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    keys: list[str],
    alpha: float,
    sasv_root: Path | None = None,
) -> dict[str, float]:
    ensure_sasv_on_path(sasv_root)
    from metrics import get_all_EERs

    preds = weighted_scores(s_asv, s_cm, alpha).tolist()
    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    return {
        "alpha": float(alpha),
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }


def sweep_alpha(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    keys: list[str],
    *,
    alphas: np.ndarray | None = None,
    sasv_root: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Sweep α on a split; return best-by-SASV-EER row and full table."""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        rows.append(eers_for_alpha(s_asv, s_cm, keys, float(alpha), sasv_root))
    best = min(rows, key=lambda r: (r["sasv_eer"], r["sv_eer"], r["spf_eer"]))
    return best, rows


def save_weighted_run(
    *,
    split: str,
    alpha: float,
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    keys: list[str],
    metrics: dict,
    output_dir: Path | None = None,
    source_csv: Path | None = None,
) -> Path:
    """Write weighted scores + metrics under ``runs/ecapa_plus_lfcc_weighted_<split>/``."""
    out = Path(output_dir or (RUNS_DIR / f"ecapa_plus_lfcc_weighted_{split}"))
    out.mkdir(parents=True, exist_ok=True)
    scores = weighted_scores(s_asv, s_cm, alpha)
    score_path = out / f"scores_{split}.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "s_asv", "s_cm", "score", "alpha"],
        )
        writer.writeheader()
        for i, key in enumerate(keys):
            writer.writerow(
                {
                    "key": key,
                    "s_asv": float(s_asv[i]),
                    "s_cm": float(s_cm[i]),
                    "score": float(scores[i]),
                    "alpha": float(alpha),
                }
            )
    payload = {
        **metrics,
        "system": "ecapa_plus_lfcc_weighted",
        "split": split,
        "fusion": "alpha * s_asv + (1 - alpha) * s_cm",
        "alpha": float(alpha),
        "num_scored": len(keys),
        "source_csv": str(source_csv) if source_csv else None,
    }
    (out / f"metrics_{split}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out
