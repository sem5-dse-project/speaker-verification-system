"""AASIST+LFCC CM ensemble helpers for SASV score CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiment_lib import RUNS_DIR, ensure_sasv_on_path
from weighted_fusion_lib import eers_for_alpha, weighted_scores


def load_aligned_cm_csvs(
    aasist_csv: Path,
    lfcc_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align two fusion CSVs on (speaker_id, test_utt, key).

    Returns ``s_asv``, ``s_cm_aasist``, ``s_cm_lfcc``, ``keys``.
    Uses AASIST file's ``s_asv`` (same ECAPA scores as LFCC runs).
    """
    aasist_csv = Path(aasist_csv)
    lfcc_csv = Path(lfcc_csv)
    if not aasist_csv.exists():
        raise FileNotFoundError(f"Missing AASIST scores: {aasist_csv}")
    if not lfcc_csv.exists():
        raise FileNotFoundError(f"Missing LFCC scores: {lfcc_csv}")

    lfcc_cm: dict[tuple[str, str, str], float] = {}
    with lfcc_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row.get("speaker_id", ""),
                row["test_utt"],
                str(row["key"]).lower(),
            )
            lfcc_cm[key] = float(row["s_cm"])

    s_asv: list[float] = []
    s_aasist: list[float] = []
    s_lfcc: list[float] = []
    keys: list[str] = []
    missing = 0
    with aasist_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row.get("speaker_id", ""),
                row["test_utt"],
                str(row["key"]).lower(),
            )
            if key not in lfcc_cm:
                missing += 1
                continue
            s_asv.append(float(row["s_asv"]))
            s_aasist.append(float(row["s_cm"]))
            s_lfcc.append(lfcc_cm[key])
            keys.append(key[2])

    if not keys:
        raise ValueError("No overlapping trials between AASIST and LFCC CSVs")
    if missing:
        print(f"Warning: skipped {missing} AASIST rows with no LFCC match")

    return (
        np.asarray(s_asv, dtype=np.float64),
        np.asarray(s_aasist, dtype=np.float64),
        np.asarray(s_lfcc, dtype=np.float64),
        keys,
    )


def ensemble_cm(s_aasist: np.ndarray, s_lfcc: np.ndarray, beta: float) -> np.ndarray:
    """``s_cm = β · s_cm_aasist + (1 − β) · s_cm_lfcc``."""
    b = float(beta)
    return b * s_aasist + (1.0 - b) * s_lfcc


def fused_ensemble_score(
    s_asv: np.ndarray,
    s_aasist: np.ndarray,
    s_lfcc: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    mode: str = "sum",
) -> np.ndarray:
    """Fuse ASV with ensembled CM.

    - ``mode="sum"``: ``s_asv + s_cm`` (same family as notebooks 03/09)
    - ``mode="weighted"``: ``α · s_asv + (1 − α) · s_cm``
    """
    s_cm = ensemble_cm(s_aasist, s_lfcc, beta)
    if mode == "sum":
        return s_asv + s_cm
    if mode == "weighted":
        return weighted_scores(s_asv, s_cm, alpha)
    raise ValueError(f"Unknown mode={mode!r}")


def eers_for_ensemble(
    s_asv: np.ndarray,
    s_aasist: np.ndarray,
    s_lfcc: np.ndarray,
    keys: list[str],
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    mode: str = "sum",
    sasv_root: Path | None = None,
) -> dict[str, float]:
    ensure_sasv_on_path(sasv_root)
    from metrics import get_all_EERs

    preds = fused_ensemble_score(
        s_asv, s_aasist, s_lfcc, alpha=alpha, beta=beta, mode=mode
    ).tolist()
    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "mode": mode,
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }


def sweep_beta(
    s_asv: np.ndarray,
    s_aasist: np.ndarray,
    s_lfcc: np.ndarray,
    keys: list[str],
    *,
    betas: np.ndarray | None = None,
    mode: str = "sum",
    alpha: float = 1.0,
    sasv_root: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if betas is None:
        betas = np.linspace(0.0, 1.0, 21)
    rows = [
        eers_for_ensemble(
            s_asv,
            s_aasist,
            s_lfcc,
            keys,
            alpha=alpha,
            beta=float(b),
            mode=mode,
            sasv_root=sasv_root,
        )
        for b in betas
    ]
    best = min(rows, key=lambda r: (r["sasv_eer"], r["sv_eer"], r["spf_eer"]))
    return best, rows


def sweep_alpha_beta(
    s_asv: np.ndarray,
    s_aasist: np.ndarray,
    s_lfcc: np.ndarray,
    keys: list[str],
    *,
    alphas: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    sasv_root: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Grid search α (ASV weight) and β (AASIST share of CM) with weighted mode."""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 11)
    if betas is None:
        betas = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float]] = []
    for a in alphas:
        for b in betas:
            rows.append(
                eers_for_ensemble(
                    s_asv,
                    s_aasist,
                    s_lfcc,
                    keys,
                    alpha=float(a),
                    beta=float(b),
                    mode="weighted",
                    sasv_root=sasv_root,
                )
            )
    best = min(rows, key=lambda r: (r["sasv_eer"], r["sv_eer"], r["spf_eer"]))
    return best, rows


def save_ensemble_run(
    *,
    split: str,
    s_asv: np.ndarray,
    s_aasist: np.ndarray,
    s_lfcc: np.ndarray,
    keys: list[str],
    metrics: dict,
    output_dir: Path | None = None,
) -> Path:
    out = Path(output_dir or (RUNS_DIR / f"ecapa_plus_aasist_lfcc_ens_{split}"))
    out.mkdir(parents=True, exist_ok=True)
    alpha = float(metrics.get("alpha", 1.0))
    beta = float(metrics["beta"])
    mode = str(metrics.get("mode", "sum"))
    scores = fused_ensemble_score(
        s_asv, s_aasist, s_lfcc, alpha=alpha, beta=beta, mode=mode
    )
    s_cm = ensemble_cm(s_aasist, s_lfcc, beta)
    score_path = out / f"scores_{split}.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "s_asv",
                "s_cm_aasist",
                "s_cm_lfcc",
                "s_cm",
                "score",
                "alpha",
                "beta",
                "mode",
            ],
        )
        writer.writeheader()
        for i, key in enumerate(keys):
            writer.writerow(
                {
                    "key": key,
                    "s_asv": float(s_asv[i]),
                    "s_cm_aasist": float(s_aasist[i]),
                    "s_cm_lfcc": float(s_lfcc[i]),
                    "s_cm": float(s_cm[i]),
                    "score": float(scores[i]),
                    "alpha": alpha,
                    "beta": beta,
                    "mode": mode,
                }
            )
    payload = {
        **metrics,
        "system": "ecapa_plus_aasist_lfcc_ensemble",
        "split": split,
        "num_scored": len(keys),
        "fusion": (
            "s_asv + (β·s_cm_aasist + (1-β)·s_cm_lfcc)"
            if mode == "sum"
            else "α·s_asv + (1-α)·(β·s_cm_aasist + (1-β)·s_cm_lfcc)"
        ),
    }
    (out / f"metrics_{split}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out
