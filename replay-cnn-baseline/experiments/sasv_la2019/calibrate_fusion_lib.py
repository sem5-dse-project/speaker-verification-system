"""B1-v2-style score calibration then fusion for SASV LFCC CSVs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiment_lib import RUNS_DIR, ensure_sasv_on_path
from weighted_fusion_lib import load_score_csv


def _labels_target(keys: list[str]) -> np.ndarray:
    return np.asarray([1 if k == "target" else 0 for k in keys], dtype=np.int32)


def _labels_bonafide(keys: list[str]) -> np.ndarray:
    return np.asarray([0 if k == "spoof" else 1 for k in keys], dtype=np.int32)


def _fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    x = scores.reshape(-1, 1)
    # class_weight helps with target-scarce SASV labels
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(x, y)
    return clf


def _proba(clf: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scores.reshape(-1, 1))[:, 1]


def _logit(clf: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return clf.decision_function(scores.reshape(-1, 1))


def eers_from_preds(
    preds: np.ndarray | list[float],
    keys: list[str],
    sasv_root: Path | None = None,
) -> dict[str, float]:
    ensure_sasv_on_path(sasv_root)
    from metrics import get_all_EERs

    sasv_eer, sv_eer, spf_eer = get_all_EERs(list(preds), keys)
    return {
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }


def fit_calibrators_dev(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    keys: list[str],
) -> dict:
    """Fit calibrators on **dev** only.

    - ``asv_sasv`` / ``cm_sasv``: Platt on each score with target vs rest
    - ``asv_sv``: Platt on ASV using target vs nontarget only
    - ``cm_spf``: Platt on CM using bona fide vs spoof
    - ``joint``: logistic on [s_asv, s_cm] with target vs rest
    """
    y_tgt = _labels_target(keys)
    y_bf = _labels_bonafide(keys)

    asv_sasv = _fit_platt(s_asv, y_tgt)
    cm_sasv = _fit_platt(s_cm, y_tgt)

    sv_mask = np.asarray([k in {"target", "nontarget"} for k in keys])
    asv_sv = _fit_platt(s_asv[sv_mask], y_tgt[sv_mask])
    cm_spf = _fit_platt(s_cm, y_bf)

    joint = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    joint.fit(np.column_stack([s_asv, s_cm]), y_tgt)

    return {
        "asv_sasv": asv_sasv,
        "cm_sasv": cm_sasv,
        "asv_sv": asv_sv,
        "cm_spf": cm_spf,
        "joint": joint,
    }


def fuse_scores(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    calibrators: dict,
    method: str,
) -> np.ndarray:
    """Apply locked calibrators and return a fused score (higher = target)."""
    method = method.strip().lower()
    if method == "raw_sum":
        return s_asv + s_cm
    if method == "platt_sum_sasv":
        # Calibrate each stream for target-vs-rest, then sum probabilities (B1-v2-ish)
        return _proba(calibrators["asv_sasv"], s_asv) + _proba(
            calibrators["cm_sasv"], s_cm
        )
    if method == "logit_sum_sasv":
        return _logit(calibrators["asv_sasv"], s_asv) + _logit(
            calibrators["cm_sasv"], s_cm
        )
    if method == "platt_sum_sv_spf":
        # ASV calibrated on SV trials; CM on spoof trials — classic CM+ASV recipe
        return _proba(calibrators["asv_sv"], s_asv) + _proba(calibrators["cm_spf"], s_cm)
    if method == "logit_sum_sv_spf":
        return _logit(calibrators["asv_sv"], s_asv) + _logit(calibrators["cm_spf"], s_cm)
    if method == "joint_proba":
        x = np.column_stack([s_asv, s_cm])
        return calibrators["joint"].predict_proba(x)[:, 1]
    if method == "joint_logit":
        x = np.column_stack([s_asv, s_cm])
        return calibrators["joint"].decision_function(x)
    raise ValueError(f"Unknown fusion method: {method}")


METHODS = [
    "raw_sum",
    "platt_sum_sasv",
    "logit_sum_sasv",
    "platt_sum_sv_spf",
    "logit_sum_sv_spf",
    "joint_proba",
    "joint_logit",
]


def evaluate_methods(
    s_asv: np.ndarray,
    s_cm: np.ndarray,
    keys: list[str],
    calibrators: dict,
    *,
    methods: list[str] | None = None,
    sasv_root: Path | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in methods or METHODS:
        preds = fuse_scores(s_asv, s_cm, calibrators, method)
        metrics = eers_from_preds(preds, keys, sasv_root)
        metrics["method"] = method
        out[method] = metrics
    return out


def pick_best_method(
    dev_results: dict[str, dict[str, float]],
    *,
    exclude: set[str] | None = None,
) -> str:
    """Pick method with lowest SASV-EER on **dev** (skip raw_sum by default for 'tuned')."""
    exclude = exclude or set()
    candidates = {
        k: v for k, v in dev_results.items() if k not in exclude and k != "raw_sum"
    }
    if not candidates:
        candidates = dev_results
    return min(candidates.items(), key=lambda kv: kv[1]["sasv_eer"])[0]


def save_calibrators(calibrators: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(calibrators, handle)


def load_calibrators(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_run_summary(
    *,
    split: str,
    locked_method: str,
    results: dict[str, dict[str, float]],
    output_dir: Path | None = None,
    extra: dict | None = None,
) -> Path:
    out = Path(output_dir or (RUNS_DIR / f"ecapa_plus_lfcc_calibrated_{split}"))
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": "ecapa_plus_lfcc_calibrated",
        "split": split,
        "locked_method": locked_method,
        "results": results,
        **(extra or {}),
    }
    (out / f"metrics_{split}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out
