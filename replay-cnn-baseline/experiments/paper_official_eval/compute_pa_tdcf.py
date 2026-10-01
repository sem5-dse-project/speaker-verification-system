"""Minimum t-DCF on ASVspoof 2019 PA eval for saved countermeasure scores.

Uses the ASVspoof 2019 evaluation-plan cost model and the legacy t-DCF
normalisation. The ASV system is the official PA eval score file, with its
threshold fixed at the target-versus-nontarget equal-error-rate point.
Countermeasure scores in the CSVs are spoof probabilities; t-DCF expects
higher scores to support the bona fide hypothesis, so the sign is flipped.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from common import pa_root, paper_runs  # noqa: E402

PSPOOF = 0.05
COST_MODEL = {
    "Pspoof": PSPOOF,
    "Ptar": (1.0 - PSPOOF) * 0.99,
    "Pnon": (1.0 - PSPOOF) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}


def _asv_scores(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    protocol = root / "ASVspoof2019_PA_asv_protocols" / "ASVspoof2019.PA.asv.eval.gi.trl.txt"
    scores = root / "ASVspoof2019_PA_asv_scores" / "ASVspoof2019.PA.asv.eval.gi.trl.scores.txt"
    target, nontarget, spoof = [], [], []
    with protocol.open(encoding="utf-8") as proto, scores.open(encoding="utf-8") as score_file:
        for line_number, (pline, sline) in enumerate(zip(proto, score_file), start=1):
            pfields = pline.split()
            sfields = sline.split()
            if len(pfields) < 5 or len(sfields) < 3:
                raise ValueError(f"Bad ASV line {line_number}")
            if pfields[-2:] != sfields[:2]:
                raise ValueError(
                    f"ASV protocol and scores disagree at line {line_number}: "
                    f"{pfields[-2:]} vs {sfields[:2]}"
                )
            kind = pfields[-1]
            value = float(sfields[-1])
            if kind == "target":
                target.append(value)
            elif kind == "nontarget":
                nontarget.append(value)
            elif kind == "spoof":
                spoof.append(value)
            else:
                raise ValueError(f"Unexpected ASV trial type {kind!r} at line {line_number}")
        if proto.readline() or score_file.readline():
            raise ValueError("ASV protocol and score files have different lengths")
    return {
        "target": np.asarray(target, dtype=np.float64),
        "nontarget": np.asarray(nontarget, dtype=np.float64),
        "spoof": np.asarray(spoof, dtype=np.float64),
    }


def _det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )
    frr = np.concatenate((np.atleast_1d(0.0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1.0), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def _eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple[float, float]:
    frr, far, thresholds = _det_curve(target_scores, nontarget_scores)
    index = int(np.argmin(np.abs(frr - far)))
    eer = float(np.mean((frr[index], far[index])))
    return eer, float(thresholds[index])


def _min_tdcf(
    bonafide_cm: np.ndarray,
    spoof_cm: np.ndarray,
    pfa_asv: float,
    pmiss_asv: float,
    pmiss_spoof_asv: float,
) -> float:
    """ASVspoof 2019 legacy minimum normalised t-DCF. Higher CM score = bona fide."""
    pmiss_cm, pfa_cm, _thresholds = _det_curve(bonafide_cm, spoof_cm)
    c1 = COST_MODEL["Ptar"] * (
        COST_MODEL["Cmiss_cm"] - COST_MODEL["Cmiss_asv"] * pmiss_asv
    ) - COST_MODEL["Pnon"] * COST_MODEL["Cfa_asv"] * pfa_asv
    c2 = COST_MODEL["Cfa_cm"] * COST_MODEL["Pspoof"] * (1.0 - pmiss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise RuntimeError(f"Negative t-DCF weights c1={c1} c2={c2}")
    tdcf = c1 * pmiss_cm + c2 * pfa_cm
    return float(np.min(tdcf / min(c1, c2)))


def _load_cm(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    bonafide: list[float] = []
    spoof: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "replay_probability" not in reader.fieldnames:
            raise ValueError(f"{path} has no replay_probability column")
        label_key = "true_label" if "true_label" in reader.fieldnames else None
        if label_key is None:
            raise ValueError(f"{path} has no true_label column: {reader.fieldnames}")
        for row in reader:
            label = row[label_key].strip().lower()
            # Stored value is a spoof probability. t-DCF wants bona fide support.
            score = -float(row["replay_probability"])
            if label in {"bonafide", "live", "genuine"}:
                bonafide.append(score)
            elif label in {"spoof", "replay"}:
                spoof.append(score)
            else:
                raise ValueError(f"Unexpected label {label!r} in {path}")
    if not bonafide or not spoof:
        raise ValueError(f"{path} is missing one class")
    return np.asarray(bonafide), np.asarray(spoof), len(bonafide) + len(spoof)


def main() -> None:
    grouped = _asv_scores(pa_root())
    eer_asv, asv_threshold = _eer(grouped["target"], grouped["nontarget"])
    pfa_asv = float(np.mean(grouped["nontarget"] >= asv_threshold))
    pmiss_asv = float(np.mean(grouped["target"] < asv_threshold))
    pmiss_spoof_asv = float(np.mean(grouped["spoof"] < asv_threshold))
    print(
        f"ASV eval: target={grouped['target'].size} nontarget={grouped['nontarget'].size} "
        f"spoof={grouped['spoof'].size} eer={eer_asv * 100:.2f}% "
        f"Pfa={pfa_asv:.4f} Pmiss={pmiss_asv:.4f} Pmiss_spoof={pmiss_spoof_asv:.4f}"
    )

    runs = paper_runs()
    jobs = [
        ("2017-only log-Mel", runs / "2017_mel_on_pa_eval" / "pa2019_eval_predictions.csv"),
        ("2017-only inverted-Mel", runs / "2017_inverted_mel_on_pa_eval" / "pa2019_eval_predictions.csv"),
        ("PA-only inverted-Mel", runs / "pa_specialist_on_pa_eval" / "pa2019_eval_full_predictions.csv"),
        ("mixed log-Mel", runs / "mixed_frontends_eval" / "mel" / "pa2019_predictions.csv"),
        ("mixed inverted-Mel", runs / "mixed_frontends_eval" / "inverted_mel" / "pa2019_predictions.csv"),
        ("mixed LFCC", runs / "mixed_frontends_eval" / "lfcc" / "pa2019_predictions.csv"),
        ("mixed inverted-Mel seed 43", runs / "mixed_imel_seed43_eval" / "pa2019_predictions.csv"),
        ("mixed inverted-Mel seed 44", runs / "mixed_imel_seed44_eval" / "pa2019_predictions.csv"),
    ]
    rows = []
    for name, path in jobs:
        if not path.is_file():
            raise FileNotFoundError(path)
        bonafide, spoof, n_files = _load_cm(path)
        min_tdcf = _min_tdcf(bonafide, spoof, pfa_asv, pmiss_asv, pmiss_spoof_asv)
        row = {
            "model": name,
            "test": "PA eval",
            "min_tdcf": min_tdcf,
            "n_bonafide": int(bonafide.size),
            "n_spoof": int(spoof.size),
            "n_files": n_files,
            "score_file": str(path),
        }
        rows.append(row)
        print(f"{name:<32} min t-DCF {min_tdcf:.4f}  n={n_files}")

    payload = {
        "metric": "minimum normalised t-DCF",
        "dataset": "ASVspoof2019_PA",
        "split": "eval",
        "cost_model": COST_MODEL,
        "asv_eer_percent": eer_asv * 100.0,
        "asv_threshold": asv_threshold,
        "Pfa_asv": pfa_asv,
        "Pmiss_asv": pmiss_asv,
        "Pmiss_spoof_asv": pmiss_spoof_asv,
        "cm_score_orientation": "negated spoof probability; higher supports bona fide",
        "rows": rows,
    }
    out = runs / "pa_tdcf.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
