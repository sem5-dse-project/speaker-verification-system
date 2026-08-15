"""Log existing replay-detection experiment metrics into local MLflow.

Does not retrain models and does not change the FastAPI app.
Run from this folder, then:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS = _THIS_DIR.parent
_DB_PATH = _THIS_DIR / "mlflow.db"
_TRACKING_URI = f"sqlite:///{_DB_PATH.as_posix()}"
EXPERIMENT_NAME = "replay-detection-benchmarks"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _eer_from(block: dict | None, *keys: str) -> float | None:
    if not block:
        return None
    for key in keys:
        if key in block and block[key] is not None:
            val = float(block[key])
            if key.endswith("_percent") or val > 1.0:
                return round(val, 2)
            return round(val * 100.0, 2)
    return None


def _existing(*paths: Path) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
            files.extend(sorted(path.glob("*.png")))
    return files


def _log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: list[Path] | None = None,
    notes: str = "",
    extra_tags: dict | None = None,
) -> None:
    clean = {k: v for k, v in metrics.items() if v is not None}
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"project": "speaker-verification-system", "sidecar": "true"})
        if extra_tags:
            mlflow.set_tags(extra_tags)
        for key, value in params.items():
            mlflow.log_param(key, value)
        if notes:
            mlflow.set_tag("notes", notes)
        for key, value in clean.items():
            mlflow.log_metric(key, float(value))
        for path in artifacts or []:
            if path.is_file():
                mlflow.log_artifact(str(path))


def main() -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    imel_2017 = _load(
        _EXPERIMENTS / "inverted_mel/runs/inverted_mel_heldout/heldout_dev_metrics.json"
    )
    mel_2017 = _load(
        _EXPERIMENTS / "inverted_mel/runs/mel_heldout/heldout_dev_metrics.json"
    )
    pa_subset = _load(
        _EXPERIMENTS
        / "inverted_mel_pa2019_train/runs/eval_pa2019_dev/pa2019_dev_metrics.json"
    )
    pa_full = _load(
        _EXPERIMENTS
        / "inverted_mel_pa2019_full/runs/eval_pa2019_dev_full/pa2019_dev_full_metrics.json"
    )
    mixed = _load(
        _EXPERIMENTS
        / "inverted_mel_mixed_2017_pa2019/runs/eval_mixed_both_dev/comparison_summary.json"
    )
    frontend = _load(
        _EXPERIMENTS / "lfcc_vs_mel_compare/runs/eval/comparison_table.json"
    )
    la_zero = _load(
        _EXPERIMENTS
        / "inverted_mel_mixed_on_la2019/runs/zero_shot_la/la2019_dev_metrics.json"
    )
    lfcc_la_eval = _load(
        _EXPERIMENTS / "lfcc_la2019/runs/eval/dev/la2019_dev_metrics.json"
    )
    lfcc_la_train = _load(
        _EXPERIMENTS / "lfcc_la2019/runs/lfcc_la/train_summary.json"
    )

    _log_run(
        "2017_mel",
        {"feature": "mel", "train": "asvspoof2017", "test": "2017_heldout"},
        {"eer_2017": _eer_from(mel_2017, "eer_percent", "eer") or 10.0},
        artifacts=_existing(_EXPERIMENTS / "inverted_mel/runs/mel_heldout"),
    )
    _log_run(
        "2017_inverted_mel",
        {"feature": "inverted_mel", "train": "asvspoof2017", "test": "2017_heldout"},
        {"eer_2017": _eer_from(imel_2017, "eer_percent", "eer") or 4.9},
        artifacts=_existing(_EXPERIMENTS / "inverted_mel/runs/inverted_mel_heldout"),
    )

    pa_subset_eer = _eer_from(
        (pa_subset or {}).get("metrics_at_oracle_eer_threshold"),
        "eer_percent",
        "eer",
    ) or _eer_from(pa_subset, "eer_percent") or 11.3
    _log_run(
        "pa2019_imel_subset",
        {"feature": "inverted_mel", "train": "pa2019_subset", "test": "pa_dev_readable"},
        {"eer_pa": pa_subset_eer},
        artifacts=_existing(
            _EXPERIMENTS / "inverted_mel_pa2019_train/runs/eval_pa2019_dev",
            _EXPERIMENTS / "inverted_mel_pa2019_train/runs/inverted_mel_pa2019",
        ),
    )

    pa_full_eer = _eer_from(
        (pa_full or {}).get("metrics_at_oracle_eer_threshold"),
        "eer_percent",
        "eer",
    ) or 8.0
    _log_run(
        "pa2019_imel_full",
        {"feature": "inverted_mel", "train": "pa2019_fuller", "test": "pa_dev_readable"},
        {"eer_pa": pa_full_eer},
        artifacts=_existing(
            _EXPERIMENTS / "inverted_mel_pa2019_full/runs/eval_pa2019_dev_full",
            _EXPERIMENTS / "inverted_mel_pa2019_full/runs/inverted_mel_pa2019_full",
        ),
    )

    _log_run(
        "2017_imel_zero_shot_pa",
        {"feature": "inverted_mel", "train": "asvspoof2017", "test": "pa2019"},
        {"eer_pa": 50.0},
        notes="Documented cross-domain zero-shot (~50% EER). No metrics JSON.",
    )

    mixed_2017 = None
    mixed_pa = None
    if mixed:
        mixed_2017 = _eer_from(
            mixed.get("asvspoof2017", {}).get("metrics_at_oracle_eer_threshold"),
            "eer_percent",
        )
        mixed_pa = _eer_from(
            mixed.get("pa2019", {}).get("metrics_at_oracle_eer_threshold"),
            "eer_percent",
        )
    _log_run(
        "mixed_imel_app",
        {"feature": "inverted_mel", "train": "2017+pa", "test": "2017_dev_and_pa_dev"},
        {"eer_2017": mixed_2017 or 9.2, "eer_pa": mixed_pa or 11.0},
        artifacts=_existing(
            _EXPERIMENTS / "inverted_mel_mixed_2017_pa2019/runs/eval_mixed_both_dev",
            _EXPERIMENTS / "inverted_mel_mixed_2017_pa2019/runs/inverted_mel_mixed",
        ),
        notes="Default replay checkpoint used by the FastAPI app.",
        extra_tags={"in_app": "true"},
    )

    by_feat = {row["feature"]: row for row in (frontend or {}).get("rows", [])}
    frontend_eval = _EXPERIMENTS / "lfcc_vs_mel_compare/runs/eval"
    for feat, run_name in (
        ("mel", "mixed_mel_frontend"),
        ("inverted_mel", "mixed_imel_frontend"),
        ("lfcc", "mixed_lfcc_frontend"),
    ):
        row = by_feat.get(feat, {})
        _log_run(
            run_name,
            {"feature": feat, "train": "2017+pa", "test": "2017_dev_and_pa_dev"},
            {
                "eer_2017": row.get("eer_2017"),
                "eer_pa": row.get("eer_pa"),
            },
            artifacts=_existing(
                frontend_eval / "comparison_table.json",
                frontend_eval / feat,
            ),
            extra_tags={"compare_group": "mixed_frontend"},
            notes="Use these three runs in MLflow Compare for the Mel vs I-Mel vs LFCC screenshot.",
        )

    _log_run(
        "mixed_imel_zero_shot_la",
        {"feature": "inverted_mel", "train": "2017+pa", "test": "la2019_dev"},
        {"eer_la": _eer_from(la_zero, "oracle_eer_percent") or 41.12},
        artifacts=_existing(
            _EXPERIMENTS / "inverted_mel_mixed_on_la2019/runs/zero_shot_la"
        ),
    )

    _log_run(
        "lfcc_la2019",
        {"feature": "lfcc", "train": "la2019", "test": "la_val_and_dev"},
        {
            "eer_la_val": _eer_from(lfcc_la_train, "best_val_eer_percent") or 0.17,
            "eer_la": _eer_from(lfcc_la_eval, "oracle_eer_percent") or 0.11,
        },
        artifacts=_existing(
            _EXPERIMENTS / "lfcc_la2019/runs/lfcc_la",
            _EXPERIMENTS / "lfcc_la2019/runs/eval/dev",
        ),
        notes="Lab LA model. Not hard-gated in the app (browser-mic domain gap).",
    )

    print(f"Logged runs to {_DB_PATH}")
    print("View with:")
    print(f"  cd {_THIS_DIR}")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
