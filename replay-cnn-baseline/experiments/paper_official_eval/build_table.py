"""Collect the official-eval JSON files into one table.

Missing files are listed instead of being filled with the old development
numbers. Equal error rate is the sweep on the evaluation split. The checkpoint
threshold is not refit here.
"""

from __future__ import annotations

import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_THIS))

from common import paper_runs, protocol_count  # noqa: E402


def _eer(block: dict) -> float | None:
    if not block:
        return None
    if "eer_percent" in block and not isinstance(block["eer_percent"], dict):
        return float(block["eer_percent"])
    for key in ("metrics_at_oracle_eer_threshold", "metrics_at_train_val_threshold", "metrics_at_pa_train_val_threshold"):
        inner = block.get(key)
        if isinstance(inner, dict) and "eer_percent" in inner:
            return float(inner["eer_percent"])
    return None


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs = paper_runs()
    expected = {
        "2017_eval": protocol_count("2017_eval"),
        "pa_eval": protocol_count("pa_eval"),
        "la_eval": protocol_count("la_eval"),
    }
    rows = []
    missing = []

    def add(row: dict, path: Path) -> None:
        if row.get("eer") is None:
            missing.append(str(path))
        rows.append(row)

    for feature in ("mel", "inverted_mel"):
        path = runs / f"2017_{feature}_eval" / "eval_metrics.json"
        data = _load(path)
        scored = None
        eer = None
        if data:
            matrix = data.get("confusion_matrix_live_replay")
            if matrix:
                scored = sum(sum(r) for r in matrix)
            eer = _eer(data)
        add(
            {
                "model": f"2017-only {feature}",
                "test": "2017 eval",
                "eer": eer,
                "scored": scored,
                "protocol": expected["2017_eval"],
            },
            path,
        )

    pa_on_pa = runs / "pa_specialist_on_pa_eval" / "pa2019_eval_full_metrics.json"
    data = _load(pa_on_pa)
    add(
        {
            "model": "PA-only inverted-Mel",
            "test": "PA eval",
            "eer": _eer(data) if data else None,
            "scored": None if data is None else data.get("num_scored_files"),
            "protocol": expected["pa_eval"],
            "skipped": None if data is None else data.get("num_skipped_corrupt"),
        },
        pa_on_pa,
    )
    pa_on_2017 = runs / "pa_specialist_on_2017_eval" / "asvspoof2017_eval_metrics.json"
    data = _load(pa_on_2017)
    add(
        {
            "model": "PA-only inverted-Mel",
            "test": "2017 eval",
            "eer": _eer(data) if data else None,
            "scored": None if data is None else data.get("num_scored_files"),
            "protocol": expected["2017_eval"],
        },
        pa_on_2017,
    )

    table_path = runs / "mixed_frontends_eval" / "comparison_table.json"
    metrics_path = runs / "mixed_frontends_eval" / "all_metrics.json"
    table = _load(table_path) or []
    metrics = _load(metrics_path) or []
    by_feature = {item.get("feature_type"): item for item in metrics}
    if not table:
        missing.append(str(table_path))
    for row in table:
        feature = row.get("feature")
        detail = by_feature.get(feature, {})
        add(
            {
                "model": f"mixed {feature}",
                "test": "2017 eval",
                "eer": row.get("eer_2017"),
                "scored": row.get("n_asvspoof2017"),
                "protocol": expected["2017_eval"],
            },
            table_path,
        )
        add(
            {
                "model": f"mixed {feature}",
                "test": "PA eval",
                "eer": row.get("eer_pa"),
                "scored": row.get("n_pa2019"),
                "protocol": expected["pa_eval"],
                "skipped": detail.get("pa_num_skipped"),
            },
            table_path,
        )

    mixed_both = runs / "mixed_imel_both_eval" / "comparison_summary.json"
    data = _load(mixed_both)
    if data is None:
        missing.append(str(mixed_both))
    else:
        for block in data.get("comparison", []):
            test = "2017 eval" if block.get("corpus") == "asvspoof2017" else "PA eval"
            protocol = expected["2017_eval"] if test.startswith("2017") else expected["pa_eval"]
            rows.append(
                {
                    "model": "mixed inverted-Mel (LA checkpoint)",
                    "test": test,
                    "eer": block.get("eer_percent_oracle"),
                    "scored": block.get("num_files"),
                    "protocol": protocol,
                }
            )

    la_path = runs / "mixed_imel_on_la_eval" / "metrics.json"
    data = _load(la_path)
    add(
        {
            "model": "mixed inverted-Mel (LA checkpoint)",
            "test": "LA eval",
            "eer": None if data is None else data.get("eer_percent"),
            "scored": None if data is None else data.get("num_scored_files"),
            "protocol": expected["la_eval"],
            "skipped": None if data is None else data.get("num_skipped_corrupt"),
        },
        la_path,
    )

    print(f"{'Model':<42} {'Test':<12} {'EER':>8} {'Scored':>10} {'Protocol':>10} {'Cover':>8}")
    for row in rows:
        eer = row.get("eer")
        eer_s = "—" if eer is None else f"{float(eer):.2f}"
        scored = row.get("scored")
        protocol = row.get("protocol")
        cover = "—"
        if isinstance(scored, int) and isinstance(protocol, int) and protocol:
            cover = f"{100.0 * scored / protocol:.1f}%"
        print(
            f"{row['model']:<42} {row['test']:<12} {eer_s:>8} "
            f"{str(scored if scored is not None else '—'):>10} {str(protocol):>10} {cover:>8}"
        )
        if isinstance(scored, int) and isinstance(protocol, int) and scored != protocol:
            print(f"  WARNING: {row['model']} on {row['test']} did not score the full protocol.")

    out = runs / "paper_table.json"
    out.write_text(json.dumps({"rows": rows, "missing": missing}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    if missing:
        print("\nStill missing:")
        for path in missing:
            print(" ", path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
