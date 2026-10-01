"""Score the three mixed-training front-ends on official eval lists.

Uses the checkpoints in ``lfcc_vs_mel_compare/runs``. The threshold stored in
each checkpoint is left unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_COMPARE = _THIS.parent / "lfcc_vs_mel_compare"
sys.path.insert(0, str(_COMPARE))
sys.path.insert(0, str(_THIS))

from common import paper_runs  # noqa: E402
from experiment_lib import (  # noqa: E402
    FEATURE_TYPES,
    build_comparison_table,
    default_checkpoint,
    eval_checkpoint_on_corpora,
    markdown_eer_table,
)


def main() -> None:
    output = paper_runs() / "mixed_frontends_eval"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for feature in FEATURE_TYPES:
        checkpoint = default_checkpoint(feature)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"\n=== {feature}  {checkpoint.name} ===", flush=True)
        feature_out = output / feature
        result = eval_checkpoint_on_corpora(
            checkpoint,
            asv17_split="eval",
            pa_split="eval",
            batch_size=16,
            output_dir=feature_out,
            refresh_cache=False,
        )
        skip_file = feature_out / "pa_skipped_utts.txt"
        skipped = 0
        if skip_file.is_file():
            skipped = sum(1 for line in skip_file.read_text(encoding="utf-8").splitlines() if line.strip())
        result["pa_num_skipped"] = skipped
        scored = result.get("pa2019", {}).get("n")
        print(f"{feature}: 2017 n={result.get('asvspoof2017', {}).get('n')}  PA n={scored}  PA skipped={skipped}")
        results.append(result)

    rows = build_comparison_table(results)
    (output / "comparison_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "all_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + markdown_eer_table(rows))


if __name__ == "__main__":
    main()
