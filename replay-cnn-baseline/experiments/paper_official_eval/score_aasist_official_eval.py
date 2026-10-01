"""Score the published AASIST checkpoint on the official replay evaluation lists.

Weights are the ASVspoof 2019 LA model in ``aasist_zeroshot/AASIST.pth``.
Nothing is fine-tuned. Spoof probability is softmax class 0, matching the
SASV scoring code. Equal error rate is the sweep on this split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from common import asv17_root, experiments_dir, pa_root, paper_runs, run  # noqa: E402

_AASIST_DIR = experiments_dir() / "aasist_zeroshot"
sys.path.insert(0, str(_AASIST_DIR))

import zero_shot_eval as zs  # noqa: E402


def _checkpoint() -> Path:
    path = _AASIST_DIR / "AASIST.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _config() -> Path:
    path = _AASIST_DIR / "config" / "AASIST.conf"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def score_2017() -> None:
    output = paper_runs() / "aasist_2017_eval"
    run(
        [
            sys.executable,
            str(_AASIST_DIR / "zero_shot_eval.py"),
            "eval",
            "--data-root",
            str(asv17_root()),
            "--checkpoint",
            str(_checkpoint()),
            "--config",
            str(_config()),
            "--split",
            "eval",
            "--output",
            str(output),
        ]
    )


def _pa_eval_records() -> list:
    protocol = (
        pa_root()
        / "ASVspoof2019_PA_cm_protocols"
        / "ASVspoof2019.PA.cm.eval.trl.txt"
    )
    records = []
    for line_number, line in enumerate(protocol.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Bad PA protocol line {line_number}")
        speaker_id, utt_id = fields[0], fields[1]
        label_text = fields[-1].lower()
        if label_text not in {"bonafide", "spoof"}:
            raise ValueError(f"Unexpected label {label_text!r} at line {line_number}")
        records.append((utt_id, int(label_text == "spoof"), speaker_id))
    return records


def score_pa(batch_size: int, max_files: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zs.load_aasist(_checkpoint(), _config(), device)
    print(f"Loaded AASIST on {device}")
    records = _pa_eval_records()
    n_protocol = len(records)
    if max_files > 0:
        records = records[:max_files]
    audio_dir = pa_root() / "ASVspoof2019_PA_eval" / "flac"
    labels: list[int] = []
    scores: list[float] = []
    skipped = 0
    for start in tqdm(range(0, len(records), batch_size), desc="AASIST PA eval"):
        chunk = records[start : start + batch_size]
        waves = []
        chunk_labels = []
        for utt_id, label, _speaker in chunk:
            path = audio_dir / f"{utt_id}.flac"
            if not path.is_file():
                path = audio_dir / f"{utt_id}.wav"
            try:
                waves.append(zs.load_waveform(path))
            except Exception:
                skipped += 1
                continue
            chunk_labels.append(label)
        if not waves:
            continue
        batch = torch.stack(waves).to(device)
        with torch.inference_mode():
            _, logits = model(batch, Freq_aug=False)
            spoof_prob = F.softmax(logits, dim=1)[:, 0]
        scores.extend(spoof_prob.detach().cpu().tolist())
        labels.extend(chunk_labels)
    labels_np = np.asarray(labels, dtype=int)
    scores_np = np.asarray(scores, dtype=float)
    eer, eer_thr = zs.calculate_eer(labels_np, scores_np)
    result = {
        "model": "AASIST",
        "mode": "zero_shot_published_la_weights",
        "dataset": "ASVspoof2019_PA",
        "split": "eval",
        "checkpoint": str(_checkpoint().resolve()),
        "num_protocol_files": n_protocol,
        "num_scored_files": int(len(labels_np)),
        "num_skipped_corrupt": skipped,
        "eer_percent": float(eer * 100.0),
        "eer_threshold_on_this_split": float(eer_thr),
        "spoof_class_index": 0,
    }
    output = paper_runs() / "aasist_pa_eval"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if max_files <= 0 and result["num_scored_files"] != 134730:
        print(
            f"WARNING: scored {result['num_scored_files']} of "
            f"{result['num_protocol_files']} PA eval files."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["both", "2017", "pa"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0, help="PA only. 0 = full protocol.")
    args = parser.parse_args()
    if args.corpus in {"both", "2017"}:
        score_2017()
    if args.corpus in {"both", "pa"}:
        score_pa(args.batch_size, args.max_files)


if __name__ == "__main__":
    main()
