"""Rewrite ASVspoof 2019 PA FLACs that libsndfile cannot decode.

The official archive contains files whose headers are valid but whose frames
make libsndfile 1.2.2 report "flac decoder lost sync". ffmpeg decodes those
files. This script writes a new FLAC beside the original, checks that
soundfile can read it, then replaces the original.

Train and dev use the utterance ids already recorded as skipped. Eval is
scanned from the protocol because it was never fully cached.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from common import experiments_dir, pa_root  # noqa: E402

FFMPEG = Path(
    os.environ.get(
        "FFMPEG",
        r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    )
)


def _skipped_ids(name: str) -> list[str]:
    caches = sorted(experiments_dir().rglob(f"pa2019_{name}_readable.json"))
    if not caches:
        raise FileNotFoundError(f"No pa2019_{name}_readable.json cache")
    best: list[str] = []
    for path in caches:
        data = json.loads(path.read_text(encoding="utf-8"))
        skipped = list(data.get("skipped", []))
        if len(skipped) > len(best):
            best = skipped
    return best


def _protocol_ids(split: str) -> list[str]:
    filename = {
        "train": "ASVspoof2019.PA.cm.train.trn.txt",
        "dev": "ASVspoof2019.PA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.PA.cm.eval.trl.txt",
    }[split]
    protocol = pa_root() / "ASVspoof2019_PA_cm_protocols" / filename
    ids = []
    for line in protocol.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            ids.append(fields[1])
    return ids


def _audio_dir(split: str) -> Path:
    return pa_root() / f"ASVspoof2019_PA_{split}" / "flac"


def repair_one(path_str: str) -> str:
    """Return ok, repaired, missing, ffmpeg_fail, or still_bad."""
    import soundfile as sf

    path = Path(path_str)
    if not path.is_file():
        return "missing"
    try:
        sf.read(str(path), dtype="float32")
        return "ok"
    except Exception:
        pass
    tmp = path.with_suffix(".repairing.flac")
    result = subprocess.run(
        [str(FFMPEG), "-y", "-v", "error", "-i", str(path), "-c:a", "flac", str(tmp)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return "ffmpeg_fail"
    try:
        sf.read(str(tmp), dtype="float32")
    except Exception:
        tmp.unlink(missing_ok=True)
        return "still_bad"
    os.replace(tmp, path)
    return "repaired"


def _run_split(split: str, ids: list[str], workers: int) -> dict[str, int]:
    folder = _audio_dir(split)
    paths = [str(folder / f"{utt}.flac") for utt in ids]
    counts = {"ok": 0, "repaired": 0, "missing": 0, "ffmpeg_fail": 0, "still_bad": 0}
    print(f"{split}: {len(paths)} files, {workers} workers", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(repair_one, path) for path in paths]
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 2000 == 0 or done == len(paths):
                print(f"  {split} {done}/{len(paths)} {counts}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    args = parser.parse_args()
    if not FFMPEG.is_file():
        raise SystemExit(f"ffmpeg not found: {FFMPEG}")
    summary = {}
    for split in args.splits:
        if split == "eval":
            ids = _protocol_ids("eval")
        else:
            ids = _skipped_ids(split)
        summary[split] = _run_split(split, ids, args.workers)
    out = _THIS / "runs" / "pa_repair_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
