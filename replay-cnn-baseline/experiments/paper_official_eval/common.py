"""Shared checks for the official-split replay evaluation.

Run ``00_check_pa_replacement.ipynb`` after the new ASVspoof 2019 PA audio
replaces ``data/PA``. Scoring notebooks call ``ensure_pa_ready`` first so a
stale readability cache cannot keep skipping files that now decode.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROBE_DEV_UTT = "PA_D_0000007"
EVAL_PROBE_COUNT = 40
STALE_SKIP_FRACTION = 0.02


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "replay-cnn-baseline").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not find the repository root from " + str(here))


def experiments_dir() -> Path:
    return repo_root() / "replay-cnn-baseline" / "experiments"


def pa_root() -> Path:
    return repo_root() / "data" / "PA"


def asv17_root() -> Path:
    return repo_root() / "replay-cnn-baseline" / "data"


def la_root() -> Path:
    return repo_root() / "data" / "LA"


def paper_runs() -> Path:
    path = experiments_dir() / "paper_official_eval" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def protocol_count(name: str) -> int:
    root = repo_root()
    paths = {
        "2017_eval": asv17_root() / "protocol_V2" / "ASVspoof2017_V2_eval.trl.txt",
        "pa_dev": pa_root()
        / "ASVspoof2019_PA_cm_protocols"
        / "ASVspoof2019.PA.cm.dev.trl.txt",
        "pa_eval": pa_root()
        / "ASVspoof2019_PA_cm_protocols"
        / "ASVspoof2019.PA.cm.eval.trl.txt",
        "la_eval": root
        / "data"
        / "LA"
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(path)
    return _nonempty_lines(path)


def _decode_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    import soundfile as sf

    try:
        sf.read(str(path), dtype="float32")
    except Exception:
        return False
    return True


def _eval_probe_ids() -> list[str]:
    protocol = (
        pa_root()
        / "ASVspoof2019_PA_cm_protocols"
        / "ASVspoof2019.PA.cm.eval.trl.txt"
    )
    rows = [line.split()[1] for line in protocol.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < EVAL_PROBE_COUNT:
        return rows
    stride = len(rows) // EVAL_PROBE_COUNT
    return [rows[i * stride] for i in range(EVAL_PROBE_COUNT)]


def probe_pa_replacement() -> tuple[bool, str]:
    """Return whether the replaced PA audio decodes.

    The dev file ``PA_D_0000007`` failed on the corrupt copy. The eval check
    samples 40 protocol files spread across the list. The old copy failed
    about half of those opens.
    """
    dev_path = pa_root() / "ASVspoof2019_PA_dev" / "flac" / f"{PROBE_DEV_UTT}.flac"
    if not _decode_ok(dev_path):
        return (
            False,
            f"{dev_path} still does not decode. Replace data/PA, then rerun this check.",
        )
    audio_dir = pa_root() / "ASVspoof2019_PA_eval" / "flac"
    bad = []
    for utt in _eval_probe_ids():
        if not _decode_ok(audio_dir / f"{utt}.flac"):
            bad.append(utt)
    if bad:
        return (
            False,
            f"PA eval still has unreadable files ({len(bad)} of {EVAL_PROBE_COUNT} probed"
            f", first {bad[0]}). Finish replacing data/PA before scoring.",
        )
    eval_wav = asv17_root() / "ASVspoof2017_V2_eval" / "E_1000001.wav"
    if not eval_wav.is_file():
        return False, f"Missing 2017 eval audio: {eval_wav}"
    return True, (
        f"PA dev probe {PROBE_DEV_UTT} decodes, and {EVAL_PROBE_COUNT}/{EVAL_PROBE_COUNT} "
        "PA eval probes decode. 2017 eval audio is present."
    )


def pa_cache_files() -> list[Path]:
    return sorted(experiments_dir().rglob("pa2019_*_readable.json"))


def drop_stale_pa_caches() -> list[Path]:
    """Delete readability caches that still record the corrupt skip list."""
    removed = []
    for path in pa_cache_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink()
            removed.append(path)
            continue
        skipped = int(data.get("num_skipped", len(data.get("skipped", []))))
        readable = int(data.get("num_readable", len(data.get("readable", []))))
        total = skipped + readable
        if total == 0 or (skipped / total) > STALE_SKIP_FRACTION:
            path.unlink()
            removed.append(path)
    return removed


def ensure_pa_ready() -> None:
    ok, message = probe_pa_replacement()
    print(message)
    if not ok:
        raise SystemExit(1)
    removed = drop_stale_pa_caches()
    if removed:
        print(f"Removed {len(removed)} stale PA readability caches:")
        for path in removed:
            print(" ", path.relative_to(repo_root()))
    else:
        print("No stale PA readability caches.")


def run(args: list[str]) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True)
