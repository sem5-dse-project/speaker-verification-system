"""ECAPA (+ optional CM) scoring for SASV trials."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_lib import (
    CACHE_DIR,
    RUNS_DIR,
    SasvTrial,
    cosine,
    ensure_sasv_on_path,
    ensure_server_on_path,
    l2_normalize,
    load_waveform,
    read_enroll_map,
    read_trials,
    resolve_audio_path,
    trial_key_counts,
)


def load_app_ecapa(device: str = "cpu"):
    ensure_server_on_path()
    from ml_server.ecapa import load_ecapa_encoder

    return load_ecapa_encoder(device=device)


@torch.inference_mode()
def embed_utt(
    classifier,
    la_root: Path,
    split: str,
    utt_id: str,
    device: str,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if cache is not None and utt_id in cache:
        return cache[utt_id]
    path = resolve_audio_path(la_root, split, utt_id)
    wave = load_waveform(path)
    batch = wave.unsqueeze(0).to(device)
    from ml_server.ecapa import encode_waveforms

    emb = encode_waveforms(classifier, batch).squeeze(0).cpu().numpy().astype(np.float32)
    if cache is not None:
        cache[utt_id] = emb
    return emb


def build_speaker_models(
    classifier,
    la_root: Path,
    split: str,
    device: str,
    emb_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    enroll = read_enroll_map(la_root, split)
    models: dict[str, np.ndarray] = {}
    for spk, utts in tqdm(enroll.items(), desc=f"Enrol {split}"):
        embs = [
            embed_utt(classifier, la_root, split, utt, device, emb_cache) for utt in utts
        ]
        models[spk] = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
    return models


def score_ecapa_trials(
    *,
    la_root: Path,
    sasv_root: Path,
    split: str = "dev",
    max_trials: int = 500,
    device: str = "cpu",
    force_cpu: bool = False,
    output_dir: Path | None = None,
) -> dict:
    ensure_sasv_on_path(sasv_root)
    from metrics import get_all_EERs

    if force_cpu:
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(output_dir or (RUNS_DIR / f"ecapa_only_{split}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    trials = read_trials(sasv_root, split, max_trials=max_trials)
    print(f"Trials: {trial_key_counts(trials)}")
    print(f"Device: {device}")

    classifier = load_app_ecapa(device=device)
    emb_cache: dict[str, np.ndarray] = {}
    spk_models = build_speaker_models(
        classifier, la_root, split, device, emb_cache=emb_cache
    )

    preds: list[float] = []
    keys: list[str] = []
    rows: list[dict] = []

    for trial in tqdm(trials, desc="Score trials"):
        if trial.speaker_id not in spk_models:
            raise KeyError(f"No enrolment model for {trial.speaker_id}")
        test_emb = embed_utt(
            classifier, la_root, split, trial.test_utt, device, emb_cache
        )
        score = cosine(spk_models[trial.speaker_id], test_emb)
        preds.append(score)
        keys.append(trial.key)
        rows.append(
            {
                "speaker_id": trial.speaker_id,
                "test_utt": trial.test_utt,
                "key": trial.key,
                "score": score,
            }
        )

    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    summary = {
        "system": "ecapa_only",
        "split": split,
        "max_trials": max_trials,
        "num_scored": len(preds),
        "key_counts": trial_key_counts(trials),
        "device": device,
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
        "note": "ECAPA-alone: SV-EER usually low, SPF-EER high (spoofs look like the target).",
    }

    csv_path = output_dir / f"scores_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["speaker_id", "test_utt", "key", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def score_fused_trials(
    *,
    la_root: Path,
    sasv_root: Path,
    split: str = "dev",
    max_trials: int = 500,
    device: str = "cpu",
    force_cpu: bool = False,
    cm_backend: str = "lfcc",
    output_dir: Path | None = None,
) -> dict:
    """ECAPA cosine + (1 - P_spoof) score-sum fusion."""
    ensure_sasv_on_path(sasv_root)
    ensure_server_on_path()
    from metrics import get_all_EERs
    import ml_server.config as cfg
    import ml_server.la_spoof as la_mod
    from ml_server.la_spoof import score_la

    cm_backend = (cm_backend or "lfcc").strip().lower()
    cfg.LA_BACKEND = cm_backend
    la_mod.LA_BACKEND = cm_backend
    if cm_backend == "lfcc":
        cfg.LA_CHECKPOINT = cfg._DEFAULT_LFCC_LA_CKPT
    else:
        cfg.LA_CHECKPOINT = cfg._DEFAULT_WAVLM_LA_CKPT
    la_mod.LA_CHECKPOINT = cfg.LA_CHECKPOINT
    la_mod._la_model = None
    la_mod._la_ckpt_path = None
    la_mod._la_backend = None

    if force_cpu:
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(output_dir or (RUNS_DIR / f"ecapa_plus_{cm_backend}_{split}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = read_trials(sasv_root, split, max_trials=max_trials)
    print(f"Trials: {trial_key_counts(trials)} | CM={cm_backend}")

    classifier = load_app_ecapa(device=device)
    emb_cache: dict[str, np.ndarray] = {}
    spk_models = build_speaker_models(
        classifier, la_root, split, device, emb_cache=emb_cache
    )

    preds: list[float] = []
    keys: list[str] = []
    rows: list[dict] = []

    for trial in tqdm(trials, desc="Score fused"):
        test_emb = embed_utt(
            classifier, la_root, split, trial.test_utt, device, emb_cache
        )
        s_asv = cosine(spk_models[trial.speaker_id], test_emb)
        wave = load_waveform(resolve_audio_path(la_root, split, trial.test_utt))
        cm = score_la(wave, device=device, check_speech=False)
        # High score = bona fide (invert spoof probability)
        s_cm = 1.0 - float(cm["score"])
        score = s_asv + s_cm
        preds.append(score)
        keys.append(trial.key)
        rows.append(
            {
                "speaker_id": trial.speaker_id,
                "test_utt": trial.test_utt,
                "key": trial.key,
                "s_asv": s_asv,
                "s_cm": s_cm,
                "p_spoof": float(cm["score"]),
                "score": score,
            }
        )

    sasv_eer, sv_eer, spf_eer = get_all_EERs(preds, keys)
    summary = {
        "system": f"ecapa_plus_{cm_backend}_sum",
        "split": split,
        "max_trials": max_trials,
        "num_scored": len(preds),
        "key_counts": trial_key_counts(trials),
        "device": device,
        "cm_backend": cm_backend,
        "fusion": "s_asv + (1 - p_spoof)",
        "sasv_eer": float(sasv_eer),
        "sv_eer": float(sv_eer),
        "spf_eer": float(spf_eer),
        "sasv_eer_percent": float(sasv_eer) * 100.0,
        "sv_eer_percent": float(sv_eer) * 100.0,
        "spf_eer_percent": float(spf_eer) * 100.0,
    }

    with (output_dir / f"scores_{split}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["speaker_id", "test_utt", "key", "s_asv", "s_cm", "p_spoof", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary
