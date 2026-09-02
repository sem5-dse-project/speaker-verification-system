"""Train frozen Wav2Vec2-base-960h + MLP on mixed ASVspoof 2017 + PA2019."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

_MIXED_DIR = Path(__file__).resolve().parent.parent / "inverted_mel_mixed_2017_pa2019"
_INVERTED_MEL_DIR = Path(__file__).resolve().parent.parent / "inverted_mel"
sys.path.insert(0, str(_MIXED_DIR))
sys.path.insert(0, str(_INVERTED_MEL_DIR))

from build_mixed import load_asv17_train_val, load_pa_train_val
from experiment_lib import (
    CACHE_DIR,
    DEFAULT_ASV17,
    DEFAULT_PA,
    RUNS_DIR,
    build_waveform_loaders,
    calculate_eer,
    collect_scores,
    forward_batch,
    pick_device,
    save_artifacts,
    seed_everything,
)
from mixed_data import balance_by_corpus, count_summary, limit_mixed_stratified
from model import DEFAULT_MODEL_ID, Wav2Vec2AudioConfig, Wav2Vec2ReplayDetector


def train_wav2vec2_replay(
    *,
    asv17_root: Path | None = None,
    pa_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    freeze_encoder: bool = True,
    epochs: int = 12,
    patience: int = 4,
    batch_size: int = 8,
    workers: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seconds: float = 4.0,
    validation_fraction: float = 0.2,
    balance_corpora: bool = True,
    max_train: int = 0,
    max_val: int = 0,
    seed: int = 42,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> Path:
    asv17_root = Path(asv17_root or DEFAULT_ASV17)
    pa_root = Path(pa_root or DEFAULT_PA)
    output_dir = Path(output_dir or (RUNS_DIR / "wav2vec2_mixed"))
    cache_dir = Path(cache_dir or CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = pick_device(force_cpu)
    audio_config = Wav2Vec2AudioConfig(seconds=seconds)

    print(
        f"Wav2Vec2 mixed 2017+PA | model={model_id} | freeze={freeze_encoder} | device={device}"
    )

    asv_train, asv_val = load_asv17_train_val(asv17_root, validation_fraction, seed)
    pa_train, pa_val, skipped = load_pa_train_val(
        pa_root,
        cache_dir,
        validation_fraction,
        seed + 1,
        refresh_cache,
    )
    if skipped:
        (output_dir / "pa_train_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    train_records = asv_train + pa_train
    val_records = asv_val + pa_val
    train_records = limit_mixed_stratified(train_records, max_train, seed)
    val_records = limit_mixed_stratified(val_records, max_val, seed + 2)
    if balance_corpora:
        train_records = balance_by_corpus(train_records, seed)

    train_summary = count_summary(train_records)
    val_summary = count_summary(val_records)
    print(f"Train: {json.dumps(train_summary)}")
    print(f"Val:   {json.dumps(val_summary)}")
    (output_dir / "data_summary.json").write_text(
        json.dumps({"train": train_summary, "val": val_summary}, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader = build_waveform_loaders(
        train_records,
        val_records,
        sample_rate=audio_config.sample_rate,
        num_samples=audio_config.samples,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )

    model = Wav2Vec2ReplayDetector(
        model_id=model_id,
        freeze_encoder=freeze_encoder,
    ).to(device)

    spoof_count = sum(r.label for r in train_records)
    live_count = len(train_records) - spoof_count
    if spoof_count == 0 or live_count == 0:
        raise ValueError("Training set must contain both classes")
    pos_weight = torch.tensor([live_count / spoof_count], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = float("inf")
    patience_left = patience
    ckpt_path = output_dir / "best_wav2vec2_mixed_2017_pa2019.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_encoder:
            model.encoder.eval()
        running = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for waveforms, labels, _ in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = forward_batch(model, waveforms, device)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_labels, val_scores, val_ids = collect_scores(model, val_loader, device)
        val_eer, val_thr = calculate_eer(val_labels, val_scores)
        mean_loss = running / max(len(train_loader.dataset), 1)
        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} "
            f"val_eer={val_eer * 100:.2f}% thr={val_thr:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_eer": val_eer,
                "val_threshold": val_thr,
            }
        )

        if val_eer < best_eer:
            best_eer = val_eer
            patience_left = patience
            model.save_checkpoint(
                ckpt_path,
                audio_config=audio_config,
                threshold=val_thr,
                val_eer=val_eer,
                epoch=epoch,
                train_summary=train_summary,
                val_summary=val_summary,
                extra={
                    "balance_corpora": balance_corpora,
                    "model_type": "wav2vec2_replay_mlp",
                },
            )
            save_artifacts(
                val_labels,
                val_scores,
                val_ids,
                val_thr,
                output_dir,
                "best_val",
                extra={
                    "model_id": model_id,
                    "split": "mixed_speaker_val",
                    "epoch": epoch,
                },
            )
            print(f"Saved best checkpoint -> {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Done. Best mixed-val EER={best_eer * 100:.2f}%. Checkpoint: {ckpt_path}")
    return ckpt_path
