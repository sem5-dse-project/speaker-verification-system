"""Train frozen WavLM-Base + ASP + FC on ASVspoof 2019 LA."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from experiment_lib import (
    CACHE_DIR,
    DEFAULT_LA,
    PAPER_LA_EER_PERCENT,
    RUNS_DIR,
    build_la_loaders,
    calculate_eer,
    collect_scores,
    count_summary,
    forward_batch,
    load_la_train_val,
    pick_device,
    save_artifacts,
    seed_everything,
)
from model import DEFAULT_MODEL_ID, WavLMAudioConfig, WavLMSpoofDetector


def train_wavlm_la(
    *,
    la_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    freeze_encoder: bool = True,
    epochs: int = 12,
    patience: int = 4,
    batch_size: int = 4,
    workers: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seconds: float = 4.0,
    validation_fraction: float = 0.2,
    max_train: int = 0,
    max_val: int = 0,
    seed: int = 42,
    force_cpu: bool = False,
    refresh_cache: bool = False,
) -> Path:
    la_root = Path(la_root or DEFAULT_LA)
    output_dir = Path(output_dir or (RUNS_DIR / "wavlm_la"))
    cache_dir = Path(cache_dir or CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = pick_device(force_cpu)
    audio_config = WavLMAudioConfig(seconds=seconds)

    print(
        f"WavLM LA2019 | model={model_id} | freeze={freeze_encoder} | device={device}"
    )
    print(
        "Paper target (ASVspoof2019 LA, not replay): "
        f"{PAPER_LA_EER_PERCENT:.2f}% EER — doi:10.1145/3708597.3708621"
    )

    train_records, val_records, skipped = load_la_train_val(
        la_root=la_root,
        cache_dir=cache_dir,
        validation_fraction=validation_fraction,
        seed=seed,
        max_train=max_train,
        max_val=max_val,
        refresh_cache=refresh_cache,
    )
    if skipped:
        (output_dir / "la_train_skipped_utts.txt").write_text(
            "\n".join(skipped) + "\n", encoding="utf-8"
        )

    train_summary = count_summary(train_records)
    val_summary = count_summary(val_records)
    print(f"Train: {json.dumps(train_summary)}")
    print(f"Val:   {json.dumps(val_summary)}")
    (output_dir / "data_summary.json").write_text(
        json.dumps({"train": train_summary, "val": val_summary}, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader = build_la_loaders(
        la_root,
        train_records,
        val_records,
        sample_rate=audio_config.sample_rate,
        num_samples=audio_config.samples,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )

    model = WavLMSpoofDetector(
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
    ckpt_path = output_dir / "best_wavlm_la2019.pt"
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
                extra={"model_type": "wavlm_asp_fc", "paper_eer_percent": PAPER_LA_EER_PERCENT},
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
                    "split": "la_train_speaker_val",
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
    print(
        f"Done. Best speaker-val EER={best_eer * 100:.2f}%. "
        f"Paper LA EER={PAPER_LA_EER_PERCENT:.2f}%. Checkpoint: {ckpt_path}"
    )
    return ckpt_path
