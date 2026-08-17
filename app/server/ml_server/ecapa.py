"""SpeechBrain ECAPA-TDNN loader and embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ml_server.config import DEVICE, ECAPA_SAVEDIR, ECAPA_SOURCE
from ml_server.enhancement import PassThroughEnhancer


def load_ecapa_encoder(
    source: str = ECAPA_SOURCE,
    savedir: Path | None = None,
    device: str = DEVICE,
):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install speechbrain: pip install speechbrain huggingface_hub") from exc

    savedir = Path(savedir) if savedir is not None else ECAPA_SAVEDIR
    savedir.mkdir(parents=True, exist_ok=True)
    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=str(savedir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    classifier.eval()
    return classifier


@torch.inference_mode()
def encode_waveforms(classifier, waveforms: torch.Tensor) -> torch.Tensor:
    """waveforms [B, T] -> L2-normalized embeddings [B, D]."""
    if waveforms.dim() == 1:
        waveforms = waveforms.unsqueeze(0)
    emb = classifier.encode_batch(waveforms)
    if emb.dim() == 3:
        emb = emb.squeeze(1)
    return torch.nn.functional.normalize(emb, dim=-1)


@torch.inference_mode()
def embed_audio_list(
    classifier,
    waves: list[torch.Tensor],
    device: str = DEVICE,
) -> np.ndarray:
    embs = []
    for wave in waves:
        batch = wave.unsqueeze(0).to(device)
        emb = encode_waveforms(classifier, batch)
        embs.append(emb.squeeze(0).cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)


@torch.inference_mode()
def embed_audio_list_fused(
    classifier,
    waves: list[torch.Tensor],
    enhancer,
    fusion_model,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Return fused embeddings for a list of waveforms.
    If fusion_model is None, fallback to noisy embeddings.
    """
    if fusion_model is None:
        return embed_audio_list(classifier, waves, device)

    embs = []
    for wave in waves:
        # 1. Noisy embedding
        noisy_emb = encode_waveforms(classifier, wave.unsqueeze(0).to(device))
        # 2. Enhanced waveform & embedding
        enhanced_wave = enhancer.process(wave.cpu())
        enhanced_emb = encode_waveforms(classifier, enhanced_wave.unsqueeze(0).to(device))
        # 3. Fuse
        fused, _ = fusion_model(noisy_emb, enhanced_emb)

        embs.append(fused.squeeze(0).cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)
