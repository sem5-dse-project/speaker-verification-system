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

    # SpeechBrain LazyModule + Windows inspect paths break later transformers imports.
    try:
        from speechbrain.utils.importutils import LazyModule
        import importlib
        import inspect as _inspect
        import sys
        import warnings

        if not getattr(LazyModule.ensure_module, "_sv_win_patch", False):

            def ensure_module(self, stacklevel: int):  # type: ignore[no-untyped-def]
                importer_frame = None
                try:
                    importer_frame = _inspect.getframeinfo(
                        sys._getframe(stacklevel + 1)
                    )
                except AttributeError:
                    warnings.warn(
                        "Failed to inspect frame for SpeechBrain lazy import guard."
                    )
                if importer_frame is not None:
                    filename = importer_frame.filename.replace("\\", "/")
                    if filename.endswith("/inspect.py"):
                        raise AttributeError()
                if self.lazy_module is None:
                    try:
                        if self.package is None:
                            self.lazy_module = importlib.import_module(self.target)
                        else:
                            self.lazy_module = importlib.import_module(
                                f".{self.target}", self.package
                            )
                    except Exception as e:
                        raise ImportError(
                            f"Lazy import of {repr(self)} failed"
                        ) from e
                return self.lazy_module

            ensure_module._sv_win_patch = True  # type: ignore[attr-defined]
            LazyModule.ensure_module = ensure_module  # type: ignore[method-assign]
    except Exception:
        pass

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
        fusion_out = fusion_model(noisy_emb, enhanced_emb)

        # Handle models that return tuples (like NoiseAwareFusion) vs single tensors
        if isinstance(fusion_out, tuple):
            fused = fusion_out[0]
        else:
            fused = fusion_out

        embs.append(fused.squeeze(0).cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)
