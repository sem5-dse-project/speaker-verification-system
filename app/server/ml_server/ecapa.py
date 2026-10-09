"""SpeechBrain ECAPA-TDNN loader, embedding, and frame-level split for v11 fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ml_server.config import DEVICE, ECAPA_SAVEDIR, ECAPA_SOURCE


# ============================================================================
# Loader (unchanged behaviour, Windows / LazyModule patches retained)
# ============================================================================
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
        import importlib
        import inspect as _inspect
        import sys
        import warnings

        from speechbrain.utils.importutils import LazyModule

        if not getattr(LazyModule.ensure_module, "_sv_win_patch", False):

            def ensure_module(self, stacklevel: int):  # type: ignore[no-untyped-def]
                importer_frame = None
                try:
                    importer_frame = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
                except AttributeError:
                    warnings.warn("Failed to inspect frame for SpeechBrain lazy import guard.")
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
                        raise ImportError(f"Lazy import of {repr(self)} failed") from e
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


# ============================================================================
# NEW: frame-level split of ECAPA (must match the v11 training code exactly)
#
# ECAPA-TDNN splits at the end of `mfa` into:
#   frames(wav, lens) -> (B, 3072, T)   per-frame features at ~100 fps
#   head(frames, lens) -> (B, 192)      ASP -> BN -> FC -> L2 normalize
#
# The v11 fusion trains a *copy* of `head` (TrainableHead in fusion.py).
# Both paths must be available in production.
# ============================================================================
class EcapaFrames:
    """Mirrors `EcapaFrames` from the v11 training script.

    Wraps a SpeechBrain `EncoderClassifier` and exposes its internal split
    point.  Use:
        ecf = EcapaFrames(classifier)
        hn  = ecf.frames(noisy_wav, wl)     # (B, 3072, T)
        en  = ecf.head(hn, wl)              # (B, 192) L2-normalised
    """

    def __init__(self, classifier):
        self.m = classifier.mods
        self.em = classifier.mods.embedding_model
        # TDNNBlock is the only layer in em.blocks that takes just `x`.
        from speechbrain.lobes.models.ECAPA_TDNN import TDNNBlock

        self._TDNN = TDNNBlock

    @torch.inference_mode()
    def frames(self, wav: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        """wav (B, S) float, wl (B,) relative lengths in [0, 1] -> (B, 3072, T)."""
        feats = self.m.compute_features(wav.float())
        feats = self.m.mean_var_norm(feats, wl)
        x = feats.transpose(1, 2)
        xl = []
        for layer in self.em.blocks:
            x = layer(x) if isinstance(layer, self._TDNN) else layer(x, lengths=wl)
            xl.append(x)
        return self.em.mfa(torch.cat(xl[1:], dim=1))

    @torch.inference_mode()
    def head(self, h: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        """(B, 3072, T) frames -> (B, 192) L2-normalised embedding (frozen head)."""
        x = self.em.asp(h, lengths=wl)
        x = self.em.asp_bn(x)
        x = self.em.fc(x).transpose(1, 2)  # (B, 1, 192)
        return F.normalize(x.squeeze(1).float(), dim=-1)


# ============================================================================
# Plain ECAPA embedding (raw path; used for enrollment)
# ============================================================================
@torch.inference_mode()
def encode_waveforms(classifier, waveforms: torch.Tensor) -> torch.Tensor:
    """waveforms [B, T] -> L2-normalized embeddings [B, D] (frozen ECAPA)."""
    if waveforms.dim() == 1:
        waveforms = waveforms.unsqueeze(0)
    emb = classifier.encode_batch(waveforms)
    if emb.dim() == 3:
        emb = emb.squeeze(1)
    return F.normalize(emb, dim=-1)


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


# ============================================================================
# v11 fused embedding path
# ============================================================================
@torch.inference_mode()
def embed_audio_list_fused_v11(
    classifier,
    ecf: EcapaFrames,
    waves: list[torch.Tensor],
    enhancer,
    fusion_model,  # a FrameGateFusion with .head_net attached
    device: str = DEVICE,
) -> np.ndarray:
    """Fused embedding for v11 (FrameGateFusion + trainable head).

    Pipeline per utterance (matching v11 training):
      noisy_wav   -> ecf.frames  -> hn  (B, 3072, T)
                    ecf.head(hn) -> en  (B, 192)         [frozen head, no grad]
      enh_wav     -> ecf.frames  -> he  (B, 3072, T)
                    ecf.head(he) -> ee  (B, 192)
      fused_wav_frames = fusion(hn, he, en, ee, wl)     -> (ff, gate, delta)
      pooled = fusion.head_net(ff, wl) + delta          -> (B, 192) L2 norm

    Note: `enhancer.process` returns a CPU tensor; we move it back to device.
    """
    if fusion_model is None:
        # No fusion: return raw embedding of the *enhanced* signal (still useful).
        return embed_audio_list(classifier, [enhancer.process(w) for w in waves], device)

    head = getattr(fusion_model, "head_net", None)
    if head is None:
        raise RuntimeError(
            "v11 pipeline requires fusion_model.head_net. "
            "Did you load the checkpoint with load_fusion_model(..., ecapa_frames=ecf)?"
        )

    embs = []
    for wave in waves:
        wav_in = wave.detach().float().flatten()
        wav_b = wav_in.unsqueeze(0).to(device)
        wl = torch.ones(1, device=device)

        # --- frame features of the noisy signal ---
        hn = ecf.frames(wav_b, wl)  # (1, 3072, T)

        # --- enhance + frame features of the enhanced signal ---
        enh_wav = enhancer.process(wav_in.cpu()).to(device).float().flatten()
        if enh_wav.numel() < wav_in.numel():
            enh_wav = F.pad(enh_wav, (0, wav_in.numel() - enh_wav.numel()))
        elif enh_wav.numel() > wav_in.numel():
            enh_wav = enh_wav[: wav_in.numel()]
        enh_b = enh_wav.unsqueeze(0)
        he = ecf.frames(enh_b, wl)  # (1, 3072, T)

        # --- embeddings from the FROZEN head ---
        en = ecf.head(hn, wl)  # (1, 192)
        ee = ecf.head(he, wl)  # (1, 192)

        # --- fusion ---
        ff, gate, delta = fusion_model(hn, he, en, ee, wl)
        pooled = head(ff, wl)  # (1, 192) trainable head
        fused = F.normalize(pooled + delta, dim=-1)

        embs.append(fused.squeeze(0).detach().cpu().numpy())

    return np.stack(embs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Backwards-compat alias: if you call the old name, route to the v11 path
# if a FrameGateFusion-like model was provided, otherwise fall back to the
# legacy (v6) two-embedding signature.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def embed_audio_list_fused(
    classifier,
    waves: list[torch.Tensor],
    enhancer,
    fusion_model,
    device: str = DEVICE,
    ecapa_frames: Optional[EcapaFrames] = None,
) -> np.ndarray:
    """Dispatch to the v11 pipeline when `ecapa_frames` + a frame-gate model are given,
    otherwise fall back to the legacy (v6) two-embedding pipeline."""
    if ecapa_frames is not None and getattr(fusion_model, "head_net", None) is not None:
        return embed_audio_list_fused_v11(
            classifier, ecapa_frames, waves, enhancer, fusion_model, device
        )

    # ---- legacy v6 path (kept for other model types) ----
    if fusion_model is None:
        return embed_audio_list(classifier, waves, device)

    embs = []
    for wave in waves:
        noisy_emb = encode_waveforms(classifier, wave.unsqueeze(0).to(device))
        enhanced_wave = enhancer.process(wave.cpu()).to(device)
        enhanced_emb = encode_waveforms(classifier, enhanced_wave.unsqueeze(0))
        fusion_out = fusion_model(noisy_emb, enhanced_emb)
        fused = fusion_out[0] if isinstance(fusion_out, tuple) else fusion_out
        embs.append(fused.squeeze(0).cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)
