"""Speech enhancement front-ends (WebRTC, Wave-U-Net, pass-through)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    import webrtc_noise_gain as wng

    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    print("[WARN] webrtc_noise_gain not installed — WebRTC disabled.")


# ============================================================================
# WebRTC (unchanged)
# ============================================================================
class WebRTCEnhancer:
    def __init__(self, noise_suppression_level: int = 4):
        if not _WEBRTC_AVAILABLE:
            raise RuntimeError("webrtc_noise_gain is not installed.")
        self.ns_level = noise_suppression_level

    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        processor = wng.AudioProcessor(noise_suppression=self.ns_level)
        int16 = (waveform * 32767).clamp(-32768, 32767).short().numpy()
        frame_samples = 160
        frames = [int16[i : i + frame_samples] for i in range(0, len(int16), frame_samples)]
        out = bytearray()
        for f in frames:
            if len(f) < frame_samples:
                f = np.pad(f, (0, frame_samples - len(f)))
            res = processor.Process10ms(f.tobytes())
            out.extend(res.audio if res.is_speech else f.tobytes())
        out_np = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767.0
        out_t = torch.from_numpy(out_np)
        if out_t.shape[-1] < waveform.shape[-1]:
            out_t = F.pad(out_t, (0, waveform.shape[-1] - out_t.shape[-1]))
        else:
            out_t = out_t[: waveform.shape[-1]]
        return out_t


# ============================================================================
# Pass-through (unchanged)
# ============================================================================
class PassThroughEnhancer:
    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform


# ============================================================================
# NEW: Wave-U-Net enhancer + SNR gate, loaded from the fine-tuned v4 checkpoint
# ============================================================================
class SNRGateNet(nn.Module):
    """Must match the v4 training-script definition exactly."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(16, 32, 15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(32, hidden, 15, stride=4, padding=7),
            nn.GELU(),
        )
        self.total_stride = 4 * 4 * 4
        self.head = nn.Linear(hidden, 1)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        B, T = wav.shape
        h = self.conv(wav.unsqueeze(1))
        Tp = h.shape[-1]
        real_len = (wav_lens * T / self.total_stride).round().long().clamp(min=1, max=Tp)
        ar = torch.arange(Tp, device=wav.device).unsqueeze(0)
        mask = (ar < real_len.unsqueeze(1)).float().unsqueeze(1)
        pooled = (h * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        g = torch.sigmoid(self.head(pooled)).squeeze(-1)
        return g


class WaveUNetEnhancer:
    """
    Fine-tuned Wave-U-Net enhancer, single-file inference.

    Interface matches WebRTCEnhancer/PassThroughEnhancer:
        process(waveform_1d_cpu) -> waveform_1d_cpu (same length, same SR).

    Internally:
      16 kHz -> 48 kHz -> chunked Wave-U-Net -> residual subtraction -> 48 kHz -> 16 kHz.

    The v5/v6 training pipeline (STFT + SI-SDR + ID + PROTO + CLEAN losses,
    no gate) returns this enhanced signal directly -- fusion.py's
    SelfAttentionFusion is what decides how much to trust it per-example,
    not a fixed SNR heuristic. The older v4_long checkpoint (11 losses +
    SNRGateNet) is still supported for backward compatibility: if the
    checkpoint contains a `gate_net_state_dict`, it's loaded and applied;
    if not (the current/final checkpoint), the gate is skipped entirely
    rather than falling back to a random-init one.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        chunk_bs: int = 32,
        model_id: str = "wrice/waveunet-vctk-48khz-audiomentations",
    ):
        import json
        import re

        from denoisers import WaveUNetConfig, WaveUNetModel
        from huggingface_hub import hf_hub_download

        self.device = device
        self.chunk_bs = chunk_bs
        self.sr_in = 16000

        # --- load config ---
        with open(hf_hub_download(model_id, "config.json")) as f:
            cfg_dict = json.load(f)

        cfg = WaveUNetConfig(**cfg_dict)
        self.sr_model = cfg_dict["sample_rate"]  # 48000
        self.max_len = cfg_dict["max_length"]

        # --- instantiate and load enhancer ---
        self.enhancer = WaveUNetModel(cfg).to(device)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        enh_state = ckpt.get("enhancer_state_dict", ckpt.get("state_dict", None))
        if enh_state is None:
            raise RuntimeError(f"No enhancer_state_dict in {checkpoint_path}")

        # Key remap, needed only if loading a raw (pre-fine-tuning) HuggingFace
        # checkpoint. A no-op on the v5/v6 fine-tuned checkpoint, since its
        # keys were already remapped once during training before being saved
        # via enhancer.state_dict() -- harmless to leave in for either case.
        new_sd = {}
        for k, v in enh_state.items():
            k2 = k
            if (".encoder_layers." in k2 or ".decoder_layers." in k2) and ".batch_norm." in k2:
                k2 = k2.replace(".batch_norm.", ".norm.norm.")
            m = re.match(
                r"^(model\.middle\.\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$",
                k2,
            )
            if m:
                k2 = f"{m.group(1)}.norm.{m.group(2)}"
            new_sd[k2] = v

        miss, unexp = self.enhancer.load_state_dict(new_sd, strict=False)
        if miss or unexp:
            print(f"[WaveUNetEnhancer] enhancer load: missing={len(miss)} unexpected={len(unexp)}")
        self.enhancer.eval()
        for p in self.enhancer.parameters():
            p.requires_grad = False

        # --- gate: optional, only present in legacy v4_long checkpoints ---
        gate_state = ckpt.get("gate_net_state_dict")
        if gate_state is not None:
            self.gate_net = SNRGateNet().to(device)
            gm, gu = self.gate_net.load_state_dict(gate_state, strict=False)
            if gm or gu:
                print(f"[WaveUNetEnhancer] gate load: missing={len(gm)} unexpected={len(gu)}")
            self.gate_net.eval()
            for p in self.gate_net.parameters():
                p.requires_grad = False
            print(
                "[WaveUNetEnhancer] gate found in checkpoint -- using gated blend (legacy v4_long mode)."
            )
        else:
            self.gate_net = None
            print(
                "[WaveUNetEnhancer] no gate in checkpoint -- returning pure enhancer output "
                "(v5/v6 pipeline; fusion.py handles the noisy/enhanced blend instead)."
            )

    @staticmethod
    def _unwrap(out):
        if hasattr(out, "audio"):
            return out.audio
        if hasattr(out, "waveform"):
            return out.waveform
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            return next(v for v in out.values() if torch.is_tensor(v))
        return out

    @torch.inference_mode()
    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: 1-D CPU float32 tensor at 16 kHz. Returns same-shape CPU tensor."""
        dev = self.device
        wav = waveform.detach().float().to(dev).flatten()
        T_in = wav.numel()
        if T_in < 400:  # nothing meaningful to enhance
            return waveform.cpu()

        # 16 kHz -> 48 kHz
        wav_48 = torchaudio.functional.resample(wav.unsqueeze(0), self.sr_in, self.sr_model)
        T48 = wav_48.shape[-1]

        # Pad to a whole number of MAX_LEN chunks and reshape
        pad = (self.max_len - (T48 % self.max_len)) % self.max_len
        wav_48p = F.pad(wav_48, (0, pad)) if pad else wav_48
        n_chunks = wav_48p.shape[-1] // self.max_len
        chunks = wav_48p.view(n_chunks, self.max_len).unsqueeze(1)  # (N, 1, MAX_LEN)

        # Chunked forward, residual subtraction
        enh_chunks = []
        autocast_on = dev.startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_on):
            for i in range(0, n_chunks, self.chunk_bs):
                c = chunks[i : i + self.chunk_bs]
                out = self.enhancer(c)
                noise = self._unwrap(out).reshape(c.shape[0], -1)[:, : self.max_len].float()
                enh_chunks.append(c.squeeze(1).float() - noise)
        enh_48 = torch.cat(enh_chunks, dim=0).reshape(1, -1)[:, :T48]

        # 48 kHz -> 16 kHz
        enh_16 = torchaudio.functional.resample(enh_48, self.sr_model, self.sr_in)
        if enh_16.shape[-1] > T_in:
            enh_16 = enh_16[..., :T_in]
        elif enh_16.shape[-1] < T_in:
            enh_16 = F.pad(enh_16, (0, T_in - enh_16.shape[-1]))

        if self.gate_net is not None:
            # Legacy v4_long path: SNR-gated blend with the original noisy input.
            wav_lens = torch.ones(1, device=dev)
            g = self.gate_net(wav.unsqueeze(0), wav_lens).float()  # (1,)
            final = wav.unsqueeze(0) + g.unsqueeze(-1) * (enh_16 - wav.unsqueeze(0))
        else:
            # v5/v6 path: pure enhancer output. fusion.py's SelfAttentionFusion
            # blends this against the noisy embedding downstream, per-example.
            final = enh_16

        return final.squeeze(0).cpu()


# ============================================================================
# Factory
# ============================================================================
def get_enhancer(mode: str = "webrtc", **kwargs):
    if mode == "webrtc":
        return WebRTCEnhancer(**kwargs)
    if mode == "waveunet":
        return WaveUNetEnhancer(**kwargs)
    if mode == "none":
        return PassThroughEnhancer()
    raise ValueError(f"Unknown enhancement mode: {mode}")
