"""Speech enhancement front‑ends (WebRTC, etc.)."""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import webrtc_noise_gain as wng

    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    print("[WARN] webrtc_noise_gain not installed – WebRTC disabled.")


class WebRTCEnhancer:
    def __init__(self, noise_suppression_level: int = 4):
        if not _WEBRTC_AVAILABLE:
            raise RuntimeError("webrtc_noise_gain is not installed.")
        self.ns_level = noise_suppression_level

    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: 1D float32 tensor on CPU, range [-1, 1]
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


class PassThroughEnhancer:
    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform


def get_enhancer(mode: str = "webrtc", **kwargs):
    if mode == "webrtc":
        return WebRTCEnhancer(**kwargs)
    elif mode == "none":
        return PassThroughEnhancer()
    else:
        raise ValueError(f"Unknown enhancement mode: {mode}")
