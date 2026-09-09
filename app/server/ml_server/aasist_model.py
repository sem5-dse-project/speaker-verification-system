"""Official Clova AASIST loader for the ML server LA stage.

Expects a local clone of https://github.com/clovaai/aasist at ``AASIST_ROOT``
(default: ``<repo>/aasist``) with ``models/weights/AASIST.pth``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Official AASIST cut (~4.0375 s @ 16 kHz)
AASIST_CUT = 64600
DEFAULT_AASIST_THRESHOLD = 0.5


@dataclass
class AASISTAudioConfig:
    sample_rate: int = 16000
    samples: int = AASIST_CUT

    @property
    def seconds(self) -> float:
        return float(self.samples) / float(self.sample_rate)


def pad_aasist(x: np.ndarray, max_len: int = AASIST_CUT) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x_len = int(x.shape[0])
    if x_len >= max_len:
        return x[:max_len]
    n_rep = int(max_len / max(x_len, 1)) + 1
    return np.tile(x, n_rep)[:max_len]


def ensure_aasist_on_path(aasist_root: Path) -> Path:
    root = Path(aasist_root).resolve()
    if not (root / "models" / "AASIST.py").exists():
        raise FileNotFoundError(
            f"AASIST repo not found at {root}. "
            "Clone https://github.com/clovaai/aasist into the repo root as `aasist/`."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_aasist_weight(aasist_root: Path, weight: Path | None = None) -> Path:
    root = Path(aasist_root)
    if weight is not None and Path(weight).is_file():
        return Path(weight)
    cfg_path = root / "config" / "AASIST.conf"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        candidate = root / config.get("model_path", "models/weights/AASIST.pth")
        if candidate.is_file():
            return candidate
    fallback = root / "models" / "weights" / "AASIST.pth"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"AASIST weights missing under {root}. "
        "Expected models/weights/AASIST.pth (see AASIST.conf model_path)."
    )


def load_aasist_model(
    aasist_root: Path,
    *,
    weight: Path | None = None,
    device: str | torch.device = "cpu",
):
    """Load pretrained AASIST. Softmax class 0 = spoof, class 1 = bonafide."""
    root = ensure_aasist_on_path(aasist_root)
    weight_path = resolve_aasist_weight(root, weight)
    cfg_path = root / "config" / "AASIST.conf"
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_cfg = config["model_config"]
    module = import_module(f"models.{model_cfg['architecture']}")
    model = module.Model(model_cfg)
    state = torch.load(str(weight_path), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, weight_path, AASISTAudioConfig()


@torch.inference_mode()
def score_aasist_p_spoof(
    model,
    wave_1d: torch.Tensor | np.ndarray,
    device: str | torch.device,
) -> float:
    """Return P(spoof) in [0, 1] for a mono waveform."""
    if isinstance(wave_1d, torch.Tensor):
        arr = wave_1d.detach().float().cpu().numpy()
    else:
        arr = np.asarray(wave_1d, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    x = pad_aasist(arr)
    batch = torch.from_numpy(x).float().unsqueeze(0).to(device)
    _, logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return float(probs[0, 0].item())
