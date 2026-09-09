"""Unit tests for LA band helpers and AASIST padding (no model download)."""

import numpy as np
import pytest

from ml_server.aasist_model import AASIST_CUT, pad_aasist
from ml_server.la_spoof import decide_la_band, get_la_detector, resolve_la_band_thresholds


def test_decide_la_band_mapping():
    assert decide_la_band(0.1, 0.4, 0.6) == "LIVE"
    assert decide_la_band(0.5, 0.4, 0.6) == "UNCERTAIN"
    assert decide_la_band(0.9, 0.4, 0.6) == "SYNTHETIC"


def test_resolve_la_band_thresholds_margin(monkeypatch):
    monkeypatch.setattr("ml_server.la_spoof.LA_T_LOW", None)
    monkeypatch.setattr("ml_server.la_spoof.LA_T_HIGH", None)
    monkeypatch.setattr("ml_server.la_spoof.LA_MARGIN", 0.10)
    center, t_low, t_high = resolve_la_band_thresholds(0.50)
    assert center == 0.50
    assert abs(t_low - 0.40) < 1e-6
    assert abs(t_high - 0.60) < 1e-6


def test_pad_aasist_length():
    short = np.zeros(1000, dtype=np.float32)
    padded = pad_aasist(short)
    assert padded.shape == (AASIST_CUT,)
    long = np.arange(AASIST_CUT + 500, dtype=np.float32)
    trimmed = pad_aasist(long)
    assert trimmed.shape == (AASIST_CUT,)
    assert float(trimmed[-1]) == float(long[AASIST_CUT - 1])


def test_get_la_detector_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr("ml_server.la_spoof.LA_BACKEND", "nope")
    monkeypatch.setattr("ml_server.la_spoof._la_model", None)
    with pytest.raises(ValueError, match="Unsupported LA_BACKEND"):
        get_la_detector(device="cpu")
