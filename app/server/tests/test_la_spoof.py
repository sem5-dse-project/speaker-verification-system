"""Unit tests for LA band helpers (no Hugging Face download)."""

from ml_server.la_spoof import decide_la_band, resolve_la_band_thresholds


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
