"""FastAPI route tests with ECAPA embedding mocked (no model download)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from ml_server.vad import VadResult
from tests.conftest import make_wav_bytes


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_get_encoder():
        return object()

    def fake_embed_audio_list(_encoder, waves, device="cpu"):
        # Deterministic unit-ish embeddings: length = number of waves, dim = 4
        rows = []
        for i, _wave in enumerate(waves):
            vec = np.zeros(4, dtype=np.float32)
            vec[i % 4] = 1.0
            rows.append(vec)
        return np.stack(rows, axis=0)

    monkeypatch.setattr(main, "get_encoder", fake_get_encoder)
    monkeypatch.setattr(main, "embed_audio_list", fake_embed_audio_list)
    monkeypatch.setattr(
        main,
        "extract_speech_audio",
        lambda wave: VadResult(
            has_speech=True,
            speech_waveform=wave,
            total_ms=500.0,
            speech_ms=500.0,
            num_speech_segments=1,
            rms=0.1,
        ),
    )
    return TestClient(main.app)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "ECAPA" in body["message"]
    assert "device" in body
    assert "model" in body


def test_enroll_template(client: TestClient):
    wav = make_wav_bytes()
    files = [
        ("files", ("enroll_1.wav", wav, "audio/wav")),
        ("files", ("enroll_2.wav", wav, "audio/wav")),
        ("files", ("enroll_3.wav", wav, "audio/wav")),
    ]
    response = client.post("/enroll/template", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["num_samples"] == 3
    assert body["embedding_dim"] == 4
    assert len(body["embedding"]) == 4
    # Averaged L2-normalized template should have unit norm
    norm = float(np.linalg.norm(np.asarray(body["embedding"], dtype=np.float32)))
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_enroll_template_rejects_bad_audio(client: TestClient):
    files = [("files", ("bad.wav", b"not-wav", "audio/wav"))]
    response = client.post("/enroll/template", files=files)
    assert response.status_code == 400


def test_verify_accept(client: TestClient):
    # fake embed returns [1,0,0,0] for first wave → match same template
    template = [1.0, 0.0, 0.0, 0.0]
    wav = make_wav_bytes()
    response = client.post(
        "/verify",
        data={"embedding": json.dumps(template), "threshold": "0.25"},
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ACCEPT"
    assert body["accepted"] is True
    assert body["score"] == pytest.approx(1.0, abs=1e-4)


def test_verify_reject(client: TestClient):
    # Orthogonal to first-wave embedding [1,0,0,0]
    template = [0.0, 1.0, 0.0, 0.0]
    wav = make_wav_bytes()
    response = client.post(
        "/verify",
        data={"embedding": json.dumps(template), "threshold": "0.25"},
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "REJECT"
    assert body["accepted"] is False


def test_verify_empty_file(client: TestClient):
    response = client.post(
        "/verify",
        data={"embedding": "[1,0,0,0]"},
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_replay_detect_live(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_score_anti_spoof(wave, threshold=None, la_threshold=None, device="cpu", **_kwargs):
        return {
            "score": 0.1,
            "threshold": 0.74,
            "threshold_low": 0.64,
            "threshold_high": 0.84,
            "is_replay": False,
            "is_synthetic": False,
            "accepted": True,
            "decision": "LIVE",
            "feature_type": "inverted_mel+wavlm",
            "replay": {
                "score": 0.1,
                "threshold": 0.74,
                "threshold_low": 0.64,
                "threshold_high": 0.84,
                "decision": "LIVE",
                "feature_type": "inverted_mel",
            },
            "la": {
                "score": 0.05,
                "threshold": 0.5,
                "threshold_low": 0.4,
                "threshold_high": 0.6,
                "decision": "LIVE",
                "feature_type": "wavlm",
            },
        }

    monkeypatch.setattr(main, "score_anti_spoof", fake_score_anti_spoof)
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "LIVE"
    assert body["is_replay"] is False
    assert body["la"]["decision"] == "LIVE"


def test_replay_detect_uncertain(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_score_anti_spoof(wave, threshold=None, la_threshold=None, device="cpu", **_kwargs):
        return {
            "score": 0.72,
            "threshold": 0.74,
            "threshold_low": 0.64,
            "threshold_high": 0.84,
            "is_replay": False,
            "is_synthetic": False,
            "accepted": False,
            "decision": "UNCERTAIN",
            "feature_type": "inverted_mel+wavlm",
        }

    monkeypatch.setattr(main, "score_anti_spoof", fake_score_anti_spoof)
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "UNCERTAIN"
    assert body["accepted"] is False
    assert body["is_replay"] is False


def test_replay_detect_no_speech(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def never_called(*_args, **_kwargs):
        raise AssertionError("anti-spoof model should not run for no-speech clips")

    monkeypatch.setattr(main, "score_anti_spoof", never_called)
    monkeypatch.setattr(
        main,
        "extract_speech_audio",
        lambda wave: VadResult(
            has_speech=False,
            speech_waveform=wave,
            total_ms=1200.0,
            speech_ms=0.0,
            num_speech_segments=0,
            rms=0.001,
        ),
    )
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "NO_SPEECH"
    assert body["accepted"] is False
    assert body["num_speech_segments"] == 0


def test_verify_no_speech_from_vad(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        main,
        "extract_speech_audio",
        lambda wave: VadResult(
            has_speech=False,
            speech_waveform=wave,
            total_ms=1000.0,
            speech_ms=0.0,
            num_speech_segments=0,
            rms=0.002,
        ),
    )

    response = client.post(
        "/verify",
        data={"embedding": "[1,0,0,0]", "threshold": "0.25"},
        files={"file": ("probe.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "NO_SPEECH"
    assert body["accepted"] is False
    assert body["num_speech_segments"] == 0


def test_replay_detect_replay(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_score_anti_spoof(wave, threshold=None, la_threshold=None, device="cpu", **_kwargs):
        return {
            "score": 0.9,
            "threshold": 0.74,
            "threshold_low": 0.64,
            "threshold_high": 0.84,
            "is_replay": True,
            "is_synthetic": False,
            "accepted": False,
            "decision": "REPLAY",
            "feature_type": "inverted_mel",
        }

    monkeypatch.setattr(main, "score_anti_spoof", fake_score_anti_spoof)
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "REPLAY"


def test_replay_detect_synthetic(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_score_anti_spoof(wave, threshold=None, la_threshold=None, device="cpu", **_kwargs):
        return {
            "score": 0.95,
            "threshold": 0.5,
            "threshold_low": 0.4,
            "threshold_high": 0.6,
            "is_replay": False,
            "is_synthetic": True,
            "accepted": False,
            "decision": "SYNTHETIC",
            "feature_type": "inverted_mel+wavlm",
            "la": {
                "score": 0.95,
                "threshold": 0.5,
                "threshold_low": 0.4,
                "threshold_high": 0.6,
                "decision": "SYNTHETIC",
                "feature_type": "wavlm",
            },
        }

    monkeypatch.setattr(main, "score_anti_spoof", fake_score_anti_spoof)
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "SYNTHETIC"
    assert body["is_synthetic"] is True
