"""FastAPI route tests with ECAPA embedding mocked (no model download)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
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
    def fake_score_replay(wave, threshold=None, device="cpu"):
        return {
            "score": 0.1,
            "threshold": 0.76,
            "is_replay": False,
            "accepted": True,
            "decision": "LIVE",
            "feature_type": "inverted_mel",
        }

    monkeypatch.setattr(main, "score_replay", fake_score_replay)
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


def test_replay_detect_replay(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_score_replay(wave, threshold=None, device="cpu"):
        return {
            "score": 0.9,
            "threshold": 0.76,
            "is_replay": True,
            "accepted": False,
            "decision": "REPLAY",
            "feature_type": "inverted_mel",
        }

    monkeypatch.setattr(main, "score_replay", fake_score_replay)
    monkeypatch.setattr(main, "REPLAY_ENABLED", True)
    wav = make_wav_bytes()
    response = client.post(
        "/replay/detect",
        files={"file": ("probe.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "REPLAY"
