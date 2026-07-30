"""Schema smoke tests."""

from __future__ import annotations

from ml_server.schemas import HealthResponse, VerifyResponse


def test_health_response_defaults():
    model = HealthResponse(message="ok", device="cpu", model="ecapa")
    assert model.success is True


def test_verify_response_fields():
    model = VerifyResponse(
        score=0.9,
        threshold=0.25,
        accepted=True,
        decision="ACCEPT",
    )
    assert model.decision == "ACCEPT"
