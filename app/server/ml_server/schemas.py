"""Pydantic response / request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    success: bool = True
    message: str
    device: str
    model: str


class EmbedResponse(BaseModel):
    success: bool = True
    num_files: int
    embedding_dim: int
    embeddings: list[list[float]]


class EnrollTemplateResponse(BaseModel):
    success: bool = True
    num_samples: int
    embedding_dim: int
    embedding: list[float] = Field(
        description="Averaged L2-normalized enrollment template"
    )


class VerifyResponse(BaseModel):
    success: bool = True
    score: float
    threshold: float
    accepted: bool
    decision: str
    rms: float | None = None


class SpoofStageResult(BaseModel):
    score: float
    threshold: float
    threshold_low: float | None = None
    threshold_high: float | None = None
    decision: str
    feature_type: str | None = None


class ReplayDetectResponse(BaseModel):
    success: bool = True
    score: float
    threshold: float
    threshold_low: float | None = None
    threshold_high: float | None = None
    is_replay: bool
    is_synthetic: bool = False
    accepted: bool
    decision: str
    feature_type: str | None = None
    rms: float | None = None
    replay: SpoofStageResult | None = None
    la: SpoofStageResult | None = None
