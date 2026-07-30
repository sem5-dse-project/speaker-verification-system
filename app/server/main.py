"""FastAPI ECAPA core for enroll templates and verification scoring.

Designed to be called by the Express backend (stateless ML service).
Express owns MySQL user/auth/file storage; this service returns embeddings
and accept/reject decisions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml_server.audio import load_audio_bytes
from ml_server.config import (
    DEFAULT_THRESHOLD,
    DEVICE,
    ECAPA_SOURCE,
    HOST,
    PORT,
)
from ml_server.ecapa import embed_audio_list, load_ecapa_encoder
from ml_server.schemas import (
    EmbedResponse,
    EnrollTemplateResponse,
    HealthResponse,
    VerifyResponse,
)
from ml_server.scoring import average_template, cosine_similarity, decide

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = load_ecapa_encoder(device=DEVICE)
    return _encoder


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Lazy-load on first request is also fine; warm-up optional
    yield


app = FastAPI(
    title="Voice Auth ML Server",
    description="ECAPA-TDNN enroll template + verify scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_uploads(files: list[UploadFile]) -> list[bytes]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file is required")
    blobs: list[bytes] = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")
        blobs.append(data)
    return blobs


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        message="ECAPA ML server is running",
        device=DEVICE,
        model=ECAPA_SOURCE,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    files: Annotated[list[UploadFile], File(description="One or more audio files")],
) -> EmbedResponse:
    """Return one embedding per uploaded file."""
    blobs = await _read_uploads(files)
    try:
        waves = [load_audio_bytes(b) for b in blobs]
        encoder = get_encoder()
        embs = embed_audio_list(encoder, waves, device=DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EmbedResponse(
        num_files=len(embs),
        embedding_dim=int(embs.shape[1]),
        embeddings=embs.tolist(),
    )


@app.post("/enroll/template", response_model=EnrollTemplateResponse)
async def enroll_template(
    files: Annotated[
        list[UploadFile],
        File(description="Enrollment samples (recommend 3)"),
    ],
) -> EnrollTemplateResponse:
    """
    Embed several enrollment clips and return their averaged L2-normalized template.

    Express should store this vector in MySQL for the user.
    """
    blobs = await _read_uploads(files)
    try:
        waves = [load_audio_bytes(b) for b in blobs]
        encoder = get_encoder()
        embs = embed_audio_list(encoder, waves, device=DEVICE)
        template = average_template(embs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EnrollTemplateResponse(
        num_samples=len(blobs),
        embedding_dim=int(template.shape[0]),
        embedding=template.tolist(),
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    file: Annotated[UploadFile, File(description="Probe / verification audio")],
    embedding: Annotated[
        str,
        Form(description="JSON array string of enrollment template floats"),
    ],
    threshold: Annotated[float | None, Form()] = None,
) -> VerifyResponse:
    """
    Compare one probe audio against a stored enrollment template (JSON array).

    Form fields:
      - file: audio
      - embedding: e.g. "[0.01, -0.02, ...]"
      - threshold: optional cosine threshold (default from env)
    """
    import json

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        template = np.asarray(json.loads(embedding), dtype=np.float32)
        if template.ndim != 1 or template.size == 0:
            raise ValueError("embedding must be a 1-D JSON array of floats")
        wave = load_audio_bytes(data)
        encoder = get_encoder()
        probe = embed_audio_list(encoder, [wave], device=DEVICE)[0]
        score = cosine_similarity(template, probe)
        result = decide(score, threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VerifyResponse(**result)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
