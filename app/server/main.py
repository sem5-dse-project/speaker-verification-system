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
from ml_server.anti_spoof import score_anti_spoof
from ml_server.audio import extract_speech_audio, has_sufficient_speech, load_audio_bytes
from ml_server.config import (
    ALLOWED_ORIGINS,
    DEFAULT_THRESHOLD,
    DEVICE,
    ECAPA_SOURCE,
    ENHANCEMENT_MODE,
    FUSION_ENABLED,
    FUSION_MODEL_PATH,
    FUSION_MODEL_TYPE,
    HOST,
    LA_CHECKPOINT,
    LA_ENABLED,
    LA_HARD_GATE,
    PORT,
    REPLAY_CHECKPOINT,
    REPLAY_ENABLED,
    REPLAY_THRESHOLD,
)
from ml_server.ecapa import embed_audio_list, embed_audio_list_fused, load_ecapa_encoder
from ml_server.enhancement import get_enhancer
from ml_server.fusion import load_fusion_model
from ml_server.schemas import (
    EmbedResponse,
    EnrollTemplateResponse,
    HealthResponse,
    ReplayDetectResponse,
    VerifyResponse,
)
from ml_server.scoring import average_template, cosine_similarity, decide

# --- Global variables (module level) ---
_encoder = None
_enhancer = None
_fusion_model = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = load_ecapa_encoder(device=DEVICE)
    return _encoder


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _enhancer, _fusion_model
    # Load enhancer
    if ENHANCEMENT_MODE == "webrtc":
        try:
            _enhancer = get_enhancer("webrtc")
        except Exception as e:
            print(f"WebRTC enhancer failed: {e}. Using pass-through.")
            _enhancer = get_enhancer("none")
    else:
        _enhancer = get_enhancer("none")
    # Load fusion model if enabled
    _fusion_model = None
    if FUSION_ENABLED and FUSION_MODEL_PATH.exists():
        try:
            _fusion_model = load_fusion_model(
                FUSION_MODEL_PATH, model_type=FUSION_MODEL_TYPE, device=DEVICE
            )

            print(f"Fusion model loaded: {FUSION_MODEL_TYPE} from {FUSION_MODEL_PATH}")
        except Exception as e:
            print(f"Fusion model load failed: {e}. Disabling fusion.")
            _fusion_model = None
    yield
    # cleanup (optional)


app = FastAPI(
    title="Voice Auth ML Server",
    description="ECAPA-TDNN enroll template + verify scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    replay_note = (
        f"; replay={'on' if REPLAY_ENABLED else 'off'} ({REPLAY_CHECKPOINT.name})"
        if REPLAY_CHECKPOINT
        else ""
    )
    la_note = (
        f"; la={'on' if LA_ENABLED else 'off'}"
        f"{'/hard' if LA_ENABLED and LA_HARD_GATE else '/soft' if LA_ENABLED else ''}"
        f" ({LA_CHECKPOINT.name})"
        if LA_CHECKPOINT
        else ""
    )
    return HealthResponse(
        message=f"ECAPA ML server is running{replay_note}{la_note}",
        device=DEVICE,
        model=ECAPA_SOURCE,
    )


@app.post("/replay/detect", response_model=ReplayDetectResponse)
async def replay_detect(
    file: Annotated[UploadFile, File(description="Probe audio (WAV)")],
    threshold: Annotated[float | None, Form()] = None,
    la_threshold: Annotated[float | None, Form()] = None,
) -> ReplayDetectResponse:
    """
    Anti-spoof cascade: inverted-Mel replay, then optional LFCC-LA synthetic.

    decision: LIVE | UNCERTAIN | REPLAY | SYNTHETIC | NO_SPEECH.
    Express rejects REPLAY/SYNTHETIC, asks re-record on UNCERTAIN/NO_SPEECH,
    then speaker-verifies LIVE.
    """
    if not REPLAY_ENABLED:
        raise HTTPException(status_code=503, detail="Replay detection is disabled")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        wave = load_audio_bytes(data)
        vad = extract_speech_audio(wave)
        if not vad.has_speech:
            threshold_value = (
                float(threshold)
                if threshold is not None
                else float(REPLAY_THRESHOLD)
                if REPLAY_THRESHOLD is not None
                else 0.5
            )
            return ReplayDetectResponse(
                score=0.0,
                threshold=threshold_value,
                is_replay=False,
                is_synthetic=False,
                accepted=False,
                decision="NO_SPEECH",
                feature_type="silero_vad",
                rms=vad.rms,
                speech_ms=vad.speech_ms,
                total_ms=vad.total_ms,
                num_speech_segments=vad.num_speech_segments,
            )

        wave = vad.speech_waveform
        result = score_anti_spoof(
            wave,
            threshold=threshold,
            la_threshold=la_threshold,
            device=DEVICE,
        )
        result["speech_ms"] = vad.speech_ms
        result["total_ms"] = vad.total_ms
        result["num_speech_segments"] = vad.num_speech_segments
        result["rms"] = vad.rms
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Speech preprocessing is unavailable on this server",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported audio input") from exc

    return ReplayDetectResponse(**result)


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    files: Annotated[list[UploadFile], File(description="One or more audio files")],
) -> EmbedResponse:
    """Return one embedding per uploaded file."""
    blobs = await _read_uploads(files)
    try:
        waves = []
        for blob in blobs:
            wave = load_audio_bytes(blob)
            vad = extract_speech_audio(wave)
            if not vad.has_speech:
                raise ValueError("No usable speech detected in one or more audio files")
            waves.append(vad.speech_waveform)
        encoder = get_encoder()

        if _fusion_model is not None:
            embs = embed_audio_list_fused(encoder, waves, _enhancer, _fusion_model, device=DEVICE)
        else:
            embs = embed_audio_list(encoder, waves, device=DEVICE)

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Speech preprocessing is unavailable on this server",
        ) from None
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
        waves = []
        for blob in blobs:
            wave = load_audio_bytes(blob)
            vad = extract_speech_audio(wave)
            if not vad.has_speech:
                raise ValueError("No usable speech detected in one or more audio files")
            waves.append(vad.speech_waveform)
        encoder = get_encoder()
        embs = embed_audio_list(encoder, waves, device=DEVICE)

        template = average_template(embs)
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Speech preprocessing is unavailable on this server",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported audio input") from exc

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
        vad = extract_speech_audio(wave)
        if not vad.has_speech:
            thr = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
            return VerifyResponse(
                score=0.0,
                threshold=thr,
                accepted=False,
                decision="NO_SPEECH",
                rms=vad.rms,
                speech_ms=vad.speech_ms,
                total_ms=vad.total_ms,
                num_speech_segments=vad.num_speech_segments,
            )
        wave = vad.speech_waveform
        encoder = get_encoder()

        # --- Use the same embedding pipeline as enrollment ---
        if _fusion_model is not None:
            probes = embed_audio_list_fused(
                encoder, [wave], _enhancer, _fusion_model, device=DEVICE
            )
        else:
            probes = embed_audio_list(encoder, [wave], device=DEVICE)
        probe = probes[0]  # shape (embedding_dim,)

        score = cosine_similarity(template, probe)
        result = decide(score, threshold)
        result["rms"] = vad.rms
        result["speech_ms"] = vad.speech_ms
        result["total_ms"] = vad.total_ms
        result["num_speech_segments"] = vad.num_speech_segments
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Speech preprocessing is unavailable on this server",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported audio input") from exc

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
