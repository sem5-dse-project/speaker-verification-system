# Voice Auth ML Server (Python / FastAPI + ECAPA-TDNN)

Stateless ML core for the voice authentication system.

- **Express** (`app/backend`): users, JWT, WAV file storage, MySQL  
- **This server** (`app/server`): Silero VAD speech extraction, anti-spoof gating, ECAPA embeddings, cosine verify

## Runtime pipeline

Incoming audio goes through:

1. **Silero VAD** (speech-only extraction, silence removal)
2. **Replay / anti-spoof** detection
3. **ECAPA-TDNN** embedding and scoring

This improves robustness by removing non-speech regions before replay and speaker checks.

Model: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health + device/model info |
| `POST` | `/embed` | One embedding per uploaded file |
| `POST` | `/enroll/template` | Average template from N enrollment files (use 3) |
| `POST` | `/verify` | Probe audio + template JSON → score / ACCEPT|REJECT |
| `POST` | `/replay/detect` | Silero VAD + anti-spoof → LIVE\|UNCERTAIN\|REPLAY\|SYNTHETIC\|NO_SPEECH |

Default replay weights: mixed ASVspoof **2017 + PA2019** **inverted-Mel** checkpoint  
(`replay-cnn-baseline/experiments/inverted_mel_mixed_2017_pa2019/.../best_inverted_mel_mixed_2017_pa2019.pt`).  
(LFCC is available via `REPLAY_CHECKPOINT` but tends to false-REPLAY on browser mics.)  
Override with `REPLAY_CHECKPOINT` / `REPLAY_THRESHOLD` / `REPLAY_MARGIN` (or `REPLAY_T_LOW` + `REPLAY_T_HIGH`) in `.env`.

Banding: `score < t_low` → LIVE, `t_low ≤ score < t_high` → UNCERTAIN (re-record), `score ≥ t_high` → REPLAY.  
Default margin is `0.10` around the checkpoint EER threshold.

Interactive docs: `http://localhost:8000/docs`

## Setup

```powershell
cd D:\speaker-verification-system\app\server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
copy .env.example .env
```

First request downloads ECAPA weights into `pretrained_models/` (gitignored).

## Unit tests

```powershell
cd D:\speaker-verification-system\app\server
.\.venv\Scripts\Activate.ps1
pytest
```

Tests cover scoring, WAV decode, schemas, and FastAPI routes with the ECAPA encoder **mocked** (no model download required).

VAD tests cover speech, silence, pauses, noisy input, too-short clips, invalid audio, and no-speech responses.

## Run

```powershell
cd D:\speaker-verification-system\app\server
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

## Example: build enrollment template (3 WAVs)

```powershell
curl -X POST http://localhost:8000/enroll/template `
  -F "files=@D:\path\enroll1.wav" `
  -F "files=@D:\path\enroll2.wav" `
  -F "files=@D:\path\enroll3.wav"
```

Response includes `embedding` (float array). Express should store this in MySQL for the user.

## Example: verify

```powershell
curl -X POST http://localhost:8000/verify `
  -F "file=@D:\path\probe.wav" `
  -F "embedding=[0.01, -0.02, ...]" `
  -F "threshold=0.25"
```

## How Express should call this later

1. User uploads 3 enrollment WAVs → Express saves files  
2. Express `POST`s those files to `/enroll/template`  
3. Express stores returned `embedding` for that user  
4. On verify, Express sends probe WAV + stored embedding to `/verify`  
5. Express returns `decision` / `score` to the frontend  

## Layout

```text
server/
├── main.py                 # FastAPI app
├── ml_server/
│   ├── config.py
│   ├── audio.py
│   ├── ecapa.py
│   ├── scoring.py
│   └── schemas.py
├── requirements.txt
├── .env.example
├── pretrained_models/      # downloaded weights (local)
└── README.md
```

## Notes

- Default threshold `0.25` (tune from EER experiments; LibriSpeech/Sinhala runs were ~0.28–0.29).
- Audio is converted to mono 16 kHz; clips longer than `MAX_SECONDS` are center-cropped.
- Silero VAD runs before replay/ECAPA and is CPU-friendly with ONNX (`VAD_USE_ONNX=true`).

## VAD configuration

Use `.env` values to tune speech extraction behavior:

- `VAD_SPEECH_THRESHOLD`: frame-level speech probability threshold.
- `VAD_MIN_SPEECH_MS`: minimum duration of a speech segment.
- `VAD_MIN_SILENCE_MS`: silence required to split adjacent segments.
- `VAD_SPEECH_PAD_MS`: padding added around each speech segment.
- `VAD_MIN_TOTAL_SPEECH_MS`: minimum total retained speech duration.
- `VAD_MIN_AUDIO_MS`: fail fast for very short clips.
