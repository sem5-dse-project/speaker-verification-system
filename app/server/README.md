# Voice Auth ML Server (Python / FastAPI + ECAPA-TDNN)

Stateless ML core for the voice authentication system.

- **Express** (`app/backend`): users, JWT, WAV file storage, MySQL  
- **This server** (`app/server`): ECAPA embeddings, average enrollment template, cosine verify  

Model: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health + device/model info |
| `POST` | `/embed` | One embedding per uploaded file |
| `POST` | `/enroll/template` | Average template from N enrollment files (use 3) |
| `POST` | `/verify` | Probe audio + template JSON → score / ACCEPT|REJECT |

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
- Replay detection / enhancement can be added as extra routes later.
