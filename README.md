# Development of a Voice-Based Speaker Verification System for Secure User Authentication

**Status: project scaffold only** — folder structure, interfaces, configs, documentation, placeholders, and unit tests. No datasets downloaded, no model weights bundled, and no full training loops yet.

## Problem

Passwords and one-time codes are easy to share or steal. Voice biometrics can authenticate a claimed user conveniently, but real deployments face **background noise** and **replay attacks** (playback of a recorded voice). A practical system must verify identity while rejecting suspicious replayed audio and remaining robust under noisy conditions.

## Proposed solution

An end-to-end pipeline that:

1. **Enrolls** users by averaging L2-normalized speaker embeddings into a stored template.
2. **Detects replay** with a Log-Mel CNN before verification scoring.
3. For likely live audio, extracts **ECAPA-TDNN** embeddings from both original and **enhanced** speech.
4. Estimates a **5-D quality vector** and uses a **Quality-Conditioned Fusion Gate** to blend embeddings:

```text
e_fused = α · e_original + (1 − α) · e_enhanced
```

5. Compares the fused embedding to the enrollment template with **cosine similarity** and a **calibrated threshold**.

## Simplified architecture

```text
Enrollment:  audio → preprocess → encode → L2-norm → average → template(user_id)

Verification:
  audio → preprocess → replay CNN ──(reject)──► REJECT
                         │ live
                         ▼
              encode(original) + enhance → encode(enhanced)
                         │
                    quality vector → FusionGate(α)
                         │
                    fused embedding → cosine(template) → threshold → ACCEPT/REJECT
```

See [docs/architecture.md](docs/architecture.md) for details.

## Enrollment flow

1. Load multiple enrollment recordings.
2. Convert to mono, 16 kHz, float32.
3. Extract speaker embeddings.
4. L2-normalize and average.
5. Store the enrollment template with the user ID.

## Verification flow

1. Capture or load verification audio.
2. Apply basic audio preprocessing.
3. Estimate replay probability with the replay CNN.
4. Reject suspicious replay audio.
5. For likely live audio:
   - Extract an ECAPA-TDNN embedding from the original audio.
   - Apply frozen speech enhancement.
   - Extract another ECAPA-TDNN embedding from the enhanced audio.
   - Calculate a five-dimensional audio-quality vector.
   - Fuse with the Quality-Conditioned Fusion Gate.
6. Compare the fused embedding to the enrollment template (cosine similarity).
7. Apply a calibrated threshold to accept or reject.

## Dataset plan

| Dataset | Use |
|---------|-----|
| OpenSLR SLR52 Sinhala (FLAC) | Speaker enrollment / verification |
| ASVspoof 2017 Version 2 | Replay detection |
| Manageable MUSAN or RIRS_NOISES subset | Noise augmentation (20/10/5/0 dB) |
| Consent-based local Sinhala live/replay | Realistic evaluation |

**Do not commit** datasets, generated features, model checkpoints, or experiment outputs. See [docs/datasets.md](docs/datasets.md).

## Team responsibilities

| Member | Scope | Branch |
|--------|-------|--------|
| **HERATH H.M.M.M** | Pre-trained x-vector & ECAPA-TDNN; trials; enrollment; cosine scoring; threshold calibration; model comparison | `feature/speaker-verification` |
| **HERATH H.M.M.P.B** | ASVspoof processing; Log-Mel features; replay CNN; threshold tuning; error analysis | `feature/replay-detection` |
| **HEWARATHNA Y.M.** | Speech enhancement & noise augmentation; quality features; Fusion Gate; path comparison | `feature/quality-fusion` |

Details: [docs/team_workflow.md](docs/team_workflow.md).

## Folder structure

```text
speaker-verification-system/
├── src/voice_auth/
│   ├── common/                 # interfaces, audio, metrics, types
│   ├── speaker_verification/   # Member 1
│   ├── replay_detection/       # Member 2
│   ├── quality_fusion/         # Member 3
│   └── pipeline/               # enrollment + verification orchestration
├── configs/                    # YAML configs
├── scripts/                    # CLI entrypoints (placeholders)
├── tests/                      # lightweight unit tests
├── data/                       # local datasets only (gitignored)
├── checkpoints/                # local weights only (gitignored)
├── outputs/                    # experiment outputs (gitignored)
└── docs/                       # architecture, datasets, eval, workflow
```

## Environment setup (Python 3.11)

```bash
# From the repository root
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

Copy `.env.example` to `.env` and set local dataset/checkpoint paths as needed.

## Running tests

App unit tests (also run automatically on PRs via GitHub Actions — see `.github/workflows/ci.yml`):

```bash
# Express backend
cd app/backend && npm ci && npm test

# React frontend
cd app/frontend && npm ci && npm test

# Python ML server
cd app/server && pip install -r requirements.txt -r requirements-dev.txt && pytest
```

Root / research-area tests:

```bash
pytest
```

Tests use mock encoders and synthetic waveforms. They do **not** require datasets, internet access, or model weights.

Optional quality checks:

```bash
ruff check src tests scripts
black --check src tests scripts
```

## Git branch and pull-request workflow

1. Create/update your feature branch (`feature/speaker-verification`, `feature/replay-detection`, or `feature/quality-fusion`).
2. Keep commits focused; do not add audio, weights, or `.env` files.
3. Push and open a pull request into `main`.
4. Request review from the other members before merging.
5. Resolve review comments; ensure `pytest` passes.

## Offline experiments (to implement later)

1. Compare pre-trained x-vector and ECAPA-TDNN (EER, FAR, FRR, inference time).
2. Compare original, enhanced, fixed 50/50 fusion, and quality-conditioned fusion.
3. Evaluate clean, 20, 10, 5, and 0 dB conditions.
4. Evaluate replay detection (EER, precision, recall, F1, confusion matrix).

Protocol notes: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## License

MIT — see [LICENSE](LICENSE).
