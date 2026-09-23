# Noise-aware gated SASV (publishable track)

Offline experiments: **ASVspoof 2019 LA + SASV trials**, with **test-side noise** and
**SNR-gated enhancement fusion** (Wave-U-Net + ECAPA).

> Not the Docker demo. Tune on **dev**, report **eval** once.

## Notebooks (4 steps)

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_protocol_and_harness.ipynb` | Freeze protocol; smoke offline ECAPA scoring |
| 2 | `02_noise_injection.ipynb` | MUSAN/white noise at fixed SNRs (test only) |
| 3 | `03_baselines_and_gated_systems.ipynb` | B0 / P2 / P3 under a chosen SNR |
| 4 | `04_matrix_tables_and_claim.ipynb` | Tables, plots, claim checklist |

Shared code: `noise_gated_lib.py` (imports helpers from `../sasv_la2019`).

## Systems

| ID | Meaning |
|----|---------|
| B0 | ECAPA only |
| B1 | ECAPA + AASIST weighted α=0.30 (from `sasv_la2019`, clean locked ~0.83%) |
| P2 | Gated raw/enhanced ECAPA (+ CM in full paper runs) |
| P3 | Always-enhance ECAPA |

## Prerequisites

- `data/LA`, `SASVC2022_Baseline/`, `app/server` venv (SpeechBrain, etc.)
- Optional: MUSAN noise root; Wave-U-Net checkpoint used by `ml_server.enhancement`
- Optional AASIST clone for full B1 under noise

## Outputs

Everything under `runs/` and `cache/noisy_wavs/` (gitignored locally as needed).
