# Noise-aware gated SASV (publishable track)

Offline experiments: **ASVspoof 2019 LA + SASV trials**, with **test-side noise** and
**SNR-gated enhancement fusion** (Wave-U-Net + ECAPA).

> Not the Docker demo. Tune on **dev**, report **eval** once.

## Notebooks

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_protocol_and_harness.ipynb` | Freeze protocol; smoke offline ECAPA scoring |
| 2 | `02_noise_injection.ipynb` | MUSAN/white noise at fixed SNRs (test only) |
| 3 | `03_baselines_and_gated_systems.ipynb` | B0 / P2 / P3 under a chosen SNR |
| 4 | `04_matrix_tables_and_claim.ipynb` | Tables, plots, claim checklist |
| 5 | `05_offline_score_fusion.ipynb` | Step A: offline B0/P3 score fusion (no re-score) |
| 6 | `06_enhancer_gate_ablation.ipynb` | Step B: Wave-U-Net gate ablation smoke (SI-SDR + EER) |
| 7 | `07_ecapa_cm_under_noise.ipynb` | ECAPA + AASIST (α=0.30) / optional LFCC under SNR grid |
| 8 | `08_cm_noise_matrix_and_claim.ipynb` | CM-under-noise tables, plot, claim checklist |

Shared code: `noise_gated_lib.py` (imports helpers from `../sasv_la2019`).

## Systems

| ID | Meaning |
|----|---------|
| B0 | ECAPA only |
| B1 | ECAPA + AASIST weighted α=0.30 (from `sasv_la2019`, clean locked ~0.83%) |
| B2 | ECAPA + LFCC score-sum (optional) |
| P2 | Gated raw/enhanced ECAPA (+ CM in full paper runs) |
| P3 | Always-enhance ECAPA |

## Track notes

- **Enhancement track (01–06):** Wave-U-Net gating did **not** beat raw ECAPA (negative result).
- **CM-under-noise track (07–08):** evaluate whether AASIST/LFCC holds under MUSAN noise.

## Prerequisites

- `data/LA`, `SASVC2022_Baseline/`, `app/server` (SpeechBrain, etc.)
- **AASIST** clone + weights at repo `aasist/` (required for notebook 07)
- Optional: MUSAN noise root; Wave-U-Net checkpoint (notebooks 03/06)
- Optional: app LFCC-LA checkpoint for **B2** in notebook 07 (`RUN_LFCC=True`)

## Outputs

Everything under `runs/` and `cache/noisy_wavs/` (gitignored locally as needed).


## Run-All defaults (current)

- Notebooks **01–04**: default **`SMOKE = False`** (full dev) when left as configured
- Notebook **07**: default **`SMOKE = True`** (1500 trials) — set `False` for full grid
- Wave-U-Net: `app/server/checkpoints/waveunet_finetuned_v4_best.pt`
- Noise: MUSAN under common paths; else white-noise fallback
- SNR grid: clean, 15, 10, 5, 0 dB

Override MUSAN with `NOISE_ROOT` in the setup cell, or env var `NOISE_ROOT`.
