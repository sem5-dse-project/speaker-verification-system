# SASV-style eval on ASVspoof 2019 LA

Joint **speaker verification + anti-spoof** metrics using the official SASV 2022 trial lists and `get_all_EERs`.

## What you need

- `data/LA` (ASVspoof 2019 LA audio + `ASVspoof2019_LA_asv_protocols`)
- Local clone: `SASVC2022_Baseline/` at repo root  
  (`git clone https://github.com/sasv-challenge/SASVC2022_Baseline.git`)
- For AASIST notebooks: local `aasist/` clone + `models/weights/AASIST.pth`  
  (`git clone https://github.com/clovaai/aasist.git`)
- `app/server` venv with `speechbrain`, `scikit-learn`, `scipy`  
  (and `transformers` for WavLM notebooks `05` / `06`)

## Best locked **eval** result

| System | SASV-EER | SV-EER | SPF-EER |
|--------|---------:|-------:|--------:|
| **ECAPA + AASIST weighted (α=0.30)** | **0.83%** | 0.87% | 0.79% |
| ECAPA + AASIST raw sum | 1.14% | 0.82% | 1.39% |
| ECAPA + LFCC raw sum | 7.13% | 1.56% | 9.71% |
| ECAPA only | 20.67% | 0.76% | 27.05% |

Published B1-v2 (ECAPA+AASIST) ≈ **1.71%** SASV-EER. Full tables: [`results.md`](results.md).

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_metrics.ipynb` | Paths, protocol, metric smoke test |
| `02_ecapa_only_sasv.ipynb` | ECAPA cosine only → SASV/SV/SPF EER |
| `03_ecapa_plus_cm_sasv.ipynb` | ECAPA + LFCC (or WavLM) score-sum on **dev** |
| `04_eval_locked.ipynb` | Locked **eval** for ECAPA / ECAPA+LFCC |
| `05_ecapa_plus_wavlm_dev.ipynb` | ECAPA + WavLM score-sum on **dev** |
| `06_ecapa_plus_wavlm_eval.ipynb` | Locked **eval** for ECAPA + WavLM |
| `07_lfcc_weighted_fusion.ipynb` | Tune `α` for LFCC scores on **dev**, lock on **eval** |
| `08_lfcc_calibrated_fusion.ipynb` | Platt / joint logistic calibrate-then-fuse |
| `09_ecapa_plus_aasist_sasv.ipynb` | ECAPA + AASIST on **dev** (smoke → full) |
| `10_ecapa_plus_aasist_eval.ipynb` | Locked **eval** for ECAPA + AASIST |
| `11_aasist_weighted_fusion.ipynb` | Tune `α` on AASIST scores → locked eval (**best**) |
| `12_aasist_lfcc_ensemble.ipynb` | Ensemble AASIST+LFCC CM (`β`); lock on eval |

Always start with `SMOKE = True` (500 trials) on **dev** for scoring notebooks.

## Metrics

| Metric | Trials |
|--------|--------|
| SV-EER | target vs nontarget |
| SPF-EER | target vs spoof |
| SASV-EER | target vs nontarget+spoof |

Tune on **dev**; report **eval** once (`04` / `06` / `10` / locked `07`–`08` / `11`–`12`) — no further tuning.

## Outputs

Under `runs/`:

- `ecapa_only_{dev,eval}/`
- `ecapa_plus_lfcc_{dev,eval}/`
- `ecapa_plus_wavlm_{dev,eval}/`
- `ecapa_plus_aasist_{dev,eval}/` (notebooks `09` / `10`)
- `ecapa_plus_lfcc_weighted_*` / `ecapa_plus_lfcc_calibrated/` (`07` / `08`)
- `ecapa_plus_aasist_weighted/` (`11`) — includes `locked_eval.json`
- `ecapa_plus_aasist_lfcc_ens/` (`12`) — includes `locked_eval.json`

## Note

`SASVC2022_Baseline/` and `aasist/` are local clones — do not commit them.
