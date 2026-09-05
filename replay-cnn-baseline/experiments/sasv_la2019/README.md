# SASV-style eval on ASVspoof 2019 LA

Joint **speaker verification + anti-spoof** metrics using the official SASV 2022 trial lists and `get_all_EERs`.

## What you need

- `data/LA` (ASVspoof 2019 LA audio + `ASVspoof2019_LA_asv_protocols`)
- Local clone: `SASVC2022_Baseline/` at repo root  
  (`git clone https://github.com/sasv-challenge/SASVC2022_Baseline.git`)
- `app/server` venv with `speechbrain`, `scikit-learn`, `scipy`  
  (and `transformers` for WavLM notebooks `05` / `06`)

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_metrics.ipynb` | Paths, protocol, metric smoke test |
| `02_ecapa_only_sasv.ipynb` | ECAPA cosine only → SASV/SV/SPF EER |
| `03_ecapa_plus_cm_sasv.ipynb` | ECAPA + LFCC (or WavLM) score-sum on **dev** |
| `04_eval_locked.ipynb` | Locked **eval** for ECAPA / ECAPA+LFCC |
| `05_ecapa_plus_wavlm_dev.ipynb` | ECAPA + WavLM score-sum on **dev** |
| `06_ecapa_plus_wavlm_eval.ipynb` | Locked **eval** for ECAPA + WavLM |

Always start with `SMOKE = True` (500 trials) on **dev**.

## Metrics

| Metric | Trials |
|--------|--------|
| SV-EER | target vs nontarget |
| SPF-EER | target vs spoof |
| SASV-EER | target vs nontarget+spoof |

Tune on **dev** (`02` / `03` / `05`); report **eval** once via `04` (LFCC) and/or `06` (WavLM) — no further tuning.

## Outputs

Under `runs/`:

- `ecapa_only_dev/metrics_dev.json`
- `ecapa_plus_lfcc_dev/metrics_dev.json`
- `ecapa_plus_wavlm_dev/metrics_dev.json`
- `ecapa_only_eval/metrics_eval.json`
- `ecapa_plus_lfcc_eval/metrics_eval.json`
- `ecapa_plus_wavlm_eval/metrics_eval.json`

## Note

`SASVC2022_Baseline/` is gitignored — do not commit the upstream clone.
