# Official-eval scoring

Rescores saved replay checkpoints on the official evaluation lists. Nothing is retrained, and checkpoint thresholds are not refit.

Replace `data/PA` with a fresh ASVspoof 2019 PA download before notebooks 00, 02, and 03. Notebooks 01 and 04 only need data that is already on disk.

| Notebook | What it scores |
|---|---|
| `00_check_pa_replacement.ipynb` | Decodes a known-bad PA file and 40 eval probes, then drops stale skip caches |
| `01_2017_only_on_eval.ipynb` | Log-Mel and inverted-Mel trained on 2017, scored on 2017 eval |
| `02_pa_specialist_on_eval.ipynb` | Full PA inverted-Mel on PA eval and on 2017 eval |
| `03_mixed_frontends_on_eval.ipynb` | Mixed log-Mel, inverted-Mel, and LFCC on 2017 eval and PA eval |
| `04_mixed_imel_on_la_eval.ipynb` | Mixed inverted-Mel on LA eval |
| `06_aasist_official_eval.ipynb` | Published AASIST weights, no training, on 2017 eval and PA eval |
| `07_mixed_imel_seeds.ipynb` | Two more mixed inverted-Mel seeds, then 2017 eval and PA eval |
| `08_mixed_imel_ratios.ipynb` | Inverted-Mel at 100%, 75%, 25%, and 0% PA, then official eval and PA t-DCF |
| `09_pa_only_logmel.ipynb` | Train PA-only log-Mel, then score 2017 eval, PA eval, and PA t-DCF |
| `05_paper_table.ipynb` | One table of equal error rate and protocol coverage |

PA eval is about 135,000 files, so notebooks 02 and 03 take a long time. Results are written under `runs/`.
