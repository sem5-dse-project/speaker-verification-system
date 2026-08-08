# Mel / inverted-Mel / LFCC comparison (mixed 2017 + PA2019)

Same CNN, three front-ends. Train on mixed data; report EER on **2017** and **PA** separately.

## Run order

1. `01_lfcc_frontend.ipynb` — LFCC sanity check  
2. `02_train_all_frontends.ipynb` — train Mel, inverted-Mel, LFCC (GPU if available)  
3. `03_compare_eer_table.ipynb` — EER table on 2017 / PA  

## Paths (edit in notebooks if needed)

- ASVspoof2017: `replay-cnn-baseline/data`
- PA2019: `data/PA`
- Checkpoints: `runs/{mel,inverted_mel,lfcc}/`

CUDA + AMP are used automatically when `torch.cuda.is_available()`.
