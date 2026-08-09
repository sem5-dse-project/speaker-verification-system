# LFCC CNN on ASVspoof 2019 LA

Train an **LA-specific** spoof detector (LFCC + same lightweight CNN).  
Compare to zero-shot mixed inverted-Mel on LA (~**41% EER**).

## Run order

1. `01_train_lfcc_la.ipynb` — train on LA train (speaker-disjoint val)  
2. `02_eval_lfcc_la.ipynb` — score LA **dev** (then optional eval)

## Paths

- Data: `data/LA`
- Checkpoint: `runs/lfcc_la/best_lfcc_la2019.pt`
- Reuses LFCC front-end from `../lfcc_vs_mel_compare`

Set `SMOKE = True` first for a short GPU run.
