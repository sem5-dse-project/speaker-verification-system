# Train inverted-Mel CNN from scratch on ASVspoof 2019 PA

Train on **PA train**, validate on speaker-disjoint PA-train speakers, then
evaluate on **PA dev**.

Uses the same inverted-Mel CNN as the 2017 experiment, but **new weights**
trained only on PA2019 (not the 2017 checkpoint).

## Folder

```text
inverted_mel_pa2019_train/
├── pa_data.py          # PA protocols + FLAC filtering/cache
├── train_pa2019.py     # from-scratch training
├── eval_pa2019.py      # evaluate on PA dev/eval
├── cache/              # readable-utt caches (local)
├── runs/               # checkpoints + metrics (local)
└── README.md
```

## 1) Quick smoke train (subset)

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel_pa2019_train

python train_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --feature-type inverted_mel `
  --output runs\quick `
  --epochs 2 `
  --max-train 2000 `
  --max-val 800 `
  --batch-size 8 `
  --cpu
```

## 2) Full train (recommended when GPU available)

```powershell
python train_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --feature-type inverted_mel `
  --output runs\inverted_mel_pa2019 `
  --epochs 15 `
  --patience 4 `
  --batch-size 8
```

First run scans FLAC readability and caches under `cache/`.
Corrupt files are skipped (your PA download has many bad FLACs).

## 3) Evaluate on PA dev

```powershell
python eval_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --checkpoint runs\inverted_mel_pa2019\best_inverted_mel_pa2019.pt `
  --split dev `
  --output runs\eval_pa2019_dev `
  --batch-size 8
```

## Outputs

- `best_inverted_mel_pa2019.pt` — best validation checkpoint + threshold  
- `best_val_metrics.json` — speaker-val metrics during training  
- `pa2019_dev_metrics.json` — PA dev transfer/oracle metrics  

## Latest run (CPU, readable subset)

Trained with `--max-train 8000 --max-val 2000 --epochs 10` on readable PA train
(many FLACs corrupt; ~48% of PA dev skipped).

| Split | Files | EER | F1 (replay) |
|-------|------:|----:|------------:|
| PA train speaker-val (best) | 2,000 | **11.38%** | 0.932 |
| PA dev (readable) | 15,434 | **11.25%** | 0.917 |

Compare: zero-shot 2017→PA2019 was ~50% EER. Training on PA recovers a usable detector.

## Note on corrupt audio

If many files are skipped, re-download PA train/dev for cleaner official numbers.
Training still works on the readable subset.

For a fuller train (all readable train files, no `--max-train`):

```powershell
python train_pa2019.py --feature-type inverted_mel --epochs 15 --patience 4 --batch-size 8
```
