# Full-dataset inverted-Mel CNN on ASVspoof 2019 PA

Train on **all readable PA train** files and evaluate on **full readable PA dev**
(and optionally PA eval).

New experiment folder — does **not** change the subset scripts under
`inverted_mel_pa2019_train/`.

## Folder

```text
inverted_mel_pa2019_full/
├── train_full_pa2019.py   # full readable train (+ speaker-val or PA-dev val)
├── eval_full_pa2019.py    # full readable eval on dev/eval/train
├── cache/                 # readable-utt caches (local)
├── runs/                  # checkpoints + metrics (local)
└── README.md
```

Shared helpers: `../inverted_mel_pa2019_train/pa_data.py`, `../inverted_mel/`.

## 1) Full train (speaker-disjoint val inside PA train)

Uses ~80% of PA-train speakers for training and ~20% for early-stop / threshold.
**No file caps** — all readable files in each pool.

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel_pa2019_full

python train_full_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --feature-type inverted_mel `
  --output runs\inverted_mel_pa2019_full `
  --epochs 15 `
  --patience 4 `
  --batch-size 8
```

Add `--cpu` if you have no GPU (slow on the full set).

## 2) Full train validated on official PA dev (optional)

Trains on **100%** of readable PA train; early-stop / threshold on **readable PA dev**.

```powershell
python train_full_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --feature-type inverted_mel `
  --validate-on-dev `
  --output runs\inverted_mel_pa2019_full_devval `
  --epochs 15 `
  --patience 4 `
  --batch-size 8
```

> If you validate on PA **dev**, treat PA **eval** as the held-out report split
> (do not tune on the same split you report).

## 3) Evaluate on full PA dev

```powershell
python eval_full_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --checkpoint runs\inverted_mel_pa2019_full\best_inverted_mel_pa2019_full.pt `
  --split dev `
  --output runs\eval_pa2019_dev_full `
  --batch-size 8
```

## 4) Evaluate on full PA eval (if downloaded)

```powershell
python eval_full_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --checkpoint runs\inverted_mel_pa2019_full\best_inverted_mel_pa2019_full.pt `
  --split eval `
  --output runs\eval_pa2019_eval_full `
  --batch-size 8
```

## Outputs

| File | Meaning |
|------|---------|
| `best_inverted_mel_pa2019_full.pt` | Best checkpoint + threshold |
| `best_val_metrics.json` | Val metrics at best epoch |
| `train_history.json` | Per-epoch loss / val EER |
| `pa2019_<split>_full_metrics.json` | Transfer + oracle EER metrics |
| `pa2019_<split>_full_predictions.csv` | Per-utt scores |
| `*_roc.png` / `*_confusion_matrix.png` | Plots |

## Notes

- Corrupt FLACs are skipped and listed under `*_skipped_utts.txt`. First run builds
  readability caches under `cache/`.
- To reuse caches from the subset experiment:

```powershell
# optional one-time
cmd /c mklink /J cache ..\inverted_mel_pa2019_train\cache
```

- Previous subset run (~8k train) got ~11% EER on readable PA dev. Full training
  should use more data; re-report numbers after this run finishes.
