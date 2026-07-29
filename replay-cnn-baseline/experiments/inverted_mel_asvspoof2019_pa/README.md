# Inverted-Mel CNN → ASVspoof 2019 PA (zero-shot transfer)

Test your **ASVspoof 2017-trained** inverted-Mel CNN on **ASVspoof 2019 PA**
without retraining.

## Files

| File | Role |
|------|------|
| `eval_pa2019.py` | Score PA CM protocols with Mel/inverted-Mel checkpoint |
| `runs/` | Metrics + prediction CSVs |

## Data expected

```text
data/PA/
├── ASVspoof2019_PA_cm_protocols/
│   ├── ASVspoof2019.PA.cm.train.trn.txt
│   ├── ASVspoof2019.PA.cm.dev.trl.txt
│   └── ASVspoof2019.PA.cm.eval.trl.txt
├── ASVspoof2019_PA_train/flac/
├── ASVspoof2019_PA_dev/flac/
└── ASVspoof2019_PA_eval/flac/
```

## Run (full PA dev)

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel_asvspoof2019_pa

python eval_pa2019.py `
  --pa-root D:\speaker-verification-system\data\PA `
  --checkpoint ..\inverted_mel\runs\inverted_mel\best_replay_cnn.pt `
  --split dev `
  --output runs\pa2019_dev `
  --batch-size 8
```

## Quick subset

```powershell
python eval_pa2019.py --pa-root D:\speaker-verification-system\data\PA --split dev --max-files 500 --output runs\quick --batch-size 8
```

## Metrics reported

1. **Transfer** — uses the 2017 calibration threshold from the checkpoint  
2. **Oracle EER on PA2019** — best EER threshold on this PA split (for analysis only)

Compare to your 2017 held-out inverted-Mel result (~4.9% EER).

## Corrupt FLAC warning

If many files fail with `flac decoder lost sync`, the PA download is incomplete.
The script **skips corrupt files** and writes `*_skipped_utts.txt`. Re-download
`ASVspoof2019_PA_dev` (and train/eval if needed) for full official results.
