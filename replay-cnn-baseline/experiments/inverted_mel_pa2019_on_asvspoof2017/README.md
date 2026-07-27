# PA2019 inverted-Mel → ASVspoof 2017 (zero-shot)

Test the **PA2019-trained** inverted-Mel CNN on **ASVspoof 2017** (opposite
direction from the earlier 2017→PA zero-shot).

## Files

| File | Role |
|------|------|
| `eval_asvspoof2017.py` | Score 2017 protocols with PA2019 checkpoint |
| `runs/` | Metrics + prediction CSVs (local) |

## Default checkpoint

```text
../inverted_mel_pa2019_train/runs/inverted_mel_pa2019/best_inverted_mel_pa2019.pt
```

## Data

Uses `replay-cnn-baseline/data` (ASVspoof2017_V2_train / _dev + protocol_V2).

## Full PA→2017 zero-shot (dev)

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel_pa2019_on_asvspoof2017

python eval_asvspoof2017.py `
  --data-root D:\speaker-verification-system\replay-cnn-baseline\data `
  --checkpoint ..\inverted_mel_pa2019_train\runs\inverted_mel_pa2019\best_inverted_mel_pa2019.pt `
  --split dev `
  --output runs\eval_asvspoof2017_dev `
  --batch-size 8 `
  --cpu
```

## Quick subset

```powershell
python eval_asvspoof2017.py `
  --split dev `
  --max-files 500 `
  --output runs\quick `
  --batch-size 8 `
  --cpu
```

## Metrics

1. **Transfer** — PA train-val threshold from the checkpoint  
2. **Oracle EER** — best EER threshold on this 2017 split (analysis only)

Compare to:
- 2017-trained inverted-Mel on 2017 held-out (~4.9% EER)
- 2017→PA2019 zero-shot (~50% EER)
- PA-trained on PA2019 (~11% EER)
