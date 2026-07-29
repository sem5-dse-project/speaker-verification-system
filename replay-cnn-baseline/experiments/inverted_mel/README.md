# Inverted-Mel / high-frequency experiment (Li et al., 2017)

This folder is a **new experiment**. It does **not** change `replay_cnn.py`.

## Idea

Standard **Mel** gives more detail to low frequencies.  
**Inverted Mel (I-Mel)** gives more detail to **high frequencies**, which often help replay / device detection (Li et al., Interspeech 2017).

Same CNN as the baseline; only the spectrogram front-end changes.

## Files

| File | Role |
|------|------|
| `features.py` | Log-Mel and inverted Log-Mel front-ends |
| `inverted_mel_cnn.py` | Train / eval / smoke-test CLI |
| `compare_metrics.py` | Side-by-side Mel vs I-Mel metrics |

## Setup

From this folder (or with paths below). Uses the same ASVspoof data as the baseline:

```text
replay-cnn-baseline/data/
```

## Smoke test (no training)

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel
python inverted_mel_cnn.py smoke-test
```

## 1) Train Mel (control)

```powershell
python inverted_mel_cnn.py train `
  --feature-type mel `
  --data-root ..\..\data `
  --output runs\mel `
  --epochs 20 `
  --batch-size 8
```

## 2) Train inverted Mel

```powershell
python inverted_mel_cnn.py train `
  --feature-type inverted_mel `
  --data-root ..\..\data `
  --output runs\inverted_mel `
  --epochs 20 `
  --batch-size 8
```

## 3) Evaluate on held-out development speakers

```powershell
python inverted_mel_cnn.py eval `
  --data-root ..\..\data `
  --checkpoint runs\mel\best_replay_cnn.pt `
  --output runs\mel_heldout

python inverted_mel_cnn.py eval `
  --data-root ..\..\data `
  --checkpoint runs\inverted_mel\best_replay_cnn.pt `
  --output runs\inverted_mel_heldout
```

## 4) Compare EER / F1

```powershell
python compare_metrics.py `
  --mel-metrics runs\mel_heldout\heldout_dev_metrics.json `
  --inverted-mel-metrics runs\inverted_mel_heldout\heldout_dev_metrics.json
```

## Quick test (subset)

```powershell
python inverted_mel_cnn.py train --feature-type inverted_mel --data-root ..\..\data --output runs\quick_imel --epochs 2 --max-train 3000 --max-val 1000 --batch-size 8
```

## Citation note (for logbook)

Li, L., Chen, Y., Wang, D., Zheng, T. F. (2017).  
*A Study on Replay Attack and Anti-Spoofing for Automatic Speaker Verification.* INTERSPEECH.
