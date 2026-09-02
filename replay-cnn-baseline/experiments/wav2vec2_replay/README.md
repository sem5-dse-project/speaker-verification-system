# Wav2Vec2 replay detection (mixed 2017 + PA2019)

Frozen [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) encoder with a trainable 2-layer MLP head for **LIVE vs REPLAY** classification.

Uses the same mixed data loaders as `../inverted_mel_mixed_2017_pa2019/` (`build_mixed.py`, `mixed_data.py`).

## Setup

From repo root (with CUDA venv active):

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline
.\.venv\Scripts\Activate.ps1
pip install transformers
```

Requires:

- `replay-cnn-baseline/data/` — ASVspoof 2017 train/dev + protocol
- `data/PA/` — ASVspoof 2019 PA

## Train (CLI)

```powershell
cd experiments\wav2vec2_replay
python train_wav2vec2.py --batch-size 8 --epochs 12
```

Smoke test:

```powershell
python train_wav2vec2.py --epochs 1 --max-train 200 --max-val 100
```

Checkpoint → `runs/wav2vec2_mixed/best_wav2vec2_mixed_2017_pa2019.pt`  
(Only the **MLP head** is saved; encoder reloads from Hugging Face.)

## Eval (CLI)

```powershell
python eval_wav2vec2.py --corpus both --asv17-split dev --pa-split dev
```

Outputs per-corpus EER, ROC, confusion matrix under `runs/eval_wav2vec2_both_dev/`.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_train_wav2vec2.ipynb` | GPU check, knobs, train |
| `02_eval_wav2vec2.ipynb` | Score 2017 dev + PA dev, compare table |

## Compare with inverted-Mel

After both models are trained, compare `comparison_summary.json` from:

- `../inverted_mel_mixed_2017_pa2019/runs/eval_mixed_both_dev/`
- `runs/eval_wav2vec2_both_dev/`

Report **2017 EER** and **PA EER** separately (do not pool).

## Options

| Flag | Default | Notes |
|------|---------|-------|
| `--model-id` | `wav2vec2-base-960h` | HF model id |
| `--unfreeze-encoder` | off | Full fine-tune (slow) |
| `--batch-size` | 8 | Lower if OOM |
| `--lr` | 1e-3 | Head-only LR |

## Next step: local phone-replay data

Add admin collection WAVs as extra `MixedRecord` entries and fine-tune the saved head (or re-run train with a local manifest).
