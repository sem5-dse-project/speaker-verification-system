# WavLM-Base + ASP on ASVspoof 2019 LA

Lightweight **logical-access** anti-spoofing, following:

> *A lightweight end-to-end anti-spoofing voice model based on WavLM*  
> ICACS 2024 · [doi:10.1145/3708597.3708621](https://doi.org/10.1145/3708597.3708621)

This is **TTS / voice-conversion spoof**, not replay. Do not compare EER to the mixed 2017 + PA Wav2Vec2 experiment.

## Architecture

Frozen [`microsoft/wavlm-base`](https://huggingface.co/microsoft/wavlm-base) → **attentive statistics pooling** → small FC head (bona fide vs spoof).

Paper reports **0.45% EER** on ASVspoof 2019 LA. This run freezes the encoder (their text also stresses a cheap backend). Matching 0.45% is not guaranteed.

## Setup

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline
.\.venv\Scripts\Activate.ps1
pip install transformers
```

Needs `data/LA` (ASVspoof 2019 LA train/dev protocols + flac).

First Hugging Face download of WavLM-Base is ~360MB.

## Notebooks (preferred)

| Notebook | Purpose |
|----------|---------|
| `01_train_wavlm_la.ipynb` | Paper notes, GPU check, knobs, train |
| `02_eval_wavlm_la.ipynb` | Score LA **dev** (optional eval) |

Set `SMOKE = True` first.

## CLI

```powershell
cd experiments\wavlm_la2019
python train_wavlm.py --batch-size 4 --epochs 12
python eval_wavlm.py --split dev
```

Smoke:

```powershell
python train_wavlm.py --epochs 1 --max-train 200 --max-val 100
```

Checkpoint → `runs/wavlm_la/best_wavlm_la2019.pt` (ASP + head only; encoder reloads from HF).

## Compare

Report **LA EER only**. Nearby baselines in this repo:

- `../lfcc_la2019/` — LFCC CNN trained on LA
- `../inverted_mel_mixed_on_la2019/` — replay model zero-shot on LA (poor, as expected)

`--unfreeze-encoder` fine-tunes full WavLM (slow, easy OOM on a 8GB laptop GPU).
