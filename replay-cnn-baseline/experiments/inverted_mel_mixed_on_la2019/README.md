# Zero-shot: mixed inverted-Mel (2017+PA) → ASVspoof 2019 LA

Cross-attack test: replay-trained model on **logical access** (TTS/VC) spoof.

## Run

Open `01_eval_mixed_imel_zero_shot_la.ipynb` and run all cells.

- Data: `data/LA`
- Checkpoint: `../inverted_mel_mixed_2017_pa2019/runs/inverted_mel_mixed/best_inverted_mel_mixed_2017_pa2019.pt`
- Default split: **dev** (set `MAX_UTTS` for a quick smoke test)

Outputs under `runs/`.
