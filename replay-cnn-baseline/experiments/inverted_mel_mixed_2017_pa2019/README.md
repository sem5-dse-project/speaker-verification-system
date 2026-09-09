# Mixed ASVspoof 2017 + PA2019 inverted-Mel training

Train one inverted-Mel CNN on **both** corpora to test whether multi-domain
training improves generalization (previous single-domain models did not transfer).

## Folder

```text
inverted_mel_mixed_2017_pa2019/
├── mixed_data.py      # MixedRecord + balanced dataset
├── build_mixed.py     # load 2017 / PA into MixedRecord lists
├── train_mixed.py     # mixed train + speaker-disjoint val
├── eval_mixed.py      # score 2017 and PA separately
├── cache/
├── runs/
└── README.md
```

## 1) Train (full readable mix, corpus-balanced)

```powershell
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\inverted_mel_mixed_2017_pa2019

python train_mixed.py `
  --asv17-root D:\speaker-verification-system\replay-cnn-baseline\data `
  --pa-root D:\speaker-verification-system\data\PA `
  --feature-type inverted_mel `
  --output runs\inverted_mel_mixed `
  --epochs 15 `
  --patience 4 `
  --batch-size 8
```

Corpus balancing is **on by default** (upsamples ASVspoof2017 so PA does not dominate).
Use `--no-balance-corpora` to disable.

Smoke subset:

```powershell
python train_mixed.py --max-train 4000 --max-val 1000 --epochs 2 --cpu
```

## 2) Evaluate on both corpora (report separately)

```powershell
python eval_mixed.py `
  --asv17-root D:\speaker-verification-system\replay-cnn-baseline\data `
  --pa-root D:\speaker-verification-system\data\PA `
  --checkpoint runs\inverted_mel_mixed\best_inverted_mel_mixed_2017_pa2019.pt `
  --corpus both `
  --asv17-split dev `
  --pa-split dev `
  --output runs\eval_mixed_both_dev `
  --batch-size 8
```

Outputs `comparison_summary.json` with side-by-side EER / F1.

## Locked results

See **[results.md](results.md)** for the report table: mixed inverted-Mel on 2017 / PA,
single-domain baselines, and Mel / LFCC front-end comparison (from existing JSON runs).

## How to judge success

Mixed training is a win if **both** corpora are usable with one checkpoint
(vs PA→2017 zero-shot failure), even if each is slightly worse than its specialist.
That is what the locked table in `results.md` shows.

## Notes

- PA corrupt FLACs are skipped (same readability cache pattern).
- Labels: genuine/bonafide = 0, spoof/replay = 1 on both corpora.
- Do not quote a single pooled EER as “generalization”; always report per corpus.
