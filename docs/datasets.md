# Datasets

Datasets, generated features, model checkpoints, and experiment outputs **must not be committed to Git**. Only READMEs and small protocol templates (without audio) may live in the repository.

## 1. OpenSLR SLR52 — Sinhala speech

- **Format:** FLAC
- **Role:** Speaker enrollment and verification experiments on Sinhala speech
- **Preparation (Member 1):** index speakers/utterances, resample/manifest to 16 kHz mono float32, build trial lists

## 2. ASVspoof 2017 Version 2

- **Role:** Replay-attack detection (bona fide vs replay)
- **Preparation (Member 2):** parse official protocols, extract Log-Mel features, train/evaluate the replay CNN
- **Labels:** `0` = bona fide, `1` = replay

## 3. MUSAN or RIRS_NOISES (manageable subset)

- **Role:** Noise augmentation for quality-fusion experiments at 20 / 10 / 5 / 0 dB
- **Preparation (Member 3):** keep a small subset under `data/noise/`; document which files were used

## 4. Local consent-based Sinhala live/replay recordings

- **Role:** Realistic live vs replay evaluation complementary to ASVspoof
- **Ethics:** obtain informed consent; store only under local `data/`; never commit personal audio

## Storage layout (suggested)

```text
data/
  openslr_slr52/          # not in git
  asvspoof2017/           # not in git
  noise/                  # not in git
  local_sinhala/          # not in git
  enrollment_templates/   # not in git (JSON templates may be gitignored too)
  fusion_features/        # not in git
```
