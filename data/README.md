# Datasets (local only — do not commit)

This directory holds **local** datasets and derived artifacts.

## Planned datasets

| Dataset | Purpose | Notes |
|---------|---------|-------|
| OpenSLR SLR52 Sinhala speech (FLAC) | Speaker enrollment / verification | Download separately |
| ASVspoof 2017 Version 2 | Replay detection | Download separately |
| MUSAN or RIRS_NOISES subset | Noise augmentation | Keep a **manageable subset** only |
| Consent-based local Sinhala live/replay | Realistic replay / live evaluation | Collect with informed consent |

## Do not commit

- Raw audio (`.flac`, `.wav`, …)
- Generated features / embeddings (`.npy`, `.npz`, …)
- Protocol copies that include absolute machine-specific paths (optional)
- Any personal recordings

Place downloaded corpora under subfolders such as:

```text
data/
  openslr_slr52/
  asvspoof2017/
  noise/
  local_sinhala/
  enrollment_templates/
  fusion_features/
  speaker/
```

See `docs/datasets.md` for licensing and preparation notes.
