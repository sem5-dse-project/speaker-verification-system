# Evaluation protocol

## Speaker verification

**Metrics:** EER, FAR, FRR, inference time (ms/utterance)

**Model comparison (offline):**

1. Pre-trained **x-vector** vs **ECAPA-TDNN**
2. Report EER / FAR / FRR at a calibrated threshold (EER operating point or target FAR)
3. Measure average embedding extraction latency on the same hardware

**Trial design:**

- Target trials: same speaker enrollment vs test
- Non-target trials: different speakers
- Protocols written as CSV (`trial_id,enrolled_user_id,test_audio_path,label`)

## Noise and fusion paths

Compare embedding paths under **clean, 20, 10, 5, 0 dB**:

| Path | Description |
|------|-------------|
| Original | ECAPA on noisy/original audio |
| Enhanced | ECAPA on enhanced audio |
| Fixed 50/50 | `α = 0.5` fusion |
| Quality-conditioned | FusionGate predicts `α` |

Fusion equation:

```text
e_fused = α * e_original + (1 - α) * e_enhanced
```

All of `e_original`, `e_enhanced`, and `e_fused` are L2-normalized.

## Replay detection

**Metrics:** EER, precision, recall, F1, confusion matrix

Tune the replay probability threshold on a development split; analyze false accepts (missed replays) and false rejects (bona fide flagged as replay).

## Reproducibility

- Fix `seed` in YAML configs
- Call `voice_auth.common.reproducibility.set_seed`
- Log config path, git commit, and checkpoint hashes in `outputs/`
