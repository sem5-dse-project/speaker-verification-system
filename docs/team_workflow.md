# Team workflow

## Members and branches

| Member | Focus | Branch |
|--------|-------|--------|
| HERATH H.M.M.M | Speaker verification (encoders, enrollment, scoring, calibration) | `feature/speaker-verification` |
| HERATH H.M.M.P.B | Replay detection (ASVspoof, Log-Mel CNN, threshold tuning) | `feature/replay-detection` |
| HEWARATHNA Y.M. | Quality fusion (enhancement, quality features, FusionGate) | `feature/quality-fusion` |

## Shared rules

1. Work on your feature branch; open PRs into `main` (or `develop` if introduced).
2. Do **not** commit datasets, audio, checkpoints, embeddings, or secrets.
3. Keep interfaces in `voice_auth.common.interfaces` stable; coordinate before changing Protocols.
4. Prefer small PRs with tests for new public functions.
5. Run `pytest` and formatters before requesting review.

## Suggested Git flow

```bash
git checkout main
git pull
git checkout -b feature/speaker-verification   # or your branch

# ... implement and test ...
git add -A
git commit -m "feat(speaker): describe why"
git push -u origin HEAD
# open a pull request for review
```

## Ownership of packages

- Member 1 → `src/voice_auth/speaker_verification/`, `scripts/prepare_speaker_data.py`
- Member 2 → `src/voice_auth/replay_detection/`, `scripts/prepare_replay_data.py`, `scripts/train_replay_detector.py`
- Member 3 → `src/voice_auth/quality_fusion/`, `scripts/precompute_fusion_features.py`, `scripts/train_fusion_gate.py`
- Shared → `common/`, `pipeline/`, `configs/pipeline.yaml`, `scripts/evaluate_system.py`
