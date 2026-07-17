# System architecture

## Problem

Authenticate a claimed user from their voice while remaining robust to background noise and replay attacks.

## High-level design

```text
                    ┌─────────────────────┐
 Enrollment audio → │ Speaker Encoder     │ → L2-norm → average → Template store
                    │ (ECAPA-TDNN / x-vec)│
                    └─────────────────────┘

 Verification audio
        │
        ▼
 ┌──────────────┐   high prob    ┌──────────┐
 │ Replay CNN   │ ─────────────► │ REJECT   │
 └──────────────┘                └──────────┘
        │ low prob
        ▼
 ┌──────────────┐     ┌────────────────┐
 │ ECAPA encode │     │ Frozen enhance │ → ECAPA encode
 │ (original)   │     └────────────────┘
 └──────────────┘              │
        │                      │
        └──────────┬───────────┘
                   ▼
         Quality vector (5-D)
                   │
                   ▼
         Fusion Gate → alpha
                   │
                   ▼
    e_fused = α·e_orig + (1-α)·e_enh   (all L2-normalized)
                   │
                   ▼
         Cosine vs template → threshold → ACCEPT / REJECT
```

## Packages

| Package | Responsibility |
|---------|----------------|
| `voice_auth.common` | Audio utils, Protocols, metrics, seeds, types |
| `voice_auth.speaker_verification` | Encoders, enrollment, trials, scoring, calibration |
| `voice_auth.replay_detection` | Log-Mel features, CNN, train/inference stubs |
| `voice_auth.quality_fusion` | Enhancement, noise, quality features, FusionGate |
| `voice_auth.pipeline` | Enrollment and verification orchestration |

## Shared conventions

- Audio shape `[1, num_samples]`, float32, 16 kHz
- Embedding dim default 192; quality dim default 5
- Replay labels: `0` bona fide, `1` replay
- Configs in YAML; paths via `pathlib.Path`
