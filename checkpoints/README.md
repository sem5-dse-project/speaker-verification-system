# Model checkpoints (local only — do not commit)

Store trained or downloaded weights here, for example:

```text
checkpoints/
  speaker/
  replay/
  fusion/
  enhancement/
```

Never commit binary weight files (`.pt`, `.pth`, `.ckpt`, `.onnx`, …).
Document which checkpoint corresponds to which config/run in `outputs/` logs instead.
