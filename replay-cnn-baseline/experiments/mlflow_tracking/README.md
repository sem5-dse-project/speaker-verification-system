# MLflow experiment log (sidecar)

Logs **existing** replay-detection metrics and ROC / confusion plots into local MLflow.  
Does **not** retrain models and does **not** change FastAPI / Express / the UI.

## Setup

```powershell
pip install mlflow
cd D:\speaker-verification-system\replay-cnn-baseline\experiments\mlflow_tracking
python log_experiments.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 and select experiment `replay-detection-benchmarks`.

Re-running the script adds a new batch of runs (it does not overwrite old ones).  
`mlflow.db` is local and gitignored.

## Screenshot: Mel vs inverted-Mel vs LFCC

Use the three mixed-frontend runs (same CNN, mixed 2017+PA training):

- `mixed_mel_frontend`
- `mixed_imel_frontend`
- `mixed_lfcc_frontend`

Steps:

1. Open experiment `replay-detection-benchmarks`.
2. If you logged more than once, sort by **Start time** and pick the **latest** three runs with those names (or filter tag `compare_group = mixed_frontend`).
3. Tick those three checkboxes.
4. Click **Compare**.
5. Screenshot the metrics table (`eer_2017`, `eer_pa`).
6. Open each run → **Artifacts** for ROC / confusion PNGs (where the original experiment saved them).

Expected EERs: Mel 21.2 / 6.0, inverted-Mel 10.1 / 9.7, LFCC 9.2 / 9.0.
