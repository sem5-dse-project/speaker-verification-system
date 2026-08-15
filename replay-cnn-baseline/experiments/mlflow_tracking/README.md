# MLflow experiment log (sidecar)

Logs **existing** replay-detection metrics into local MLflow.  
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
