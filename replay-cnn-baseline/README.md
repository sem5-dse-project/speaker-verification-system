# Replay Attack Detection CNN Baseline

This project trains a lightweight Log-Mel CNN to classify speech audio as:

- `0`: genuine / likely live
- `1`: spoof / replay

It uses **ASVspoof 2017 Version 2**, which focuses on physical replay attacks.
This is a research prototype and must not be described as guaranteed liveness
detection.

## 1. Download the dataset

Official dataset page:

https://datashare.ed.ac.uk/handle/10283/3055

Download only these files:

- `ASVspoof2017_V2_train.zip` — approximately 200.7 MB
- `ASVspoof2017_V2_dev.zip` — approximately 133.7 MB
- `protocol_V2.zip` — approximately 103.5 KB

The evaluation archive is not required for the initial project. The total
required download is approximately 335 MB.

Extract all three archives under one `data` directory:

```text
data/
├── ASVspoof2017_V2_train/
│   ├── T_1000001.wav
│   └── ...
├── ASVspoof2017_V2_dev/
│   ├── D_1000001.wav
│   └── ...
└── protocol_V2/
    ├── ASVspoof2017_V2_train.trn.txt
    ├── ASVspoof2017_V2_dev.trl.txt
    └── ASVspoof2017_V2_eval.trl.txt
```

The training archive is divided internally by speaker:

- Training speakers are used to train the CNN.
- Separate validation speakers are used for early stopping and threshold
  selection.
- The official development set is used only for final testing.

Read and follow the dataset's **CC BY-NC 4.0** licence and identify it as
ASVspoof 2017 **Version 2** in reports and publications.

## 2. Project files

Place the project files as follows:

```text
replay-cnn-baseline/
├── replay_cnn.py
├── requirements.txt
└── data/
    ├── ASVspoof2017_V2_train/
    ├── ASVspoof2017_V2_dev/
    └── protocol_V2/
```

If your Python file currently has a different name, either rename it to
`replay_cnn.py` or use its actual filename in the commands.

The dataset can also be stored elsewhere. Pass the directory containing the
three extracted folders through `--data-root`.

## 3. Create the environment

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or WSL:

```bash
source .venv/bin/activate
```

Install PyTorch and Torchaudio builds that match your system, then install the
remaining dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch does not detect your NVIDIA GPU, use the installation command from:

https://pytorch.org/get-started/locally/

`torch` and `torchaudio` must use matching versions.

## 4. Quick pipeline test

First train on a small subset to confirm that the dataset loads correctly.

Linux or WSL:

```bash
python replay_cnn.py train \
  --data-root data \
  --output runs/quick_test \
  --epochs 2 \
  --batch-size 8 \
  --max-train 3000 \
  --max-val 1000
```

Windows PowerShell:

```powershell
python replay_cnn.py train --data-root "data" --output "runs\quick_test" --epochs 2 --batch-size 8 --max-train 3000 --max-val 1000
```

The default validation fraction is `0.2`, meaning approximately 20% of the
official training speakers are reserved for validation. You can change it with
`--validation-fraction`.

## 5. Full training

Linux or WSL:

```bash
python replay_cnn.py train \
  --data-root data \
  --output runs/full_training \
  --epochs 20 \
  --batch-size 8
```

Windows PowerShell:

```powershell
python replay_cnn.py train --data-root "data" --output "runs\full_training" --epochs 20 --batch-size 8
```

For an 8 GB GPU, start with batch size 8. Increase it only if memory permits.

The best model is saved as:

```text
runs/full_training/best_replay_cnn.pt
```

The checkpoint includes the replay threshold selected using the internal,
speaker-disjoint validation subset.

## 6. Final testing on the official development set

Linux or WSL:

```bash
python replay_cnn.py eval \
  --data-root data \
  --checkpoint runs/full_training/best_replay_cnn.pt \
  --output runs/final_test \
  --split dev \
  --batch-size 8
```

Windows PowerShell:

```powershell
python replay_cnn.py eval --data-root "data" --checkpoint "runs\full_training\best_replay_cnn.pt" --output "runs\final_test" --split dev --batch-size 8
```

The output directory contains:

- `dev_metrics.json`
- `dev_predictions.csv`
- `dev_roc.png`
- `dev_confusion_matrix.png`

The JSON reports:

- Equal Error Rate (EER)
- Accuracy
- Replay precision
- Replay recall
- Replay F1-score
- Confusion matrix

Accuracy alone should not be used because the number of genuine and replay
samples may be unbalanced.

If you later download `ASVspoof2017_V2_eval.zip`, extract it beside the other
folders and run the same command with `--split eval`.

## 7. Test a Sinhala recording

Record one live Sinhala sample and replay that recording through another phone.
Record the replayed version through the authentication device's microphone.

Test both recordings separately:

```bash
python replay_cnn.py predict \
  --checkpoint runs/full_training/best_replay_cnn.pt \
  --audio samples/sinhala_live.wav

python replay_cnn.py predict \
  --checkpoint runs/full_training/best_replay_cnn.pt \
  --audio samples/sinhala_replayed.wav
```

Windows PowerShell example:

```powershell
python replay_cnn.py predict --checkpoint "runs\full_training\best_replay_cnn.pt" --audio "samples\sinhala_live.wav"
```

Example output:

```json
{
  "replay_probability": 0.84,
  "threshold": 0.57,
  "decision": "replay/suspicious"
}
```

Do not evaluate the model using only one recording pair. Test several people,
rooms, phones, playback volumes and distances. Keep these Sinhala recordings
as an external test set; do not add them to training and then report results on
the same recordings.

## Important limitations

- ASVspoof 2017 contains a limited set of speakers, devices and environments.
- Performance can decrease with previously unseen microphones and speakers.
- English training data does not guarantee equal performance on Sinhala audio.
- A low replay probability means "likely live according to this model," not
  proven live.
- This model detects physical replay attacks; it is not a complete deepfake or
  text-to-speech detector.
- The final application should combine replay detection with speaker
  verification rather than relying on this CNN alone.
