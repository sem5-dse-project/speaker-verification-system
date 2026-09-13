# Shared model weights (Google Drive)

Upload **this whole folder** to Google Drive and share the link with the team (Viewer is enough).

These files are **not** in GitHub (`.pt` is gitignored). After clone, teammates copy them into the repo paths below.

## Files in this folder

| File | Required for app? | Put it here after download |
|------|-------------------|----------------------------|
| `best_inverted_mel_mixed_2017_pa2019.pt` | **Yes** (replay gate) | `replay-cnn-baseline/experiments/inverted_mel_mixed_2017_pa2019/runs/inverted_mel_mixed/` |
| `best_lfcc_la2019.pt` | No (LA optional, off by default) | `replay-cnn-baseline/experiments/lfcc_la2019/runs/lfcc_la/` |

Do **not** upload ECAPA weights. The ML server downloads them on first enroll/verify:

`speechbrain/spkrec-ecapa-voxceleb` → `app/server/pretrained_models/`

## After clone (teammate)

1. Clone the GitHub repo.
2. Download this Drive folder.
3. Copy the two `.pt` files into the paths in the table (create the `runs\...` folders if missing).
4. Copy `app/server/.env.example` → `app/server/.env` (defaults already point at the mixed inverted-Mel file).
5. Start FastAPI, Express, frontend as in the app READMEs.

PowerShell example (from the downloaded folder, repo at `D:\speaker-verification-system`):

```powershell
$repo = "D:\speaker-verification-system"
New-Item -ItemType Directory -Force -Path "$repo\replay-cnn-baseline\experiments\inverted_mel_mixed_2017_pa2019\runs\inverted_mel_mixed" | Out-Null
New-Item -ItemType Directory -Force -Path "$repo\replay-cnn-baseline\experiments\lfcc_la2019\runs\lfcc_la" | Out-Null
Copy-Item ".\best_inverted_mel_mixed_2017_pa2019.pt" "$repo\replay-cnn-baseline\experiments\inverted_mel_mixed_2017_pa2019\runs\inverted_mel_mixed\"
Copy-Item ".\best_lfcc_la2019.pt" "$repo\replay-cnn-baseline\experiments\lfcc_la2019\runs\lfcc_la\"
```

## Upload to Drive

1. Open Google Drive.
2. **New → Folder upload** (or drag `share\google-drive-weights`).
3. Right-click the folder → **Share** → **Anyone with the link → Viewer**.
4. Send the link to the team.
