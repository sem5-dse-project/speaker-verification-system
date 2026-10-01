# Replay detection — locked results table

Numbers below are taken from **existing** `runs/**/*.json` on disk (no new training).  
Primary app checkpoint: **mixed inverted-Mel** `best_inverted_mel_mixed_2017_pa2019.pt`.

**EER** = equal-error rate on that eval set (oracle EER threshold on the same split).  
**F1** = F1 for the replay/spoof class at that oracle threshold, unless noted.

Lower EER is better. Report these tables for the replay chapter; do not mix LA/SASV numbers here.

---

## A) Mixed inverted-Mel (app checkpoint) — per corpus

**Source:** `runs/eval_mixed_both_dev/comparison_summary.json`  
**Checkpoint:** `runs/inverted_mel_mixed/best_inverted_mel_mixed_2017_pa2019.pt`  
**Train-val threshold stored in ckpt:** ≈ **0.155** (also used by the app + ±0.10 band)

| Test set | Files scored | EER (%) | F1 (oracle thr) | F1 (ckpt thr ≈0.155) |
|----------|-------------:|--------:|----------------:|---------------------:|
| ASVspoof **2017** V2 (protocol `dev`) | 1,710 | **9.18** | **0.917** | 0.516\* |
| ASVspoof **2019 PA** (protocol `dev`) | 15,434† | **11.00** | **0.928** | **0.935** |

\*At the mixed train-val threshold, 2017 **recall** drops sharply → low F1. Use **EER** and **F1@oracle** for fair detection quality; ckpt-thr F1 matters for “as deployed with fixed thr.”  
†Same as other PA runs: **14,266** corrupt FLACs skipped.

### Takeaway

One mixed model covers both domains (~9% / ~11% EER). That is the checkpoint shipped in the ML server for the **replay hard gate**.

---

## B) Single-domain inverted-Mel baselines

| Train → test | EER (%) | F1 | Source JSON |
|--------------|--------:|---:|-------------|
| **2017 only** → 2017 `heldout_dev` | **4.90** | **0.955** | `../inverted_mel/runs/inverted_mel_heldout/heldout_dev_metrics.json` |
| **PA only** (full) → PA2019 `dev` | **7.96** | **0.948** | `../inverted_mel_pa2019_full/runs/eval_pa2019_dev_full/pa2019_dev_full_metrics.json` |
| **PA only** (earlier ckpt) → PA2019 `dev` | 11.25 | 0.926 | `../inverted_mel_pa2019_train/runs/eval_pa2019_dev/pa2019_dev_metrics.json` |
| **PA only** → **2017** zero-shot | **39.24** | 0.632 | `../inverted_mel_pa2019_on_asvspoof2017/runs/eval_asvspoof2017_dev/asvspoof2017_dev_metrics.json` |

### Takeaway

- In-domain inverted-Mel is strong (2017 ~4.9%, PA-full ~8%).  
- **Cross-domain** PA→2017 fails (~39% EER) — motivates mixed training.  
- Prefer the **full** PA-trained row (~8%) over the earlier PA-train eval (~11%) when citing a PA-only upper bound.

---

## C) Mixed-train front-end comparison (Mel / inverted-Mel / LFCC)

Same mixed **2017+PA** training recipe; eval on 2017 `dev` (1,710) and PA `dev` (15,434).

**Source:** `../lfcc_vs_mel_compare/runs/eval/comparison_table.json`

| Feature | EER % on 2017 | EER % on PA |
|---------|-------------:|------------:|
| Mel | 21.17 | **5.99** |
| Inverted-Mel | 10.12 | 9.66 |
| **LFCC** | **9.24** | **8.98** |

### Takeaway

LFCC is slightly best on these **lab** tables. The **app still uses inverted-Mel** because mixed/LFCC replay scores ~**1.0** on browser/laptop mics (false REPLAY). Lab EER ≠ mic behaviour.

---

## D) 2017-only Mel vs inverted-Mel (heldout)

**Sources:**  
`../inverted_mel/runs/mel_heldout/heldout_dev_metrics.json`  
`../inverted_mel/runs/inverted_mel_heldout/heldout_dev_metrics.json`

| Feature | Split | EER (%) | F1 |
|---------|-------|--------:|---:|
| Mel | heldout_dev | 10.00 | 0.915 |
| **Inverted-Mel** | heldout_dev | **4.90** | **0.955** |

### Takeaway

On ASVspoof 2017 held-out speakers, inverted-Mel clearly beats Mel (Li et al.–style high-frequency emphasis).

---

## Caveats (put in report once)

1. **Splits differ:** 2017 “heldout_dev” (speaker-disjoint experiment) is not identical to protocol `dev` used in mixed / front-end compare. Do not claim one EER is “strictly better” across those two 2017 numbers without noting the split.  
2. **PA corrupt FLACs:** large skip counts; metrics are on readable files only.  
3. **App threshold:** fixed ckpt thr ≈ 0.155 ± margin → LIVE / UNCERTAIN / REPLAY bands; that is not the same as reporting oracle-EER F1.  
4. **Browser mic:** validate with a small live vs phone-replay set; do not expect lab EER to transfer.

---

## Artifact index

| Role | Path |
|------|------|
| App / mixed ckpt | `runs/inverted_mel_mixed/best_inverted_mel_mixed_2017_pa2019.pt` |
| Mixed eval summary | `runs/eval_mixed_both_dev/comparison_summary.json` |
| Mixed eval 2017 detail | `runs/eval_mixed_both_dev/asvspoof2017_dev_metrics.json` |
| Mixed eval PA detail | `runs/eval_mixed_both_dev/pa2019_dev_metrics.json` |
| Front-end compare | `../lfcc_vs_mel_compare/runs/eval/comparison_table.json` |
| 2017 heldout inverted-Mel | `../inverted_mel/runs/inverted_mel_heldout/heldout_dev_metrics.json` |
| 2017 heldout Mel | `../inverted_mel/runs/mel_heldout/heldout_dev_metrics.json` |
| PA-full in-domain | `../inverted_mel_pa2019_full/runs/eval_pa2019_dev_full/pa2019_dev_full_metrics.json` |
| PA→2017 zero-shot | `../inverted_mel_pa2019_on_asvspoof2017/runs/eval_asvspoof2017_dev/asvspoof2017_dev_metrics.json` |
