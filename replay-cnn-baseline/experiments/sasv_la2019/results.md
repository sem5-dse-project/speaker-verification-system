# SASV experiment results (ASVspoof 2019 LA)

Locked numbers from the notebooks in this folder.  
Protocol: official SASV 2022 trial lists + `get_all_EERs`.  
**Report / compare on eval.** Dev is for development and tuning only.

## Metrics (short)

| Metric | Meaning |
|--------|---------|
| **SASV-EER** | Joint: accept targets, reject other speakers **and** spoofs |
| **SV-EER** | Speaker only: target vs nontarget |
| **SPF-EER** | Spoof only: target vs spoof |

Lower EER is better. Primary number: **SASV-EER**.

## Trial counts

| Split | Target | Nontarget | Spoof | Total |
|-------|--------|-----------|-------|-------|
| Dev | 1,484 | 5,768 | 22,296 | 29,548 |
| Eval | 5,370 | 33,327 | 63,882 | 102,579 |

## Main systems (score-sum fusion)

Fusion forms used:

```text
LFCC / WavLM:  s_sasv = s_asv + (1 - P_spoof)
AASIST:        s_sasv = s_asv + P_bona   (softmax class 1)
```

- `s_asv` = ECAPA cosine (enrol model vs test)
- CM = LFCC, WavLM, or official AASIST (`aasist/` clone)

### Dev

| System | Notebook | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|----------|-------------:|-----------:|------------:|
| ECAPA only | `02` | 15.23 | 1.25 | 17.91 |
| ECAPA + LFCC | `03` | 1.14 | 2.10 | **0.09** |
| ECAPA + WavLM | `05` | 7.35 | 11.86 | 3.84 |
| **ECAPA + AASIST** | `09` | **0.74** | 1.35 | 0.13 |

### Eval (locked — use this for reporting)

| System | Notebook | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|----------|-------------:|-----------:|------------:|
| ECAPA only | `04` | 20.67 | 0.76 | 27.05 |
| ECAPA + LFCC | `04` | 7.13 | 1.56 | 9.71 |
| ECAPA + WavLM | `06` | 12.25 | 14.69 | 6.54 |
| **ECAPA + AASIST** | `10` | **1.14** | **0.82** | **1.39** |

### What this means

- **ECAPA alone** verifies speakers well (low SV-EER) but is open to spoofs (high SPF / SASV).
- **LFCC fusion** is a strong **lightweight** option: SASV-EER ~20.7% → **~7.1%** on eval.
- **WavLM fusion** can help SPF a bit vs LFCC but **hurts SV** (~14.7%), so joint SASV is worse.
- **AASIST fusion** is the best overall: SASV-EER **~1.14%** on eval, with healthy SV and SPF.

**Best system: ECAPA + AASIST score-sum (~1.14% eval SASV-EER).**  
**Best lightweight CM: ECAPA + LFCC (~7.13% eval SASV-EER).**

## Fusion ablations (LFCC scores only)

These reuse saved LFCC `s_asv` / `s_cm` CSVs (no new GPU scoring).  
Tuned on **dev**, locked on **eval**.

### Weighted sum (`07`)

```text
score = α · s_asv + (1 − α) · s_cm
```

Locked **α = 0.65** (min SASV-EER on dev).

| System | Split | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|-------|-------------:|-----------:|------------:|
| Raw sum `s_asv + s_cm` | Dev | 1.14 | 2.10 | 0.09 |
| Weighted α=0.65 | Dev | **0.81** | 1.60 | 0.39 |
| Raw sum | Eval | **7.13** | 1.56 | 9.71 |
| Weighted α=0.65 | Eval | 7.59 | 1.16 | 10.20 |

Weighted fusion helps on **dev**, not on **eval**.

### Calibrate-then-fuse (`08`, B1-v2 style)

Platt / joint logistic fitted on **dev**. Locked method: **`platt_sum_sv_spf`**.

| System | Split | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|-------|-------------:|-----------:|------------:|
| Raw sum | Dev | 1.14 | 2.10 | 0.09 |
| Locked calibrated | Dev | **0.81** | 1.33 | 0.10 |
| Raw sum | Eval | **7.13** | 1.56 | 9.71 |
| Locked calibrated | Eval | 8.14 | 1.10 | 10.98 |

Same story: better on **dev**, worse than raw sum on **eval**.

Other calibrated methods on **eval** (all worse than raw sum):

| Method | Eval SASV-EER (%) |
|--------|------------------:|
| raw_sum | **7.13** |
| logit_sum_sv_spf | 7.36 |
| logit_sum_sasv | 7.50 |
| joint_proba / joint_logit | 7.51 |
| platt_sum_sasv | 7.54 |
| platt_sum_sv_spf (locked) | 8.14 |

## Full eval comparison (all systems)

| System | SASV-EER (%) | SV-EER (%) | SPF-EER (%) | Role |
|--------|-------------:|-----------:|------------:|------|
| ECAPA only | 20.67 | 0.76 | 27.05 | ASV baseline |
| ECAPA + LFCC raw sum | 7.13 | 1.56 | 9.71 | Lightweight CM |
| ECAPA + LFCC weighted (α=0.65) | 7.59 | 1.16 | 10.20 | Ablation |
| ECAPA + LFCC calibrated (locked) | 8.14 | 1.10 | 10.98 | Ablation |
| ECAPA + WavLM raw sum | 12.25 | 14.69 | 6.54 | Ablation |
| **ECAPA + AASIST sum** | **1.14** | **0.82** | **1.39** | **Best / vs B1-v2** |

## vs published SASV 2022 (eval, approximate)

| System | SASV-EER (%) |
|--------|-------------:|
| Official ECAPA alone | ~23.8 |
| **Our ECAPA alone** | **~20.7** |
| Official Baseline2 (DNN fusion) | ~6.5 |
| **Our ECAPA + LFCC sum** | **~7.1** |
| Official Baseline1-v2 (ECAPA + AASIST) | ~1.7 |
| **Our ECAPA + AASIST sum** | **~1.14** |

Our ECAPA alone is slightly better than the published ECAPA-alone number.  
Our ECAPA + AASIST score-sum is in the **B1-v2** ballpark (slightly better here; small gaps from ECAPA checkpoint / score details are expected).  
LFCC remains a lighter CM near **B2** level.

## CM-alone note (LA spoof EER, not SASV)

On ASVspoof 2019 LA **dev** (bona vs spoof only):

| CM | Oracle EER (%) |
|----|---------------:|
| LFCC CNN | ~0.11 |
| WavLM-Base + ASP (frozen) | ~7.63 |
| AASIST (official eval claim) | ~0.83 |

SASV needs the CM **and** ECAPA together; CM-alone EER ≠ SASV-EER.

## Conclusions

1. Adding an LA CM is necessary: ECAPA alone ~21% SASV-EER on eval.
2. **Best joint system: ECAPA + AASIST (~1.14% eval SASV-EER)** — competitive with published B1-v2 (~1.7%).
3. **Best lightweight system: ECAPA + LFCC (~7.13%)** — no AASIST dependency.
4. WavLM under simple score-sum is not better for joint SASV (SV collapses).
5. Extra LFCC fusion (weighted α, Platt calibration) overfits **dev** and does not beat LFCC raw sum on **eval**.
6. Numbers are for **lab ASVspoof LA** flacs, not browser / phone mics.

## Where the numbers live

| Result | Path |
|--------|------|
| ECAPA only | `runs/ecapa_only_{dev,eval}/metrics_*.json` |
| ECAPA + LFCC | `runs/ecapa_plus_lfcc_{dev,eval}/metrics_*.json` |
| ECAPA + WavLM | `runs/ecapa_plus_wavlm_{dev,eval}/metrics_*.json` |
| ECAPA + AASIST | `runs/ecapa_plus_aasist_{dev,eval}/metrics_*.json` |
| Weighted α | `runs/ecapa_plus_lfcc_weighted_eval/locked_alpha.json` |
| Calibrated | `runs/ecapa_plus_lfcc_calibrated/locked_eval.json` |
