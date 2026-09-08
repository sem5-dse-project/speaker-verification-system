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
Weighted:      s_sasv = α · s_asv + (1 − α) · s_cm
```

- `s_asv` = ECAPA cosine (enrol model vs test)
- CM = LFCC, WavLM, or official AASIST (`aasist/` clone)

### Dev

| System | Notebook | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|----------|-------------:|-----------:|------------:|
| ECAPA only | `02` | 15.23 | 1.25 | 17.91 |
| ECAPA + LFCC | `03` | 1.14 | 2.10 | **0.09** |
| ECAPA + WavLM | `05` | 7.35 | 11.86 | 3.84 |
| ECAPA + AASIST sum | `09` | 0.74 | 1.35 | 0.13 |
| ECAPA + AASIST weighted (α=0.30) | `11` | **0.74** | 1.35 | **0.07** |

### Eval (locked — use this for reporting)

| System | Notebook | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|----------|-------------:|-----------:|------------:|
| ECAPA only | `04` | 20.67 | 0.76 | 27.05 |
| ECAPA + LFCC | `04` | 7.13 | 1.56 | 9.71 |
| ECAPA + WavLM | `06` | 12.25 | 14.69 | 6.54 |
| ECAPA + AASIST sum | `10` | 1.14 | 0.82 | 1.39 |
| **ECAPA + AASIST weighted (α=0.30)** | `11` | **0.83** | 0.87 | **0.79** |

### What this means

- **ECAPA alone** verifies speakers well (low SV-EER) but is open to spoofs (high SPF / SASV).
- **LFCC fusion** is a strong **lightweight** option: SASV-EER ~20.7% → **~7.1%** on eval.
- **WavLM fusion** can help SPF a bit vs LFCC but **hurts SV** (~14.7%), so joint SASV is worse.
- **AASIST raw sum** reaches ~**1.14%** SASV-EER on eval (beats published B1-v2 ~1.7%).
- **AASIST weighted (α=0.30)** is best overall: ~**0.83%** SASV-EER on eval.

**Best system: ECAPA + AASIST weighted α=0.30 (~0.83% eval SASV-EER).**  
**Best lightweight CM: ECAPA + LFCC (~7.13% eval SASV-EER).**

## AASIST post-fusion ablations (`11` / `12`)

Reuse saved AASIST / LFCC score CSVs (no GPU). Tuned on **dev**, locked on **eval**.

### Weighted AASIST (`11`)

Locked **α = 0.30** (min SASV-EER on dev).

| System | Split | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|-------|-------------:|-----------:|------------:|
| AASIST raw sum | Dev | 0.74 | 1.35 | 0.13 |
| Weighted α=0.30 | Dev | 0.74 | 1.35 | 0.07 |
| AASIST raw sum | Eval | 1.14 | 0.82 | 1.39 |
| **Weighted α=0.30** | Eval | **0.83** | 0.87 | **0.79** |

Unlike LFCC weighting, AASIST **α helps on eval**.

### AASIST + LFCC CM ensemble (`12`)

```text
s_cm = β · s_cm_aasist + (1 − β) · s_cm_lfcc
s_sasv = s_asv + s_cm
```

Locked **β = 0.70** (sum mode).

| System | Split | SASV-EER (%) | SV-EER (%) | SPF-EER (%) |
|--------|-------|-------------:|-----------:|------------:|
| AASIST raw sum | Eval | **1.14** | 0.82 | 1.39 |
| Ensemble β=0.70 | Eval | 1.15 | 0.97 | 1.30 |
| LFCC raw sum | Eval | 7.13 | 1.56 | 9.71 |

Ensemble ≈ AASIST alone on eval — **no gain**.

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
| ECAPA + AASIST sum | 1.14 | 0.82 | 1.39 | vs B1-v2 |
| AASIST+LFCC ensemble (β=0.70) | 1.15 | 0.97 | 1.30 | Ablation |
| **ECAPA + AASIST weighted (α=0.30)** | **0.83** | 0.87 | **0.79** | **Best** |

## vs published SASV 2022 (eval, approximate)

| System | SASV-EER (%) |
|--------|-------------:|
| Official ECAPA alone | ~23.8 |
| **Our ECAPA alone** | **~20.7** |
| Official Baseline2 (DNN fusion) | ~6.5 |
| **Our ECAPA + LFCC sum** | **~7.1** |
| Official Baseline1-v2 (ECAPA + AASIST) | ~1.7 |
| **Our ECAPA + AASIST sum** | **~1.14** |
| **Our ECAPA + AASIST weighted (α=0.30)** | **~0.83** |

Our ECAPA alone is slightly better than the published ECAPA-alone number.  
AASIST score-sum and weighted fusion both beat **B1-v2 (~1.7%)** on this setup.  
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
2. **Best joint system: ECAPA + AASIST weighted α=0.30 (~0.83% eval SASV-EER)** — better than B1-v2 (~1.7%).
3. AASIST raw sum (~1.14%) already beats B1-v2; weighting improves further on eval.
4. **Best lightweight system: ECAPA + LFCC (~7.13%)** — no AASIST dependency.
5. WavLM under simple score-sum is not better for joint SASV (SV collapses).
6. LFCC α / Platt calibration and AASIST+LFCC ensemble do **not** beat their simpler baselines on eval.
7. Numbers are for **lab ASVspoof LA** flacs, not browser / phone mics.

## Where the numbers live

| Result | Path |
|--------|------|
| ECAPA only | `runs/ecapa_only_{dev,eval}/metrics_*.json` |
| ECAPA + LFCC | `runs/ecapa_plus_lfcc_{dev,eval}/metrics_*.json` |
| ECAPA + WavLM | `runs/ecapa_plus_wavlm_{dev,eval}/metrics_*.json` |
| ECAPA + AASIST | `runs/ecapa_plus_aasist_{dev,eval}/metrics_*.json` |
| AASIST weighted | `runs/ecapa_plus_aasist_weighted/locked_eval.json` |
| AASIST+LFCC ensemble | `runs/ecapa_plus_aasist_lfcc_ens/locked_eval.json` |
| LFCC weighted α | `runs/ecapa_plus_lfcc_weighted_eval/locked_alpha.json` |
| LFCC calibrated | `runs/ecapa_plus_lfcc_calibrated/locked_eval.json` |
