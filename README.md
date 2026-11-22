# Adolescent-Level and Parent-Level Correlates of Oppositional Defiant Disorder in Korean Adolescents

**Machine-learning and multivariable analysis of parent–adolescent factors using national survey data (NMHSK-CA 2022)**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Reporting: TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-8a2be2.svg)](https://www.tripod-statement.org/)

Official code for the paper:

> Ryemi Do†, Daehyeon Choe†, Seungbin Oh, Soyeon Kim, Hyoun-Joong Kong\*, Hewoen Sim,
> Sumin Hwang, Sohyun Park, Chae Rim Lee, Bung-Nyun Kim\*.
> **Adolescent-Level and Parent-Level Correlates of Oppositional Defiant Disorder in Korean
> Adolescents: A Machine Learning Analysis of National Survey Data.** *(Manuscript in preparation / under review).*
>
> † Co-first authors · \* Co-corresponding authors

This repository contains the complete analysis code (4-stage nested cross-validation, locked
per-target evaluation, post-hoc SHAP and permutation importance, covariate-adjusted logistic
regression with Benjamini–Hochberg FDR, and calibration/DeLong sensitivity checks) and **one
walkthrough notebook** that runs end to end on synthetic stand-in data or on the restricted
survey data.

---

## Contents

- [Highlights](#highlights)
- [Study design](#study-design)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start: run the walkthrough without data](#quick-start-run-the-walkthrough-without-data)
- [Running the full pipeline](#running-the-full-pipeline)
- [Data availability](#data-availability)
- [Methods in brief](#methods-in-brief)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license) · [Contact](#contact)

## Highlights

- **Data**: 3,382 adolescents aged 12–17 and their caregivers from the nationally
  representative NMHSK-CA 2022 survey; **177 (5.2%)** with current-or-past Oppositional
  Defiant Disorder (ODD) and **197 (5.8%)** with the broader Disruptive, Impulse-Control, and
  Conduct Disorder (DICCD) outcome, assessed via adolescent self-administered KSADS-COMP.
  Parent-rated disruptive and ADHD symptom scales (`K-DBDRS`) are excluded from all predictors.
- **Design**: nested 5×5 stratified cross-validation run **independently** for ODD and DICCD
  across a sequential 4-stage search (Classifier → Train-only Resampler → Loss/Weighting →
  Hyperparameter Grid + 40-trial Optuna TPE check). Candidates at every stage are selected by
  multi-metric rank consistency across inner out-of-fold AUC, F1, and Average Precision rather
  than raw AUC alone, converging on distinct locked-in recipes per outcome:

  | Component | ODD | DICCD |
  |---|---|---|
  | Classifier | Random Forest (`n_estimators=500`) | Random Forest (`n_estimators=500`) |
  | Train-only resampler | ADASYN (1:1) | None (no resampling) |
  | Loss / weighting | Focal ($\gamma=2.0$, class-balanced $\alpha$) | Unweighted BCE |
  | Hyperparameters | `gini`, `min_samples_leaf=1`, `max_depth=None`, `max_features=sqrt` | `entropy`, `min_samples_leaf=1`, `max_depth=12`, `max_features=sqrt` |

- **Discrimination** (outer-test AUC, mean ± SD over 5 folds):

  | Feature set | Encoded features | ODD (`n_cases = 177`) | DICCD (`n_cases = 197`) |
  |---|:---:|:---:|:---:|
  | Combined | 63 | **0.871 ± 0.010** | **0.871 ± 0.041** |
  | Adolescent | 20 | 0.853 ± 0.015 | 0.848 ± 0.066 |
  | Parent | 43 | 0.749 ± 0.034 | 0.746 ± 0.044 |

  Adolescent-reported predictors carry most of the discrimination, while adding parent-level
  variables further improves outer-test AUC over Parent-only ($p < 0.001$ for both outcomes)
  and Adolescent-only (DICCD paired DeLong $p = 0.004$; ODD $\Delta\text{AUC} = +0.018$), with
  well-calibrated inner-fold Platt probabilities (ODD Brier = 0.0398).
- **Interpretation & adjusted associations**: on outer-test folds, all six newly added
  adolescent digital, peer, and academic variables—smartphone addiction, cyberbullying
  victimization and perpetration, school bullying perpetration, academic decline, and academic
  performance—rank within the Combined SHAP top 10 for both outcomes and remain significant
  after Benjamini–Hochberg FDR adjustment ($q < 0.001$), led for ODD by smartphone addiction
  (aOR 2.44, 95% CI 2.05–2.90 per 1 SD) and cyberbullying victimization (aOR 6.43, 95% CI
  4.65–8.88) alongside parental anxiety (aOR 5.30, 95% CI 3.08–9.12) and maternal alcohol use
  (aOR 2.51, 95% CI 1.83–3.44).

## Study design

```
NMHSK-CA 2022  (n = 3,382 adolescent–caregiver pairs aged 12–17; K-DBDRS/ADHD items excluded)
   │
   ├── outcome (analysed independently) .. ODD (177 cases, 5.2%) | DICCD (197 cases, 5.8%)
   ├── feature set ....................... Parent (15 vars → 43) | Adolescent (17 → 20) | Combined (32 → 63)
   ▼
Nested 5×5 stratified CV ─────────────────────────────────────────────────────────────
   4-stage sequential search per outcome (inner 5-fold OOF rank-consistency: AUC + F1 + AP)
     Stage 1: 8 classifiers (no resampling) ────────► Random Forest
     Stage 2: 8 train-only resamplers ──────────────► ODD: ADASYN (1:1) | DICCD: None
     Stage 3: BCE / Weighted BCE / Focal (γ = 2.0) ─► ODD: Focal        | DICCD: BCE
     Stage 4: 24-config RF grid + 40-trial Optuna ──► ODD: gini/depth=None | DICCD: entropy/depth=12
   for each outer fold k = 1..5
     outer-train ─► fit imputation / z-scoring / one-hot   (never on outer-test)
                 ─► inner 5-fold OOF scores ─► Youden cut-off + Platt / isotonic calibrators
                 ─► refit locked recipe on outer-train ─► score untouched outer-test fold once
   report mean ± SD over the 5 outer-test folds + paired DeLong & bootstrap feature-set tests
   ▼
Post-hoc & epidemiological inference (outer-test folds / multivariable logit)
   TreeSHAP · permutation importance (ΔAUC, 8 repeats) · adjusted ORs (sex, age, PHQ-9, GAD-7; BH-FDR)
```

## Repository structure

```
korean-adolescent-odd-diccd-ml/
├── src/odd_diccd_ml/                analysis package (CLI: python -m odd_diccd_ml)
│   ├── config.py                    loads configs/experiment.yaml and configs/features.yaml
│   ├── data.py                      cohort loading, schema validation, no-ADHD guard
│   ├── preprocess.py                train-only imputation / z-scoring / parental-ordinal / one-hot
│   ├── models.py                    8 classifiers, 8 resamplers, BCE/Weighted BCE/Focal, metrics
│   ├── nested_cv.py                 4-stage nested 5×5 CV, Optuna TPE check, locked-recipe evaluation
│   ├── importance.py                outer-test permutation importance (ΔAUC) and TreeSHAP
│   ├── association.py               crude & adjusted logistic ORs (BH-FDR) and Spearman correlations
│   ├── robustness.py                Platt/isotonic calibration, paired DeLong tests, no-perpetration check
│   ├── plots.py                     ROC/PR ribbons, confusion matrices, importance bars, OR forest plot
│   ├── synthetic.py                 schema-identical synthetic data generator (demo/testing only)
│   └── __main__.py                  command-line interface
├── notebooks/
│   └── odd_diccd_ml_walkthrough.ipynb   ★ one notebook: synthetic end-to-end demo or full pipeline
├── configs/
│   ├── experiment.yaml              seed, CV design, Stage 1–4 grids, locked per-target recipes
│   └── features.yaml                32 raw survey variables (63 encoded features), types, labels
├── data/
│   └── README.md                    data access + exact expected file layout (no data shipped)
├── requirements.txt                 pinned versions used to verify reproduction
├── CITATION.cff
└── LICENSE                          Apache 2.0
```

## Installation

```bash
git clone https://github.com/dablro12/korean-adolescent-odd-diccd-ml.git
cd korean-adolescent-odd-diccd-ml
conda create -n odd-diccd-ml python=3.10 -y && conda activate odd-diccd-ml
pip install -r requirements.txt
```

## Quick start: run the walkthrough without data

```bash
jupyter lab notebooks/odd_diccd_ml_walkthrough.ipynb
```

The notebook runs in one of two modes, set in its first code cell:

| `MODE` | What it does | Data needed | Runtime |
|---|---|---|---|
| `"synthetic"` *(default)* | Generates schema-identical **synthetic** ODD and DICCD data and runs the pipeline (locked nested CV, outer-test SHAP & permutation importance, covariate-adjusted ORs, and diagnostic figures) | none | < 1 min |
| `"full"` | Runs the complete published pipeline end to end on the restricted NMHSK-CA 2022 extracts in `DATA_DIR` | study CSVs | ~35–40 min |

*(Numbers produced under `MODE = "synthetic"` come from random draws and are purely
illustrative — they do not represent the study's real findings.)*

## Running the full pipeline

With the restricted survey extracts placed as described in [`data/README.md`](data/README.md),
run from `src/`:

```bash
cd src
python -m odd_diccd_ml all                                    # locked-cv + importance + association + robustness (ODD & DICCD)

# or step by step per target (odd | diccd)
python -m odd_diccd_ml locked-cv    --target odd              # 3 feature sets × 5 outer folds × 5 inner folds
python -m odd_diccd_ml importance   --target odd              # outer-test permutation importance (ΔAUC) + TreeSHAP
python -m odd_diccd_ml association  --target odd              # crude & adjusted ORs (BH-FDR) + Spearman rho
python -m odd_diccd_ml robustness   --target odd              # Platt/isotonic calibration, DeLong, no-perpetration check

# optional: re-run the full 4-stage sequential candidate search + 40-trial Optuna TPE check
python -m odd_diccd_ml stage-search --target odd   --stages 1 2 3 4 --optuna
python -m odd_diccd_ml stage-search --target diccd --stages 1 2 3 4 --optuna
```

Useful options (place `--data-dir` and `--results-dir` before the sub-command):
`--feature-sets combined adolescent parent`, `--n-estimators 100` (for quick smoke checks),
and `python -m odd_diccd_ml synthetic --out-dir ../data/synthetic` to materialize synthetic
stand-in files on disk. All outputs are written to `results/`, which is git-ignored.

## Data availability

**No participant-level data or study result tables are bundled in this repository.** The
NMHSK-CA 2022 microdata are restricted human-subjects data and must be requested from the data
custodian (National Center for Mental Health, Republic of Korea). [`data/README.md`](data/README.md)
describes the access procedure, the expected CSV/Excel files, all 32 predictor variables and
their coding. This secondary analysis was exempted from review by the Seoul National University
Hospital IRB (E-2412-038-1594).

## Methods in brief

| Component | Setting |
|---|---|
| Outcomes | primary: current-or-past ODD (`ODD_CP`, 177/3,382 = 5.2%); secondary: current-or-past DICCD (`DICCD_CP`, 197/3,382 = 5.8%), adolescent self-administered KSADS-COMP (DSM-5) |
| Predictor scope | 32 survey variables (63 encoded features: 20 Adolescent + 43 Parent); `sex`, diagnostic items, and all parent-rated `K-DBDRS` / ADHD symptom scales excluded from ML predictors |
| Preprocessing | median imputation + z-scoring (continuous), mode imputation (binary and adolescent ordinals), median imputation (parental ordinals `F4_M/F`, `FA_M/F`, `FT_M/F`), mode imputation + one-hot encoding (nominal); fit on the training partition only at every CV level |
| Outer / Inner CV | 5-fold × 5-fold `StratifiedKFold(shuffle=True, random_state=42)` |
| Stage 1–4 search | Stage 1: 8 classifiers (LR, NB, Linear SVM, RF, GB, HistGB, XGBoost, CatBoost) → Stage 2: 8 train-only 1:1 resamplers → Stage 3: BCE, Weighted BCE, Focal ($\gamma=2.0$) → Stage 4: 24-config RF grid + 40-trial Optuna TPE check |
| Selection rule | lowest rank sum across inner OOF AUC, F1, and Average Precision (avoiding raw-AUC-only picks that collapse precision/F1); decision threshold = Youden-optimal on inner OOF scores |
| Locked recipes | **ODD**: RF (500) + ADASYN (1:1) + Focal ($\gamma=2.0$) + `gini, leaf=1, depth=None, sqrt` · **DICCD**: RF (500) + None + BCE + `entropy, leaf=1, depth=12, sqrt` |
| Calibration & DeLong | Platt logistic and isotonic recalibration fit strictly on inner OOF scores and evaluated on outer-test folds; paired Sun–Xu DeLong test + 2,000 class-stratified bootstrap resamples comparing Combined vs. Adolescent/Parent |
| Explainability | TreeSHAP (`mean \|SHAP\|`) and permutation importance ($\Delta\text{AUC}$, `n_repeats=8`, `seed=42`) computed on untouched outer-test folds only |
| Adjusted associations | multivariable logistic regression per unique Top-10 SHAP exposure (21 for ODD, 20 for DICCD), adjusted for adolescent sex, age, parental PHQ-9 $\ge 9$, and parental GAD-7 $\ge 5$ (dropping a covariate when it is the tested exposure); Benjamini–Hochberg FDR applied once per outcome |

Reporting follows the TRIPOD+AI statement.

## Reproducibility

- All randomness is deterministically seeded (`seed = 42` for outer/inner CV splits and
  permutation importance; `42 + k` for fold-specific resamplers and tree fits).
- Column order in [`configs/features.yaml`](configs/features.yaml) fixes the exact 5-block
  `ColumnTransformer` feature layout (`continuous`, `binary`, `ordinal`, `ordinal_parent`,
  `nominal`) used in the published analyses so tree splits, outer-test AUCs (`0.871 ± 0.010`
  for ODD; `0.871 ± 0.041` for DICCD), SHAP values, and adjusted odds ratios reproduce
  consistently.
- A single shared code path (`odd_diccd_ml`) serves both targets via `--target odd|diccd`,
  reading each outcome's locked-in recipe from [`configs/experiment.yaml`](configs/experiment.yaml)
  so one target's resampler, loss, or tree depth can never bleed into the other.

## Citation

If you use this code, please cite the accompanying manuscript (BibTeX will be updated with
journal, volume, and pages upon publication):

```bibtex
@unpublished{do2026odd,
  title  = {Adolescent-Level and Parent-Level Correlates of Oppositional Defiant Disorder in
            Korean Adolescents: A Machine Learning Analysis of National Survey Data},
  author = {Do, Ryemi and Choe, Daehyeon and Oh, Seungbin and Kim, Soyeon and
            Kong, Hyoun-Joong and Sim, Hewoen and Hwang, Sumin and Park, Sohyun and
            Lee, Chae Rim and Kim, Bung-Nyun},
  year   = {2026},
  note   = {Manuscript in preparation. Ryemi Do and Daehyeon Choe contributed equally as
            co-first authors; Hyoun-Joong Kong and Bung-Nyun Kim jointly supervised this work.}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). The license does not cover the
NMHSK-CA 2022 data, which remain subject to the custodian's terms of use.

## Contact

- Code and analysis: Daehyeon Choe — dablro12@snu.ac.kr
- Co-corresponding author: Hyoun-Joong Kong, PhD — Department of Transdisciplinary Medicine,
  Seoul National University Hospital — gongcop7@snu.ac.kr
- Co-corresponding author: Bung-Nyun Kim, MD, PhD — Division of Child and Adolescent Psychiatry,
  Seoul National University Hospital — kbn1@snu.ac.kr

## Acknowledgments

We thank the adolescents and parents who participated in the National Mental Health Survey of
Korea – Child & Adolescent 2022.
