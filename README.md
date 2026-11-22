# Korean Adolescent ODD/DICCD Classification (ADHD-Excluded Reanalysis)

Nested cross-validated machine learning pipeline for classifying adolescent Oppositional
Defiant Disorder (ODD) and Disruptive, Impulse-Control, and Conduct Disorder (DICCD), with
the ADHD-symptom-score feature removed, plus downstream SHAP / permutation-importance /
covariate-adjusted-association analysis.

📄 **Paper:** *Coming soon* — this repository accompanies a manuscript currently in
preparation / under review. Citation details will be added on acceptance.

## Overview

This project analyzes a Korean adolescent survey cohort (ages 12–17) with linked
parent-report items to predict ODD and DICCD. A prior analysis included an ADHD-symptom-score
feature; this repository implements the **ADHD-excluded reanalysis** — running an
independent 4-stage nested-CV comparison (Model → Sampler → Loss → Hyperparameter) for
**each target separately**, rather than assuming one target's optimal recipe applies to
the other. The two targets converged on genuinely different final recipes (see
[Methodology](#methodology)) at similar performance (Combined-feature-set Outer-Test AUC
~0.87 for both). With ADHD removed, **Smartphone Addiction** and **Cyberbullying
Victimization** emerge as the top predictors for both ODD and DICCD across SHAP,
permutation importance, and covariate-adjusted association analysis.

## Repository structure

```
korean-adolescent-odd-diccd-ml/
├── README.md
├── LICENSE                          # Apache 2.0
├── requirements.txt
├── configs/
│   └── recipes.yaml                 # locked-in per-target recipe + CV/feature-set config
├── data/
│   ├── README.md                    # data availability statement + expected schema
│   └── metadata_schema_example.json # structural example only, no participant data
├── src/
│   ├── recipe.py                    # loads configs/recipes.yaml
│   ├── stage1_model_comparison.py   # Stage 1: 8-model comparison
│   ├── stage2_sampler_comparison.py # Stage 2: 8-sampler comparison (Random Forest fixed)
│   ├── stage3_loss_comparison.py    # Stage 3: BCE / Weighted BCE / Focal comparison
│   ├── stage4_hyperparameter_comparison.py  # Stage 4: 24-config grid search
│   ├── stage4_optuna_search.py      # Stage 4': wider Optuna (TPE) search, robustness check
│   ├── permutation_importance.py    # held-out-fold permutation importance (ΔAUC)
│   ├── shap_importance.py           # held-out-fold SHAP (TreeExplainer)
│   ├── association_analysis.py      # covariate-adjusted OR + Benjamini-Hochberg FDR
│   ├── correlation_analysis.py      # Spearman correlation + heatmap
│   ├── run_pipeline.py              # chains all of the above for one target
│   └── utils/
│       ├── preprocess.py            # fit_transform_ml_data (train-only fit, leakage-safe)
│       ├── models.py                # compute_metrics, youden_threshold, get_positive_scores
│       ├── figure_style.py
│       └── exp15_publication_labels.py
└── notebooks/
    └── example_walkthrough.ipynb    # end-to-end demo on synthetic data (no real data needed)
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed and tested against Python 3.10, with `scikit-learn==1.7.2`,
`imbalanced-learn==0.14.1`, `shap==0.51.0`, `optuna==4.8.0`, `statsmodels==0.14.6`.

## Data availability

**No participant data is included in this repository.** The dataset is IRB-restricted
human-subjects data. See [`data/README.md`](data/README.md) for the data-access process
and the exact file layout / schema the code expects once data is obtained.

To try the pipeline without real data, run [`notebooks/example_walkthrough.ipynb`](notebooks/example_walkthrough.ipynb),
which generates a small synthetic dataset matching the real schema's shape and runs the
pipeline on it end-to-end.

## Usage

Once data is placed per `data/README.md`:

```bash
# Run one stage at a time
python src/stage1_model_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
python src/stage2_sampler_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
python src/stage3_loss_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
python src/stage4_hyperparameter_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
python src/stage4_optuna_search.py --target ODD --data-dir data --output-dir results/odd_nested --n-trials 40
python src/permutation_importance.py --target ODD --feature-set all --data-dir data --output-dir results/odd_nested
python src/shap_importance.py --target ODD --feature-set all --data-dir data --output-dir results/odd_nested
python src/association_analysis.py --target ODD --data-dir data --output-dir results/odd_nested
python src/correlation_analysis.py --target ODD --data-dir data --output-dir results/odd_nested

# Or run everything for one target in sequence
python src/run_pipeline.py --target ODD --data-dir data --output-dir results/odd_nested
python src/run_pipeline.py --target DICCD --data-dir data --output-dir results/diccd_nested
```

`--target` selects which locked-in recipe (`configs/recipes.yaml`) is used; the same code
path runs for both — there is no separate "ODD version" / "DICCD version" of any script.

## Methodology

**Nested cross-validation**: outer 5-fold × inner 5-fold `StratifiedKFold(shuffle=True,
random_state=42)`. The inner loop is used only for candidate selection (which model /
sampler / loss / hyperparameter config, and the Youden decision threshold); the outer-test
fold is touched exactly once, after that selection is final, to produce the reported
metric. Preprocessing (imputation, scaling, one-hot encoding) is fit on the training
partition only at every CV level — never on the held-out partition.

**Selection discipline**: each stage locks in the previous stage's winner before moving to
the next. Candidates are **not** selected by raw AUC alone — a rank-consistency score
across AUC, F1, and Average Precision is computed, and the raw-AUC winner is overridden
when it is materially worse on F1/AP (this mattered in practice: the AUC-best sampler for
both targets ranked 6th–8th of 8 on F1/AP and was not used).

**Final locked-in recipes** (independently determined per target):

| | ODD | DICCD |
|---|---|---|
| Model | Random Forest | Random Forest |
| Sampler (train-only) | ADASYN (1:1) | None (no resampling) |
| Loss | Focal (γ=2.0, class-balanced α) | BCE (unweighted) |
| Hyperparameters | `criterion=gini, min_samples_leaf=1, max_depth=None, max_features=sqrt` | `criterion=entropy, min_samples_leaf=1, max_depth=12, max_features=sqrt` |
| Final Outer-Test AUC (Combined feature set) | 0.871 | 0.866 (0.871 under a wider Optuna search) |

A wider, partly-continuous Optuna (TPE) hyperparameter search (`stage4_optuna_search.py`)
was run as a robustness check against the manual grid: for ODD it did not beat the grid
(tied AUC, worse F1); for DICCD it found a marginal AUC gain (+0.005) with no single
stable config winning across folds. Both results support that the manual grid was already
close to optimal, rather than indicating either recipe should change.

**Downstream interpretation**: permutation importance (`scoring="roc_auc"`, `n_repeats=8`)
and SHAP (`TreeExplainer`) are both computed on the held-out outer-test fold only, never on
train, using each target's own locked-in recipe. Covariate-adjusted odds ratios (logistic
regression, adjusted for sex, age, parental depression, parental anxiety) use
Benjamini-Hochberg FDR correction applied once per target.

## Citation

Citation details (BibTeX) will be added once the accompanying manuscript is accepted.

```bibtex
@article{coming_soon_odd_diccd_korean_adolescent,
  title   = {Coming soon},
  author  = {Coming soon},
  journal = {Coming soon},
  year    = {Coming soon}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
