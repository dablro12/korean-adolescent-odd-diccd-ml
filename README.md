# Korean Adolescent ODD/DICCD Classification (ADHD-Excluded Reanalysis)

👋 **Welcome!** This repository holds the machine learning pipeline behind a study of
adolescent Oppositional Defiant Disorder (ODD) and Disruptive, Impulse-Control, and Conduct
Disorder (DICCD) in a Korean adolescent cohort. If you're a reviewer, a reader of the paper,
or just curious how the analysis works, this README will walk you through it — no prior
familiarity with the project assumed.

📄 **Paper:** *Coming soon* — this repository accompanies a manuscript currently in
preparation / under review. Title and venue to be determined.

**Authors:** Daehyeon Choe† · Ryemi Do† · *(additional co-authors to be finalized)* ·
Hyoun-Joong Kong* · Bung-Nyun Kim*
† Co-first authors. \* Co-corresponding authors.

## What is this study about?

We built a model that predicts ODD and DICCD from survey answers given by Korean
adolescents (ages 12–17) and, separately, by their parents. An earlier analysis had
included an ADHD-symptom score as one of the predictors — since ADHD symptoms are known to
overlap heavily with both conditions, we re-ran the whole analysis **with that feature
removed**, to see what rises to the top once it's no longer dominating the model.

The short version of what we found: once ADHD is out of the picture, **smartphone
addiction** and **cyberbullying victimization** become the strongest predictors of both ODD
and DICCD — consistently across three independent lines of evidence (SHAP, permutation
importance, and covariate-adjusted odds ratios).

We also didn't assume ODD and DICCD needed the same recipe. Each target went through its
own independent 4-stage search (Model → Sampler → Loss → Hyperparameter), and they landed on
genuinely different best settings — see [Methodology](#methodology) below.

## Quick start

New here and just want to see it run? You don't need the real data for this part:

```bash
git clone https://github.com/dablro12/korean-adolescent-odd-diccd-ml.git
cd korean-adolescent-odd-diccd-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/example_walkthrough.ipynb
```

That notebook is fully self-contained — it builds a small synthetic dataset on the spot and
runs the same preprocessing → resampling → fitting → thresholding → explainability steps
used in the real pipeline, so you can see the mechanics work without needing data access.
(The numbers in that notebook are synthetic, not the study's real findings — see
[Methodology](#methodology) for the real results.)

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
    └── example_walkthrough.ipynb    # standalone demo on synthetic data (no data access needed)
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed and tested against Python 3.10, with `scikit-learn==1.7.2`,
`imbalanced-learn==0.14.1`, `shap==0.51.0`, `optuna==4.8.0`, `statsmodels==0.14.6`.

## Data availability

**No participant data is included in this repository**, and none of the study's result
numbers are bundled here either — both are intentionally kept out until the manuscript is
further along. The dataset itself is IRB-restricted human-subjects data. See
[`data/README.md`](data/README.md) for how to request access and the exact file layout /
schema the code expects once data is obtained.

In the meantime, [`notebooks/example_walkthrough.ipynb`](notebooks/example_walkthrough.ipynb)
lets you see every step of the pipeline run on a small synthetic dataset, so you can confirm
the code works and follow the logic without needing data access.

## Usage

Once data is placed per `data/README.md`, you can run the pipeline stage by stage, or all at
once:

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

The `--target` flag is the only thing that changes between an ODD run and a DICCD run — it
picks which locked-in recipe to read from `configs/recipes.yaml`. There's no separate "ODD
version" / "DICCD version" of any script, by design (see the note on this in
[Methodology](#methodology) — keeping one shared code path was a deliberate choice to avoid
a class of bug we ran into with an earlier, duplicated-script version of this pipeline).

## Methodology

If you're not familiar with nested cross-validation, here's the short version: we're
comparing a lot of candidates (8 models, 8 resampling methods, 3 loss functions, dozens of
hyperparameter settings), and if you let that comparison "peek" at the same data you later
report performance on, your numbers end up optimistically biased. Nested CV avoids this by
keeping two loops strictly separate — an **inner** loop that's allowed to look around and
pick a winner, and an **outer** loop whose test fold is never touched until the winner is
already locked in.

**Nested cross-validation**: outer 5-fold × inner 5-fold `StratifiedKFold(shuffle=True,
random_state=42)`. The inner loop is used only for candidate selection (which model /
sampler / loss / hyperparameter config, and the Youden decision threshold); the outer-test
fold is touched exactly once, after that selection is final, to produce the reported
metric. Preprocessing (imputation, scaling, one-hot encoding) is fit on the training
partition only at every CV level — never on the held-out partition.

**Selection discipline**: each stage locks in the previous stage's winner before moving to
the next. Candidates are **not** selected by raw AUC alone — a rank-consistency score
across AUC, F1, and Average Precision is computed, and the raw-AUC winner is overridden
when it is materially worse on F1/AP. This mattered in practice: the AUC-best resampling
method for both targets ranked 6th–8th out of 8 candidates on F1/Average-Precision, and was
not used in the end.

**Final locked-in recipes** (independently determined per target — ODD and DICCD were never
assumed to need the same settings):

| | ODD | DICCD |
|---|---|---|
| Model | Random Forest | Random Forest |
| Sampler (train-only) | ADASYN (1:1) | None (no resampling) |
| Loss | Focal (γ=2.0, class-balanced α) | BCE (unweighted) |
| Hyperparameters | `criterion=gini, min_samples_leaf=1, max_depth=None, max_features=sqrt` | `criterion=entropy, min_samples_leaf=1, max_depth=12, max_features=sqrt` |
| Final Outer-Test AUC (Combined feature set) | 0.871 | 0.866 (0.871 under a wider Optuna search) |

As a robustness check, we also ran a much wider, partly-continuous Optuna (TPE) search
(`stage4_optuna_search.py`) against the manual grid above, to make sure we weren't missing a
meaningfully better configuration: for ODD it did not beat the grid (tied AUC, worse F1);
for DICCD it found a marginal AUC gain (+0.005) with no single stable config winning across
folds. Both results point the same way — the manual grid was already close to optimal,
rather than either recipe needing to change.

**Downstream interpretation**: permutation importance (`scoring="roc_auc"`, `n_repeats=8`)
and SHAP (`TreeExplainer`) are both computed on the held-out outer-test fold only, never on
train, using each target's own locked-in recipe. Covariate-adjusted odds ratios (logistic
regression, adjusted for sex, age, parental depression, parental anxiety) use
Benjamini-Hochberg FDR correction applied once per target.

## Questions or data requests?

See [`data/README.md`](data/README.md) for the data access process. For anything else about
this repository or the methodology, feel free to open a GitHub issue — we're happy to
clarify.

## Citation

Citation details (BibTeX) will be added once the accompanying manuscript is accepted.

```bibtex
@article{coming_soon_odd_diccd_korean_adolescent,
  title   = {[Title to be determined]},
  author  = {Choe, Daehyeon and Do, Ryemi and {(additional co-authors to be finalized)} and Kong, Hyoun-Joong and Kim, Bung-Nyun},
  note    = {Daehyeon Choe and Ryemi Do contributed equally (co-first authors). Hyoun-Joong Kong and Bung-Nyun Kim are co-corresponding authors.},
  journal = {Coming soon},
  year    = {Coming soon}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
