# Data availability

**No participant data is included in this repository.**

The dataset is a Korean adolescent survey cohort (ages 12-17) with linked parent-report
items, collected under institutional review board (IRB) approval. As a human-subjects
dataset with no public-release consent, it cannot be redistributed in this repository.

## Requesting access

Data requests are evaluated case by case and require:
- A description of the intended research use.
- The requester's own IRB approval (or equivalent ethics review) covering secondary
  analysis of this dataset.
- A signed data use agreement.

Contact: `[Corresponding author email -- to be added]`

Turnaround time varies; please allow for institutional review on both sides.

## Expected file layout

Once data is obtained, place it under this `data/` directory (or point `--data-dir` at
wherever you keep it) with the following files:

```
data/
  adolescents_data_ODD_all.csv              # one row per adolescent, ODD cohort
  adolescents_data_odd_metadata_noADHD.json # feature-group metadata, ADHD symptom item removed
  adolescents_data_DICCD_all.csv            # one row per adolescent, DICCD cohort
  adolescents_data_diccd_metadata_noADHD.json
  adolescents_data.xlsx                     # (optional) raw export providing 'age', used only by association_analysis.py
```

## Metadata JSON schema

Each metadata file groups the CSV's columns by respondent and by statistical type, and
is read by `src/recipe.py` / the stage scripts via `--data-dir`. Structure:

```json
{
  "other_features": ["dong"],
  "adolescent_features": ["PHA1", "ST1_1", "..."],
  "adult_features": ["F4_M", "FA_M", "..."],
  "feature_types": {"<column>": "binary|ordinal|nominal|continuous", "...": "..."},
  "feature_groups": {
    "binary": ["..."],
    "ordinal": ["..."],
    "nominal": ["..."],
    "continuous": ["..."]
  }
}
```

- `other_features`: cohort-level fields not tied to a specific respondent (e.g. region code).
- `adolescent_features`: self-reported adolescent survey items (screen time, cyberbullying,
  smartphone addiction, academic items, etc.).
- `adult_features`: parent-reported items (parental mental health, alcohol use, education,
  occupation, household income, etc.).
- `feature_types` / `feature_groups`: tell the preprocessing pipeline (`src/utils/preprocess.py`)
  which imputation/encoding strategy to use per column (median-impute+scale for continuous,
  most-frequent-impute for binary/ordinal, most-frequent-impute+one-hot for nominal).

The `_noADHD` metadata variants are identical to the project's full metadata except that the
ADHD-symptom-score item (`KDBDRS_ADHD`) has been removed from `adolescent_features`,
`feature_types`, and `feature_groups.continuous` -- see the methods paper for why (this
repository's entire analysis is the "ADHD excluded" sensitivity/main reanalysis).

A small structural example with the real group-key names (no participant data, no
numeric content) is provided at `data/metadata_schema_example.json` for reference.

## Target columns

- ODD cohort CSV: binary target column `ODD_CP` (current-or-past Oppositional Defiant Disorder).
- DICCD cohort CSV: binary target column `DICCD_CP` (current-or-past Disruptive,
  Impulse-Control, and Conduct Disorder).
