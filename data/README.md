# Data

**No participant-level data are distributed with this repository.**

The analyses use the **National Mental Health Survey of Korea – Child & Adolescent 2022
(NMHSK-CA 2022)**, a nationally representative household survey of adolescents aged 12–17 and
their caregivers. The data are restricted human-subjects microdata and cannot be redistributed.
This secondary analysis was exempted from review by the Institutional Review Board of Seoul
National University Hospital (exemption No. E-2412-038-1594).

## Requesting access

Access to NMHSK-CA 2022 microdata is managed by the survey's data custodian, the
National Center for Mental Health (Republic of Korea). Requests generally require an approved
research protocol and an IRB determination from the requesting institution. Questions about
the variables and coding used here can be directed to the corresponding authors (see the main
[README](../README.md#contact)).

## Expected files

Once access is granted, place the analysis extracts in this folder (or pass `--data-dir`):

| File | Cohort | Rows | Cases (`= 1`) | Definition |
|---|---|---:|---:|---|
| `adolescents_data_ODD_all.csv` | `odd` | 3,382 | 177 (5.2%) | current-or-past Oppositional Defiant Disorder (`ODD_CP = 1`, KSADS-COMP) vs. none |
| `adolescents_data_DICCD_all.csv` | `diccd` | 3,382 | 197 (5.8%) | current-or-past Disruptive, Impulse-Control, and Conduct Disorder (`DICCD_CP = 1`) vs. none |
| `adolescents_data.xlsx` *(optional)* | both | 3,382 (`age >= 12`) | — | provides continuous `age` (years) used only as a covariate in `association.py` |

File names are set in [`configs/experiment.yaml`](../configs/experiment.yaml). One row per
adolescent–caregiver pair, with the target column (`ODD_CP` or `DICCD_CP`), `sex`, and the
**32 raw survey predictors** below (expanding to **63 encoded features** after one-hot
encoding: 20 Adolescent + 43 Parent). Column codes follow the NMHSK-CA 2022 codebook, and the
complete schema is in [`configs/features.yaml`](../configs/features.yaml).

> **Note on ADHD symptom exclusion:** All parent-rated Korean Disruptive Behavior Disorders
> Rating Scale items (including `KDBDRS_ADHD`) are excluded from predictors by design so that
> co-occurring externalizing symptom ratings do not dominate psychosocial correlates;
> `odd_diccd_ml.data` enforces a runtime guard rejecting any `adhd` or `kdbdrs` predictor column.

| Column | Type | Set | Description |
|---|---|---|---|
| `ODD_CP` / `DICCD_CP` | binary | target | 1 = current-or-past case, 0 = control |
| `sex` | code | covariate only | 1 = male, 2 = female (adjusted in logistic models, **never** an ML predictor) |
| `dong` | nominal | Adolescent, Combined | residential area: 1 = urban (dong), 2 = rural (eup/myeon) |
| `PHA1` | ordinal | Adolescent | days/week with ≥60 min physical activity (1 = 0 days … 5 = every day) |
| `ST1_1`–`ST1_4` | ordinal | Adolescent | daily video / social-media / gaming / educational screen time (1 = <1 h … 5 = ≥7 h) |
| `CS_code` | binary | Adolescent | adolescent sleep problem |
| `SV2_code`, `SV3_code` | binary | Adolescent | school bullying victimization / perpetration |
| `CV1_code`, `CV2_code` | binary | Adolescent | cyberbullying victimization / perpetration |
| `SAS1_total` | continuous | Adolescent | Smartphone Addiction Scale (SAS-SV) total score |
| `IGD_P` | nominal | Adolescent | Internet gaming disorder, SCI-IGD (0 = no, 1 = high-risk, 9999 = unknown) |
| `ksQ15_t` | ordinal | Adolescent | self-reported academic standing (1 = top … 5 = bottom) |
| `ksQ16_t` | binary | Adolescent | recent decline in academic performance |
| `ksQ22_t`, `ksQ24_t` | binary | Adolescent | presence of a best friend / a regular peer group |
| `F3` | nominal | Parent | parental marital status |
| `F4_M`, `F4_F` | ordinal | Parent | maternal / paternal education (1 = ≤middle school … 4 = graduate, 5 = unknown) |
| `F5M`, `F5F` | nominal | Parent | maternal / paternal occupation (KSCO major group) |
| `F6` | continuous | Parent | monthly household income (10,000 KRW units) |
| `PS_code` | binary | Parent | responding caregiver sleep problem |
| `PHQ_total`, `GAD_total` | continuous | Parent | parental PHQ-9 / GAD-7 total score |
| `FA_M`, `FA_F` | ordinal | Parent | maternal / paternal drinking frequency (1 = never … 5 = ≥4/week) |
| `FT_M`, `FT_F` | ordinal | Parent | maternal / paternal smoking intensity (1 = non-smoker … 5 = ≥31 cigarettes/day) |
| `mother_missing`, `father_missing` | binary | Parent | maternal / paternal caregiver information absent in household |

The **Combined** feature set contains every predictor above (32 raw variables; 63 encoded
features). The **Adolescent** set contains `dong` and the 16 adolescent variables (20 encoded
features). The **Parent** set contains the 15 parent variables only (43 encoded features).

### Coding conventions

* `9999` (unknown / don't know / not applicable) and `9998` (other / none) are kept as observed
  codes, exactly as in the published analysis. They become their own one-hot category for
  nominal variables.
* Missing values (`NaN`) are imputed **inside each training partition**: median for continuous
  and parental ordinal variables (`F4_M`, `F4_F`, `FA_M`, `FA_F`, `FT_M`, `FT_F`), and mode
  otherwise.
* No other cleaning happens inside the pipeline. The CSVs above are the analysis-ready
  extracts derived from the raw survey files.

## Synthetic stand-in data

To verify that every command runs end-to-end without restricted data access, generate random
data matching the exact schema:

```bash
cd src
python -m odd_diccd_ml synthetic --out-dir ../data/synthetic
python -m odd_diccd_ml --data-dir ../data/synthetic locked-cv --target odd --n-estimators 100
python -m odd_diccd_ml --data-dir ../data/synthetic importance --target odd --feature-sets combined --n-estimators 100
python -m odd_diccd_ml --data-dir ../data/synthetic association --target odd
```

You can also run [`notebooks/odd_diccd_ml_walkthrough.ipynb`](../notebooks/odd_diccd_ml_walkthrough.ipynb)
directly with its default `MODE = "synthetic"`.

The synthetic values are random draws with an artificial log-odds signal. They say nothing
about the study population and must not be compared with the published estimates.

`data/*` (except `data/README.md`) and `/results/` are git-ignored, so restricted data and
participant-level outputs cannot be committed by accident.
