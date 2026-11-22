#!/usr/bin/env python3
"""Covariate-adjusted association (odds ratio) analysis -- crude vs. adjusted
logistic regression of the target on each candidate exposure, Benjamini-Hochberg
FDR correction applied once across all exposures tested for a single target
(ODD and DICCD corrections are never pooled). Adjusted models control for
sex, age, parental depression (PHQ-9>=9), and parental anxiety (GAD-7>=5),
dropping a covariate from the adjustment set only when it is itself the exposure
being tested. ADHD is never included as an exposure (removed by design -- see README.md).

This mirrors the method already used for the original ADHD-included Table 3 in
this project (see the methods paper); it is reproduced here, not reinvented.

Data note: age is not present in the main analysis CSV in this project's raw
files and is read from a separate raw export (--raw-xlsx). If that file is not
supplied, age is dropped from the adjustment set with a warning (results will
then be adjusted for sex + parental mental health only).

Usage:
    python src/association_analysis.py --target ODD --data-dir data --output-dir results/odd_nested \
        [--raw-xlsx data/adolescents_data.xlsx] [--exposures-file results/odd_nested/permutation_odd_all.csv]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths  # noqa: E402
from stage1_model_comparison import meta_target_column  # noqa: E402

# Derived-variable definitions, matching this project's published (ADHD-included)
# Table 3 construction exactly, minus the ADHD exposure (removed by design).
EXPOSURES = (
    ("Cyberbullying Victimization", "cyber_victim", "yes vs no", None),
    ("Cyberbullying Perpetration", "cyber_perp", "yes vs no", None),
    ("Smartphone Addiction", "phone_z", "per 1 SD", None),
    ("Academic Decline", "academic_decline", "yes vs no", None),
    ("Academic Performance", "grades", "per step toward a lower grade", None),
    ("Bullying Perpetration", "bully_perp", "yes vs no", None),
    ("Educational Screen Time", "edu_screen", ">=3 h/day vs <3 h/day", None),
    ("Parental Anxiety", "gad_high", "GAD-7 >=5 vs <5", "gad_high"),
    ("Parental Depression", "phq_high", "PHQ-9 >=9 vs <9", "phq_high"),
    ("Maternal Alcohol Use", "malc_high", ">=2 times/month vs less", None),
)


def build_exposure_frame(df: pd.DataFrame, target_col: str, age_series: pd.Series | None) -> pd.DataFrame:
    grades = pd.to_numeric(df["ksQ15_t"], errors="coerce")
    grades = grades.where(grades.between(1, 5))
    alcohol = pd.to_numeric(df["FA_M"], errors="coerce")
    alcohol = alcohol.where(alcohol.between(1, 5))
    phone = df["SAS1_total"].astype(float)

    frame = pd.DataFrame({
        "y": df[target_col].astype(int),
        "male": df["sex"].astype(int).eq(1).astype(int),
        "phq_high": df["PHQ_total"].ge(9).astype(float),
        "gad_high": df["GAD_total"].ge(5).astype(float),
        "phone_z": (phone - phone.mean()) / phone.std(ddof=0),
        "cyber_victim": df["CV1_code"].astype(float),
        "cyber_perp": df["CV2_code"].astype(float),
        "academic_decline": df["ksQ16_t"].astype(float),
        "grades": grades.astype(float),
        "bully_perp": df["SV3_code"].astype(float),
        "edu_screen": df["ST1_4"].ge(3).astype(float),
        "malc_high": alcohol.ge(3).astype(float),
    })
    frame.loc[df["PHQ_total"].isna(), "phq_high"] = np.nan
    frame.loc[df["GAD_total"].isna(), "gad_high"] = np.nan
    frame.loc[df["ST1_4"].isna(), "edu_screen"] = np.nan
    frame.loc[alcohol.isna(), "malc_high"] = np.nan
    if age_series is not None:
        frame["age"] = age_series.astype(float).reset_index(drop=True)
    return frame


def fit_logit(frame: pd.DataFrame, exposure: str, drop_covariate: str | None, covariates: list[str]) -> dict:
    covariates = [c for c in covariates if c != drop_covariate]
    use = frame[["y", exposure, *covariates]].dropna()
    y = use["y"].to_numpy()
    crude_x = sm.add_constant(use[[exposure]].astype(float), has_constant="add")
    adjusted_x = sm.add_constant(use[[exposure, *covariates]].astype(float), has_constant="add")
    crude = sm.Logit(y, crude_x).fit(disp=False)
    adjusted = sm.Logit(y, adjusted_x).fit(disp=False)
    return {"n": int(len(use)), "n_cases": int(y.sum()), "crude": _term(crude, exposure), "adjusted": _term(adjusted, exposure)}


def _term(result, name: str) -> dict:
    beta, se = float(result.params[name]), float(result.bse[name])
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    return {"or": float(np.exp(beta)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)), "p": float(result.pvalues[name])}


def fmt_or(term: dict) -> str:
    return f"{term['or']:.2f} ({term['lo']:.2f}-{term['hi']:.2f})"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["ODD", "DICCD"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--raw-xlsx", default=None, type=Path, help="Optional raw export providing 'age' (row-aligned with the main CSV).")
    args = ap.parse_args()

    out = args.output_dir or (REPO_ROOT / "results" / f"{args.target.lower()}_nested")
    out.mkdir(parents=True, exist_ok=True)
    csv_path, _ = data_paths(args.target, args.data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected data at {csv_path}. See data/README.md.")

    df = pd.read_csv(csv_path)
    target_col = meta_target_column(args.target)

    age_series = None
    covariates = ["male", "phq_high", "gad_high"]
    if args.raw_xlsx is not None and args.raw_xlsx.exists():
        raw = pd.read_excel(args.raw_xlsx, sheet_name=0)
        raw = raw.loc[raw["age"] >= 12].reset_index(drop=True)
        if len(raw) == len(df):
            age_series = raw["age"]
            covariates.append("age")
        else:
            warnings.warn("--raw-xlsx row count does not match the main CSV; dropping age from covariates.")
    else:
        warnings.warn("No --raw-xlsx supplied; adjusting for sex + parental mental health only (no age).")

    frame = build_exposure_frame(df, target_col, age_series)

    fits = {label: fit_logit(frame, col, drop, covariates) for label, col, _contrast, drop in EXPOSURES}
    q = dict(zip(fits, multipletests([fits[l]["adjusted"]["p"] for l in fits], method="fdr_bh")[1]))

    rows = []
    for label, col, contrast, _drop in EXPOSURES:
        f = fits[label]
        rows.append({
            "Variable": label, "Contrast": contrast, "N": f["n"], "N_cases": f["n_cases"],
            "Crude OR (95% CI)": fmt_or(f["crude"]), "Adjusted OR (95% CI)": fmt_or(f["adjusted"]),
            "p": fmt_p(f["adjusted"]["p"]), "q (FDR)": fmt_p(q[label]),
        })
    out_df = pd.DataFrame(rows)
    assert not out_df["Variable"].str.contains("adhd", case=False).any(), "ADHD must not appear as an exposure"

    path = out / f"association_{args.target.lower()}.csv"
    out_df.to_csv(path, index=False)
    print("\n=== DONE ===")
    print(out_df.to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
