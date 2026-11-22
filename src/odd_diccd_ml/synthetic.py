"""Synthetic data with the exact study schema -- for smoke tests and demos ONLY.

The generator reproduces column names, codings, and rough value ranges of the NMHSK-CA 2022
adolescent ODD and DICCD extracts so the entire pipeline can be executed end-to-end without
access to restricted survey data. Values are random draws with an artificial log-odds signal:
results obtained on synthetic data say nothing about Korean adolescents and must not be
compared with the published estimates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

_BINARY_P = {
    "CS_code": 0.12, "SV2_code": 0.14, "SV3_code": 0.12,
    "CV1_code": 0.20, "CV2_code": 0.18, "ksQ16_t": 0.18,
    "ksQ22_t": 0.89, "ksQ24_t": 0.58, "PS_code": 0.22,
    "mother_missing": 0.03, "father_missing": 0.08,
}

_SIGNAL = {
    "CV1_code": 1.35,
    "CV2_code": 1.05,
    "SV3_code": 0.90,
    "ksQ16_t": 0.85,
    "SAS1_total": 0.07,
    "ST1_4": 0.22,
    "ksQ15_t": 0.25,
    "GAD_total": 0.11,
    "PHQ_total": 0.08,
    "FA_M": 0.20,
}


def make_synthetic_cohort(
    cfg: Config,
    cohort: str,
    n: int = 800,
    prevalence: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = cfg.features["categories"]
    target_col = cfg.target_col(cohort)

    df = pd.DataFrame({cfg.sex_column: rng.choice([1, 2], size=n)})
    for col, p in _BINARY_P.items():
        df[col] = rng.binomial(1, p, size=n)
    for col in ("PHA1", "ST1_1", "ST1_2", "ST1_3", "ST1_4", "ksQ15_t"):
        df[col] = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.32, 0.34, 0.17, 0.11, 0.06])
    for col in ("F4_M", "F4_F"):
        df[col] = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.35, 0.50, 0.08, 0.02])
    for col in ("FA_M", "FA_F", "FT_M", "FT_F"):
        df[col] = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    for col in ("dong", "IGD_P", "F3", "F5M", "F5F"):
        codes = [float(k) for k in cats[col]]
        df[col] = rng.choice(codes, size=n)

    df["SAS1_total"] = np.clip(rng.normal(24, 7, size=n), 10, 60).round()
    df["F6"] = rng.integers(1, 13, size=n) * 100.0
    df["PHQ_total"] = np.clip(rng.gamma(0.9, 3.2, size=n), 0, 27).round()
    df["GAD_total"] = np.clip(rng.gamma(0.85, 2.8, size=n), 0, 21).round()

    logit = sum(w * df[c] for c, w in _SIGNAL.items()) + rng.normal(0, 1.1, size=n)
    cut = float(np.quantile(logit, 1.0 - prevalence))
    df[target_col] = (logit >= cut).astype(int)

    # Sprinkle a small fraction of missing values so train-only imputation is exercised
    for col in ("PHQ_total", "GAD_total", "F6", "ksQ15_t"):
        df.loc[rng.random(n) < 0.02, col] = np.nan

    return df[[cfg.sex_column, target_col] + cfg.all_feature_columns]


def write_synthetic_bundle(cfg: Config, out_dir: Path, n: int = 800, seed: int = 42) -> Dict[str, Path]:
    """Write schema-identical synthetic ODD CSV, DICCD CSV, and age Excel file to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for idx, (cohort, prev) in enumerate((("odd", 0.06), ("diccd", 0.07))):
        df = make_synthetic_cohort(cfg, cohort=cohort, n=n, prevalence=prev, seed=seed + idx)
        p = out_dir / cfg.cohorts[cohort]["csv"]
        df.to_csv(p, index=False)
        paths[cohort] = p

    rng = np.random.default_rng(seed + 99)
    age_df = pd.DataFrame({"age": rng.integers(12, 18, size=n)})
    age_path = out_dir / cfg.age_xlsx_name
    age_df.to_excel(age_path, index=False)
    paths["age_xlsx"] = age_path
    return paths
