"""Cohort loading, schema validation, and design-matrix extraction."""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .config import Config


def load_cohort(cfg: Config, cohort: str) -> pd.DataFrame:
    cohort = cohort.lower()
    path = cfg.cohort_csv(cohort)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The study data are not distributed with this repository; "
            "see data/README.md for the access procedure and the expected file layout."
        )
    df = pd.read_csv(path)
    validate_columns(cfg, df, cohort)
    return df


def validate_columns(cfg: Config, df: pd.DataFrame, cohort: str) -> None:
    target_col = cfg.target_col(cohort)
    required = [target_col, cfg.sex_column] + cfg.all_feature_columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV for '{cohort}' is missing required columns: {missing}")
    bad = set(df[target_col].dropna().unique()) - {0, 1}
    if bad:
        raise ValueError(f"Target column '{target_col}' must be coded 0/1, found {sorted(bad)}")
    forbidden = [c for c in cfg.all_feature_columns if "adhd" in c.lower() or "kdbdrs" in c.lower()]
    if forbidden:
        raise ValueError(f"Parent-rated ADHD/K-DBDRS columns must not enter predictors: {forbidden}")


def prepare_xy(
    df: pd.DataFrame,
    cfg: Config,
    cohort: str,
    feature_set: str,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract raw predictor matrix X and binary target y, asserting zero ADHD leakage."""
    cols = [c for c in cfg.feature_columns(feature_set) if c not in set(drop_columns or [])]
    forbidden = [c for c in cols if "adhd" in c.lower() or "kdbdrs" in c.lower()]
    if forbidden:
        raise ValueError(f"Forbidden ADHD/K-DBDRS column(s) selected: {forbidden}")
    X = df[cols].copy()
    y = df[cfg.target_col(cohort)].astype(int).copy()
    return X, y
