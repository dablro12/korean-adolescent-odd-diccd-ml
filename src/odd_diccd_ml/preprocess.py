"""Train-only imputation, z-scoring, parental-ordinal handling, and one-hot encoding."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def build_preprocessor(
    continuous_cols: List[str],
    binary_cols: List[str],
    ordinal_cols: List[str],
    ordinal_parent_cols: List[str],
    nominal_cols: List[str],
) -> ColumnTransformer:
    """Construct the 5-block leakage-safe ColumnTransformer used in the published analyses.

    - continuous: median imputation + z-scoring (StandardScaler)
    - binary: mode (most_frequent) imputation
    - ordinal (adolescent): mode (most_frequent) imputation
    - ordinal_parent (F4_M, F4_F, FA_M, FA_F, FT_M, FT_F): median imputation
    - nominal: mode imputation + OneHotEncoder(handle_unknown='ignore')
    """
    transformers = []
    if continuous_cols:
        transformers.append((
            "continuous",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            continuous_cols,
        ))
    if binary_cols:
        transformers.append((
            "binary",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
            binary_cols,
        ))
    if ordinal_cols:
        transformers.append((
            "ordinal",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
            ordinal_cols,
        ))
    if ordinal_parent_cols:
        transformers.append((
            "ordinal_parent",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            ordinal_parent_cols,
        ))
    if nominal_cols:
        transformers.append((
            "nominal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            nominal_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_transform_fold(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing strictly on X_train and transform both X_train and X_eval."""
    present = set(X_train.columns)
    t = cfg.type_lists()
    pre = build_preprocessor(
        continuous_cols=[c for c in t["continuous"] if c in present],
        binary_cols=[c for c in t["binary"] if c in present],
        ordinal_cols=[c for c in t["ordinal"] if c in present],
        ordinal_parent_cols=[c for c in t["ordinal_parent"] if c in present],
        nominal_cols=[c for c in t["nominal"] if c in present],
    )
    X_tr_arr = pre.fit_transform(X_train)
    names = list(pre.get_feature_names_out())
    X_tr_df = pd.DataFrame(X_tr_arr, columns=names, index=X_train.index)
    X_ev_df = pd.DataFrame(pre.transform(X_eval), columns=names, index=X_eval.index)
    return X_tr_df, X_ev_df, pre
