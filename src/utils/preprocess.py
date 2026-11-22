# NOTE (2026-10-28): Changed for the ODD/DICCD analyses.
# build_ml_preprocessor() now moves the parental ordinals (F4_M, F4_F, FA_M, FA_F,
# FT_M, FT_F) into a separate "ordinal_parent" block with median imputation.
# This changes the output column order and the prefix (ordinal__ -> ordinal_parent__).
# The saved suicide results were produced before this change and are not affected.
# Re-running suicide scripts with this file can give different results. With
# encode_nominal=False (e.g. 10_2_wo_encoding.py, 15_1_nested_cv_wo_encoding.py),
# parental ordinal values can end up under the wrong column names, because
# fit_transform_ml_data() renames columns by an assumed order (ct_order).
# Do not re-run suicide analyses with this file without restoring the old
# ordinal handling first.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def _cols_in_frame(df, cols):
    return [c for c in cols if c in df.columns]


def get_ml_feature_lists(meta_dict, columns, target_col):
    """metadata feature_groups에서 현재 X 컬럼에 해당하는 타입별 리스트를 만든다."""
    cols_set = set(columns)
    fg = meta_dict["feature_groups"]
    return {
        "binary": [c for c in fg["binary"] if c in cols_set and c != target_col],
        "ordinal": [c for c in fg["ordinal"] if c in cols_set],
        "nominal": [c for c in fg["nominal"] if c in cols_set],
        "continuous": [c for c in fg["continuous"] if c in cols_set],
    }


# Parental ordinals stay numeric. NaN is median-imputed inside each train fold.
# Other ordinals keep most-frequent imputation. Suicide tables already fill these
# columns before ML, so this branch does not change those completed results.
_PARENT_ORDINAL_MEDIAN = {"F4_M", "F4_F", "FA_M", "FA_F", "FT_M", "FT_F"}


def build_ml_preprocessor(
    binary_cols,
    ordinal_cols,
    nominal_cols,
    continuous_cols,
    encode_nominal: bool = True,
    scale_continuous: bool = True,
):
    """
    ML 모델 입력용 sklearn 전처리기(ColumnTransformer)를 만든다.

    - 연속형: median impute (+ optional StandardScaler)
    - 이진형: most_frequent impute (스케일링 없음)
    - 순서형: most_frequent impute (스케일링 없음)
    - 명목형: most_frequent impute (+ optional OneHotEncoder)
      encode_nominal=False 이면 정수 코드를 그대로 두고 impute만 함

    train 데이터에만 fit하고 test는 transform만 해야 누수가 없다.
    """
    if scale_continuous:
        continuous_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        continuous_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    if encode_nominal:
        nominal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        nominal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

    parent_ordinal = [c for c in ordinal_cols if c in _PARENT_ORDINAL_MEDIAN]
    other_ordinal = [c for c in ordinal_cols if c not in _PARENT_ORDINAL_MEDIAN]
    transformers = [
        ("continuous", continuous_pipe, continuous_cols),
        (
            "binary",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
            binary_cols,
        ),
    ]
    if other_ordinal:
        transformers.append(
            (
                "ordinal",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                other_ordinal,
            )
        )
    if parent_ordinal:
        transformers.append(
            (
                "ordinal_parent",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                parent_ordinal,
            )
        )
    transformers.append(("nominal", nominal_pipe, nominal_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_transform_ml_data(
    X_train,
    X_test,
    meta_dict,
    target_col,
    preprocessor=None,
    encode_nominal: bool = True,
    scale_continuous: bool = True,
):
    """
    train/test 분할 이후 스케일링·인코딩을 적용한다.

    encode_nominal=False 이면 OneHot을 쓰지 않고, 컬럼명·순서를 원본 그대로 유지한다.

    Returns
    -------
    X_train_ml, X_test_ml : pandas.DataFrame
    preprocessor : sklearn ColumnTransformer | dict
    """
    feature_lists = get_ml_feature_lists(meta_dict, X_train.columns, target_col=target_col)
    binary_cols = _cols_in_frame(X_train, feature_lists["binary"])
    ordinal_cols = _cols_in_frame(X_train, feature_lists["ordinal"])
    nominal_cols = _cols_in_frame(X_train, feature_lists["nominal"])
    continuous_cols = _cols_in_frame(X_train, feature_lists["continuous"])

    # No OneHot: keep original names + original column order (Script 10-2)
    if not encode_nominal:
        keep_cols = [
            c
            for c in X_train.columns
            if c in set(binary_cols + ordinal_cols + nominal_cols + continuous_cols)
        ]
        if preprocessor is None:
            preprocessor = build_ml_preprocessor(
                binary_cols=binary_cols,
                ordinal_cols=ordinal_cols,
                nominal_cols=nominal_cols,
                continuous_cols=continuous_cols,
                encode_nominal=False,
                scale_continuous=scale_continuous,
            )
            preprocessor.fit(X_train[keep_cols])
        # Transform then re-order/rename back to original keep_cols
        # ColumnTransformer output order = continuous, binary, ordinal, nominal
        ct_order = continuous_cols + binary_cols + ordinal_cols + nominal_cols
        X_train_arr = preprocessor.transform(X_train[keep_cols])
        X_test_arr = preprocessor.transform(X_test[keep_cols])
        train_ct = pd.DataFrame(X_train_arr, columns=ct_order, index=X_train.index)
        test_ct = pd.DataFrame(X_test_arr, columns=ct_order, index=X_test.index)
        X_train_ml = train_ct[keep_cols].copy()
        X_test_ml = test_ct[keep_cols].copy()
        return X_train_ml, X_test_ml, preprocessor

    if preprocessor is None:
        preprocessor = build_ml_preprocessor(
            binary_cols=binary_cols,
            ordinal_cols=ordinal_cols,
            nominal_cols=nominal_cols,
            continuous_cols=continuous_cols,
            encode_nominal=encode_nominal,
            scale_continuous=scale_continuous,
        )

    X_train_arr = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_ml = pd.DataFrame(
        X_train_arr,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_ml = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )
    return X_train_ml, X_test_ml, preprocessor


def filter_cohort_by_gender(df_all, gender):
    """all 코호트에서 gender(all/lady/man) 서브셋을 반환한다.

    Accepts either `gender` or `sex` column (1=man, 2=lady).
    """
    df = df_all.copy()
    if "gender" not in df.columns and "sex" in df.columns:
        df["gender"] = df["sex"]
    if gender == "all":
        return df
    if "gender" not in df.columns:
        raise KeyError("Expected column 'gender' or 'sex' in dataframe")
    if gender == "lady":
        return df[df["gender"] == 2].copy()
    if gender == "man":
        return df[df["gender"] == 1].copy()
    raise ValueError("gender must be one of: all, lady, man")


def build_gender_balanced_test_indices(
    df_all,
    target_col,
    n_cases_per_gender=25,
    n_controls_per_gender=25,
    random_state=42,
):
    """
    all 코호트에서 test를 한 번 정의한다.
    - cases: man 25 + lady 25
    - controls: man 25 + lady 25
    """
    df_all = df_all.copy()
    if "gender" not in df_all.columns and "sex" in df_all.columns:
        df_all["gender"] = df_all["sex"]

    rng = np.random.default_rng(random_state)
    test_parts = []
    gender_labels = [(1, "man"), (2, "lady")]

    for gender_code, gender_name in gender_labels:
        for class_label, n_pick in [(1, n_cases_per_gender), (0, n_controls_per_gender)]:
            pool = df_all.index[
                (df_all["gender"] == gender_code) & (df_all[target_col] == class_label)
            ].to_numpy()
            class_name = "cases" if class_label == 1 else "controls"
            if len(pool) < n_pick:
                raise ValueError(
                    f"{gender_name} {class_name} 부족: test에 {n_pick}명 필요, 현재 {len(pool)}명"
                )
            test_parts.append(rng.choice(pool, size=n_pick, replace=False))

    return np.concatenate(test_parts)


def load_or_create_test_indices(
    df_all,
    target_col,
    cache_path,
    n_cases_per_gender=25,
    n_controls_per_gender=25,
    random_state=42,
):
    """all 기준 test index를 캐시에 저장/재사용해 lady/man이 동일 test를 쓰게 한다."""
    cache_path = Path(cache_path)
    expected = {
        "seed": random_state,
        "n_cases_per_gender": n_cases_per_gender,
        "n_controls_per_gender": n_controls_per_gender,
    }

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if (
            cached.get("seed") == expected["seed"]
            and cached.get("n_cases_per_gender") == expected["n_cases_per_gender"]
            and cached.get("n_controls_per_gender") == expected["n_controls_per_gender"]
        ):
            return np.asarray(cached["test_indices"], dtype=df_all.index.dtype)

    test_indices = build_gender_balanced_test_indices(
        df_all,
        target_col,
        n_cases_per_gender=n_cases_per_gender,
        n_controls_per_gender=n_controls_per_gender,
        random_state=random_state,
    )
    payload = {
        **expected,
        "test_indices": [int(i) for i in test_indices],
        "test_n": int(len(test_indices)),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return test_indices


def apply_shared_test_split(X, y, test_indices):
    """공유 test index로 train/test 분할 (cohort에 존재하는 test index만 사용)."""
    test_indices = np.asarray(test_indices)
    cohort_test_indices = X.index[X.index.isin(test_indices)]
    if len(cohort_test_indices) == 0:
        raise ValueError("cohort에 test index가 없습니다.")

    test_mask = X.index.isin(cohort_test_indices)
    return (
        X.loc[~test_mask],
        X.loc[test_mask],
        y.loc[~test_mask],
        y.loc[test_mask],
    )
