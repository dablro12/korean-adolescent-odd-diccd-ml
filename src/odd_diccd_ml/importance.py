"""Post-hoc permutation importance (delta-AUC) and SHAP values on outer-test folds only."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from .config import FEATURE_SET_LABEL, TARGET_LABEL, Config
from .data import prepare_xy
from .models import fit_rf_with_loss, resample_xy
from .preprocess import fit_transform_fold


def _pos_shap_matrix(shap_out) -> np.ndarray:
    arr = getattr(shap_out, "values", shap_out)
    if isinstance(arr, list):
        return np.asarray(arr[1])
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def run_importance(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    feature_sets: Optional[Iterable[str]] = None,
    n_repeats: Optional[int] = None,
    no_shap: bool = False,
    n_estimators_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute held-out outer-test permutation importance (delta-AUC) and TreeSHAP."""
    cohort = cohort.lower()
    rec = cfg.recipe(cohort)
    n_est = int(n_estimators_override or rec["n_estimators"])
    repeats = int(n_repeats or cfg.importance["permutation_repeats"])
    perm_seed = int(cfg.importance["permutation_seed"])

    out = cfg.cohort_out_dir(cohort) / "importance"
    out.mkdir(parents=True, exist_ok=True)

    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.outer_seed)
    perm_fold_rows, perm_summary_rows = [], []
    shap_fold_rows, shap_summary_rows = [], []

    for fs in (feature_sets or cfg.feature_sets):
        X_all, y_all = prepare_xy(df, cfg, cohort, fs)
        fold_perm_series, fold_shap_series = [], []

        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr = y_all.iloc[tr_idx].to_numpy(dtype=int)
            y_te = y_all.iloc[te_idx].to_numpy(dtype=int)

            X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
            X_res, y_res = resample_xy(X_tr_ml, pd.Series(y_tr, index=X_tr_ml.index), rec["sampler"], cfg.outer_seed + fold)
            clf = fit_rf_with_loss(
                X_res, y_res, rec["loss"], rec["hyperparameters"],
                cfg.outer_seed + fold, focal_gamma=rec["focal_gamma"], n_estimators=n_est,
            )
            feat_names = list(X_te_ml.columns)

            perm = permutation_importance(
                clf, X_te_ml, y_te, scoring="roc_auc", n_repeats=repeats,
                random_state=perm_seed, n_jobs=-1,
            )
            fold_perm_series.append(pd.Series(perm.importances_mean, index=feat_names, name=f"fold{fold}"))
            for fname, d_auc in zip(feat_names, perm.importances_mean):
                perm_fold_rows.append({
                    "cohort": cohort, "feature_set": fs, "outer_fold": fold,
                    "feature": fname, "label": cfg.label(fname), "domain": cfg.domain_of(fname),
                    "delta_auc": float(d_auc),
                })

            if not no_shap:
                sv = _pos_shap_matrix(shap.TreeExplainer(clf).shap_values(X_te_ml))
                m_abs = np.abs(sv).mean(axis=0)
                fold_shap_series.append(pd.Series(m_abs, index=feat_names, name=f"fold{fold}"))
                for fname, sval in zip(feat_names, m_abs):
                    shap_fold_rows.append({
                        "cohort": cohort, "feature_set": fs, "outer_fold": fold,
                        "feature": fname, "label": cfg.label(fname), "domain": cfg.domain_of(fname),
                        "mean_abs_shap": float(sval),
                    })

            if verbose:
                print(f"[{TARGET_LABEL[cohort]}][{FEATURE_SET_LABEL[fs]}] fold{fold} importance done", flush=True)

        p_mat = pd.concat(fold_perm_series, axis=1).fillna(0.0)
        p_df = pd.DataFrame({
            "cohort": cohort,
            "feature_set": fs,
            "feature": p_mat.index,
            "label": [cfg.label(c) for c in p_mat.index],
            "domain": [cfg.domain_of(c) for c in p_mat.index],
            "delta_auc_mean": p_mat.mean(axis=1).to_numpy(),
            "delta_auc_sd": p_mat.std(axis=1, ddof=1).to_numpy(),
        }).sort_values("delta_auc_mean", ascending=False).reset_index(drop=True)
        p_df["rank"] = np.arange(1, len(p_df) + 1)
        assert not p_df["label"].str.contains("adhd", case=False).any(), "ADHD must not appear in permutation output"
        p_df.to_csv(out / f"permutation_{fs}.csv", index=False)
        perm_summary_rows.append(p_df)

        if not no_shap:
            s_mat = pd.concat(fold_shap_series, axis=1).fillna(0.0)
            s_df = pd.DataFrame({
                "cohort": cohort,
                "feature_set": fs,
                "feature": s_mat.index,
                "label": [cfg.label(c) for c in s_mat.index],
                "domain": [cfg.domain_of(c) for c in s_mat.index],
                "mean_abs_shap": s_mat.mean(axis=1).to_numpy(),
                "sd_abs_shap": s_mat.std(axis=1, ddof=1).to_numpy(),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            s_df["rank"] = np.arange(1, len(s_df) + 1)
            assert not s_df["label"].str.contains("adhd", case=False).any(), "ADHD must not appear in SHAP output"
            s_df.to_csv(out / f"shap_{fs}.csv", index=False)
            shap_summary_rows.append(s_df)

    perm_by_fold = pd.DataFrame(perm_fold_rows)
    perm_summary = pd.concat(perm_summary_rows, ignore_index=True)
    perm_by_fold.to_csv(out / "permutation_by_fold.csv", index=False)
    perm_summary.to_csv(out / "permutation_summary.csv", index=False)

    res = {"permutation_by_fold": perm_by_fold, "permutation_summary": perm_summary}
    if not no_shap and shap_summary_rows:
        shap_by_fold = pd.DataFrame(shap_fold_rows)
        shap_summary = pd.concat(shap_summary_rows, ignore_index=True)
        shap_by_fold.to_csv(out / "shap_by_fold.csv", index=False)
        shap_summary.to_csv(out / "shap_summary.csv", index=False)
        res["shap_by_fold"] = shap_by_fold
        res["shap_summary"] = shap_summary
    return res
