#!/usr/bin/env python3
"""Stage 1 (Nested CV): Model comparison, run independently for ODD or DICCD.

Outer stratified 5-fold (seed 42) = unbiased hold-out.
Inner stratified 5-fold (seed 42) within each outer-train = model-selection OOF.
No resampling at this stage. Selection is by inner-OOF AUC; see README.md for why
later stages use a multi-metric (AUC/F1/AP) rank-consistency check instead where
it materially changed the outcome.

Usage:
    python src/stage1_model_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
    python src/stage1_model_comparison.py --target DICCD --data-dir data --output-dir results/diccd_nested
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_feature_sets  # noqa: E402
from utils.models import compute_metrics, get_positive_scores, youden_threshold  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

MODELS = (
    "Logistic Regression", "Naive Bayes", "Linear SVM", "Random Forest",
    "Gradient Boosting", "HistGradientBoosting", "XGBoost", "CatBoost",
)
SCALE_MODELS = {"Logistic Regression", "Naive Bayes", "Linear SVM"}


def select_cols(meta: dict, groups: list[str], target_col: str) -> list[str]:
    cols: list[str] = []
    for g in groups:
        cols.extend(meta.get(g, []))
    return [c for c in cols if c != target_col]


def build_pipeline(model_name: str, seed: int) -> Pipeline:
    steps = []
    if model_name in SCALE_MODELS:
        steps.append(("scaler", StandardScaler()))
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=3000, random_state=seed)
    elif model_name == "Naive Bayes":
        clf = GaussianNB()
    elif model_name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=seed)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=seed)
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(random_state=seed)
    elif model_name == "XGBoost":
        clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=seed, eval_metric="logloss", n_jobs=-1)
    elif model_name == "CatBoost":
        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, random_state=seed, verbose=False)
    else:
        raise ValueError(model_name)
    steps.append(("clf", clf))
    return Pipeline(steps)


def inner_oof_scores(X_tr: pd.DataFrame, y_tr: pd.Series, meta: dict, target_col: str,
                      model_name: str, inner_seed: int, inner_folds: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
    oof = np.zeros(len(y_tr), dtype=float)
    for itr, iva in skf.split(X_tr, y_tr):
        Xi_tr, Xi_va, _ = fit_transform_ml_data(X_tr.iloc[itr], X_tr.iloc[iva], meta, target_col)
        pipe = build_pipeline(model_name, inner_seed)
        pipe.fit(Xi_tr, y_tr.iloc[itr])
        oof[iva] = get_positive_scores(pipe, Xi_va)
    return oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["ODD", "DICCD"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    args = ap.parse_args()

    out = args.output_dir or (REPO_ROOT / "results" / f"{args.target.lower()}_nested")
    out.mkdir(parents=True, exist_ok=True)

    csv_path, meta_path = data_paths(args.target, args.data_dir)
    if not csv_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Expected data at {csv_path} and {meta_path} -- this repository ships no data. "
            f"See data/README.md for how to obtain it under IRB approval."
        )

    cv = get_cv_config()
    feature_sets = get_feature_sets()
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta_target_column(args.target)

    rows, outer_nested_rows = [], []
    outer_cv = StratifiedKFold(n_splits=cv["outer_folds"], shuffle=True, random_state=cv["outer_seed"])
    y_full = df[target_col]

    for key, spec in feature_sets.items():
        flab = spec["label"]
        cols = select_cols(meta, spec["groups"], target_col)
        Xf = df[cols]
        t0 = time.time()
        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(Xf, y_full), start=1):
            X_tr, X_te = Xf.iloc[tr_idx], Xf.iloc[te_idx]
            y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

            fold_inner_auc = {}
            for model_name in MODELS:
                try:
                    oof = inner_oof_scores(X_tr, y_tr, meta, target_col, model_name, cv["inner_seed"], cv["inner_folds"])
                    auc = roc_auc_score(y_tr, oof)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {flab} fold{fold} {model_name} failed: {exc}", flush=True)
                    traceback.print_exc()
                    auc = np.nan
                fold_inner_auc[model_name] = auc
                rows.append({"feature_set": flab, "outer_fold": fold, "model": model_name, "inner_oof_auc": auc})
                print(f"  {flab} fold{fold} {model_name}: inner_oof_auc={auc:.4f}", flush=True)

            winner = max(fold_inner_auc, key=lambda m: (fold_inner_auc[m] if fold_inner_auc[m] == fold_inner_auc[m] else -1))
            X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
            pipe = build_pipeline(winner, cv["outer_seed"] + fold)
            pipe.fit(X_tr_ml, y_tr)
            scores_te = get_positive_scores(pipe, X_te_ml)
            oof_for_threshold = inner_oof_scores(X_tr, y_tr, meta, target_col, winner, cv["inner_seed"], cv["inner_folds"])
            th = youden_threshold(y_tr, oof_for_threshold)
            pred_te = (scores_te >= th).astype(int)
            metrics = compute_metrics(y_te, pred_te, scores_te)
            auc_te = roc_auc_score(y_te, scores_te)
            outer_nested_rows.append({
                "feature_set": flab, "outer_fold": fold, "selected_model": winner,
                "outer_test_auc": auc_te, **metrics,
            })
            print(f"  >> {flab} fold{fold} selected={winner} outer_test_auc={auc_te:.4f}", flush=True)
        print(f"{flab} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(out / "stage1_inner_oof_long.csv", index=False)
    outer_df = pd.DataFrame(outer_nested_rows)
    outer_df.to_csv(out / "stage1_outer_nested_auc.csv", index=False)

    summary_rows = []
    for key, spec in feature_sets.items():
        flab = spec["label"]
        sub = pd.DataFrame(rows)
        sub = sub[sub.feature_set == flab]
        for model_name in MODELS:
            vals = sub[sub.model == model_name]["inner_oof_auc"].dropna().to_numpy()
            n = len(vals)
            m = float(vals.mean()) if n else float("nan")
            sd = float(vals.std(ddof=1)) if n > 1 else 0.0
            se = sd / np.sqrt(n) if n > 1 else 0.0
            tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
            summary_rows.append({
                "Feature set": flab, "Model": model_name,
                "Inner-OOF AUC (mean)": m, "95% CI low": m - tcrit * se, "95% CI high": m + tcrit * se, "n_outer_folds": n,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "stage1_model_comparison.csv", index=False)

    combined_best = summary[summary["Feature set"] == "Combined"].sort_values("Inner-OOF AUC (mean)", ascending=False).iloc[0]
    print("\n=== DONE ===")
    print(f"Overall winner to carry forward (Combined, by inner-OOF AUC): {combined_best['Model']}")
    print(f"Wrote outputs to {out}")


def meta_target_column(target: str) -> str:
    return {"ODD": "ODD_CP", "DICCD": "DICCD_CP"}[target.upper()]


if __name__ == "__main__":
    main()
