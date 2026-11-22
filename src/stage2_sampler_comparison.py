#!/usr/bin/env python3
"""Stage 2 (Nested CV): Sampler comparison. Random Forest fixed (Stage 1 winner,
same for both targets). Run with --target ODD or --target DICCD.

Selection discipline: the script prints the raw "best by AUC" pick, but also
computes a rank-consistency score across AUC / F1 / Average-Precision. For both
targets in this project, the raw-AUC winner (RandomOverSampler for ODD,
SMOTEENN for DICCD) ranked 6th-8th of 8 on F1/AP -- a worse, less-balanced
choice than the rank-consistency runner-up. configs/recipes.yaml records the
already-decided final choice (ADASYN for ODD, None for DICCD); this script
reproduces the comparison that justified it, it does not re-decide it.

Usage:
    python src/stage2_sampler_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
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
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_feature_sets  # noqa: E402
from stage1_model_comparison import meta_target_column, select_cols  # noqa: E402
from utils.models import compute_metrics, get_positive_scores, youden_threshold  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

POS, NEG = 1, 0
SAMPLERS = ("None", "SMOTE", "ADASYN", "Borderline-SMOTE", "KMeansSMOTE", "RandomOverSampler", "SMOTETomek", "SMOTEENN")


def safe_k(n_pos: int) -> int:
    return max(1, min(5, n_pos - 1))


def make_sampler(method: str, n_pos: int, n_neg: int, seed: int):
    if method == "None":
        return None
    if n_pos >= n_neg:
        raise RuntimeError(f"positive class not minority: pos={n_pos} neg={n_neg}")
    k = safe_k(n_pos)
    target = {POS: n_neg}
    if method == "SMOTE":
        return SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k)
    if method == "ADASYN":
        return ADASYN(sampling_strategy=target, random_state=seed, n_neighbors=k)
    if method == "Borderline-SMOTE":
        return BorderlineSMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k)
    if method == "KMeansSMOTE":
        return KMeansSMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k, cluster_balance_threshold=0.01)
    if method == "RandomOverSampler":
        return RandomOverSampler(sampling_strategy=target, random_state=seed)
    if method == "SMOTETomek":
        return SMOTETomek(smote=SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k), random_state=seed)
    if method == "SMOTEENN":
        return SMOTEENN(smote=SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k), random_state=seed)
    raise ValueError(method)


def fit_rf_with_sampler(X_tr_ml: pd.DataFrame, y_tr: pd.Series, method: str, seed: int) -> RandomForestClassifier:
    n_pos, n_neg = int((y_tr == POS).sum()), int((y_tr == NEG).sum())
    sampler = make_sampler(method, n_pos, n_neg, seed)
    if sampler is None:
        X_res, y_res = X_tr_ml, y_tr
    else:
        X_arr, y_arr = sampler.fit_resample(X_tr_ml.to_numpy(), y_tr.to_numpy())
        X_res = pd.DataFrame(X_arr, columns=X_tr_ml.columns)
        y_res = pd.Series(y_arr, name=y_tr.name)
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(X_res, y_res)
    return clf


def inner_oof_scores(X_tr, y_tr, meta, target_col, method, inner_seed, inner_folds) -> np.ndarray:
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
    oof = np.zeros(len(y_tr), dtype=float)
    for fold_i, (itr, iva) in enumerate(skf.split(X_tr, y_tr)):
        Xi_tr, Xi_va, _ = fit_transform_ml_data(X_tr.iloc[itr], X_tr.iloc[iva], meta, target_col)
        clf = fit_rf_with_sampler(Xi_tr, y_tr.iloc[itr], method, inner_seed + fold_i)
        oof[iva] = get_positive_scores(clf, Xi_va)
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
        raise FileNotFoundError(f"Expected data at {csv_path}/{meta_path}. See data/README.md.")

    cv = get_cv_config()
    feature_sets = get_feature_sets()
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta_target_column(args.target)

    rows, outer_rows = [], []
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

            fold_auc = {}
            for method in SAMPLERS:
                try:
                    oof = inner_oof_scores(X_tr, y_tr, meta, target_col, method, cv["inner_seed"], cv["inner_folds"])
                    auc = roc_auc_score(y_tr, oof)
                    ap = average_precision_score(y_tr, oof)
                    th = youden_threshold(y_tr, oof)
                    pred = (oof >= th).astype(int)
                    m = compute_metrics(y_tr, pred, oof)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {flab} fold{fold} {method} failed: {exc}", flush=True)
                    traceback.print_exc()
                    auc, ap, m = np.nan, np.nan, {"Accuracy": np.nan, "F1-score": np.nan}
                fold_auc[method] = auc
                rows.append({"feature_set": flab, "outer_fold": fold, "sampler": method,
                             "inner_oof_auc": auc, "inner_oof_ap": ap, **m})
                print(f"  {flab} fold{fold} {method}: AUC={auc:.4f} AP={ap:.4f}", flush=True)

            winner = max(fold_auc, key=lambda s: (fold_auc[s] if fold_auc[s] == fold_auc[s] else -1))
            X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
            clf = fit_rf_with_sampler(X_tr_ml, y_tr, winner, cv["outer_seed"] + fold)
            scores_te = get_positive_scores(clf, X_te_ml)
            oof_for_th = inner_oof_scores(X_tr, y_tr, meta, target_col, winner, cv["inner_seed"], cv["inner_folds"])
            th = youden_threshold(y_tr, oof_for_th)
            pred_te = (scores_te >= th).astype(int)
            metrics = compute_metrics(y_te, pred_te, scores_te)
            auc_te = roc_auc_score(y_te, scores_te)
            outer_rows.append({"feature_set": flab, "outer_fold": fold, "selected_sampler": winner,
                                "outer_test_auc": auc_te, **metrics})
            print(f"  >> {flab} fold{fold} selected={winner} outer_test_auc={auc_te:.4f}", flush=True)
        print(f"{flab} done in {time.time()-t0:.0f}s", flush=True)

    inner_df = pd.DataFrame(rows)
    inner_df.to_csv(out / "stage2_inner_oof_long.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(out / "stage2_outer_nested_auc.csv", index=False)

    def ci(vals):
        n = len(vals); m = float(vals.mean()) if n else float("nan")
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        t = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        return m, m - t * se, m + t * se

    summary_rows = []
    for key, spec in feature_sets.items():
        flab = spec["label"]
        sub = inner_df[inner_df.feature_set == flab]
        for method in SAMPLERS:
            s = sub[sub.sampler == method]
            auc_m, auc_lo, auc_hi = ci(s["inner_oof_auc"].dropna().to_numpy())
            f1_m, f1_lo, f1_hi = ci(s["F1-score"].dropna().to_numpy())
            ap_m, ap_lo, ap_hi = ci(s["inner_oof_ap"].dropna().to_numpy())
            summary_rows.append({
                "Feature set": flab, "Sampler": method,
                "Inner-OOF AUC (mean)": auc_m, "Inner-OOF F1 (mean)": f1_m, "Inner-OOF AP (mean)": ap_m,
                "n_outer_folds": len(s),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "stage2_sampler_comparison.csv", index=False)

    for key, spec in feature_sets.items():
        flab = spec["label"]
        sub = summary[summary["Feature set"] == flab].copy()
        sub["AUC_rank"] = sub["Inner-OOF AUC (mean)"].rank(ascending=False)
        sub["F1_rank"] = sub["Inner-OOF F1 (mean)"].rank(ascending=False)
        sub["AP_rank"] = sub["Inner-OOF AP (mean)"].rank(ascending=False)
        sub["rank_sum"] = sub["AUC_rank"] + sub["F1_rank"] + sub["AP_rank"]
        best_auc = sub.sort_values("Inner-OOF AUC (mean)", ascending=False).iloc[0]["Sampler"]
        best_rank = sub.sort_values("rank_sum").iloc[0]["Sampler"]
        print(f"[{flab}] best by raw AUC: {best_auc} | best by AUC/F1/AP rank-consistency: {best_rank}")

    print("\n=== DONE ===")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
