#!/usr/bin/env python3
"""Stage 3 (Nested CV): Loss comparison. Random Forest + the target's locked-in
Stage 2 sampler (read from configs/recipes.yaml: ADASYN for ODD, None for DICCD).

Usage:
    python src/stage3_loss_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
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
from imblearn.over_sampling import ADASYN
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_feature_sets, get_recipe  # noqa: E402
from stage1_model_comparison import meta_target_column, select_cols  # noqa: E402
from utils.models import compute_metrics, get_positive_scores, youden_threshold  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

POS, NEG = 1, 0
LOSSES = ("BCE", "Weighted BCE", "Focal")


def safe_k(n_pos: int) -> int:
    return max(1, min(5, n_pos - 1))


def resample_xy(X_ml: pd.DataFrame, y: pd.Series, sampler_name: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Applies the target's locked-in Stage 2 sampler (or passes through unchanged for 'None')."""
    if sampler_name in (None, "None"):
        return X_ml, y
    if sampler_name != "ADASYN":
        raise NotImplementedError(f"Only ADASYN/None are locked in for any target; got {sampler_name!r}")
    n_pos, n_neg = int((y == POS).sum()), int((y == NEG).sum())
    if n_pos >= n_neg:
        raise RuntimeError(f"positive class not minority: pos={n_pos} neg={n_neg}")
    k = safe_k(n_pos)
    sampler = ADASYN(sampling_strategy={POS: n_neg}, random_state=seed, n_neighbors=k)
    X_arr, y_arr = sampler.fit_resample(X_ml.to_numpy(), y.to_numpy())
    return pd.DataFrame(X_arr, columns=X_ml.columns), pd.Series(y_arr, name=y.name)


def pos_weight(y) -> float:
    y = np.asarray(y)
    n_pos, n_neg = int((y == POS).sum()), int((y == NEG).sum())
    return float(n_neg / max(n_pos, 1))


def build_rf(seed: int, class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1, class_weight=class_weight)


def focal_sample_weight(y, p_pos: np.ndarray, gamma: float) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos, n_neg = max(int((y == POS).sum()), 1), max(int((y == NEG).sum()), 1)
    alpha_pos, alpha_neg = n_neg / n, n_pos / n
    p = np.clip(np.asarray(p_pos, float), 1e-6, 1.0 - 1e-6)
    p_t = np.where(y == POS, p, 1.0 - p)
    alpha_t = np.where(y == POS, alpha_pos, alpha_neg)
    sw = alpha_t * np.power(1.0 - p_t, gamma)
    sw = np.clip(sw, 1e-3, None)
    return sw * (n / sw.sum())


def fit_rf_with_loss(X_res: pd.DataFrame, y_res: pd.Series, loss: str, seed: int, focal_gamma: float) -> RandomForestClassifier:
    if loss == "BCE":
        clf = build_rf(seed); clf.fit(X_res, y_res); return clf
    if loss == "Weighted BCE":
        w = pos_weight(y_res)
        clf = build_rf(seed, class_weight={NEG: 1.0, POS: w}); clf.fit(X_res, y_res); return clf
    if loss == "Focal":
        warm = build_rf(seed); warm.fit(X_res, y_res)
        p_warm = get_positive_scores(warm, X_res)
        sw = focal_sample_weight(y_res, p_warm, focal_gamma)
        clf = build_rf(seed); clf.fit(X_res, y_res, sample_weight=sw); return clf
    raise ValueError(loss)


def inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, inner_seed, inner_folds, focal_gamma) -> np.ndarray:
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
    oof = np.zeros(len(y_tr), dtype=float)
    for fold_i, (itr, iva) in enumerate(skf.split(X_tr, y_tr)):
        Xi_tr, Xi_va, _ = fit_transform_ml_data(X_tr.iloc[itr], X_tr.iloc[iva], meta, target_col)
        Xi_res, yi_res = resample_xy(Xi_tr, y_tr.iloc[itr], sampler_name, inner_seed + fold_i)
        clf = fit_rf_with_loss(Xi_res, yi_res, loss, inner_seed + fold_i, focal_gamma)
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
    recipe = get_recipe(args.target)
    sampler_name = recipe["sampler"]
    focal_gamma = recipe["focal_gamma"]
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta_target_column(args.target)
    print(f"[{args.target}] Stage 3 fixed sampler (from recipe): {sampler_name}", flush=True)

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
            for loss in LOSSES:
                try:
                    oof = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, cv["inner_seed"], cv["inner_folds"], focal_gamma)
                    auc = roc_auc_score(y_tr, oof)
                    ap = average_precision_score(y_tr, oof)
                    th = youden_threshold(y_tr, oof)
                    pred = (oof >= th).astype(int)
                    m = compute_metrics(y_tr, pred, oof)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {flab} fold{fold} {loss} failed: {exc}", flush=True)
                    traceback.print_exc()
                    auc, ap, m = np.nan, np.nan, {"Accuracy": np.nan, "F1-score": np.nan}
                fold_auc[loss] = auc
                rows.append({"feature_set": flab, "outer_fold": fold, "loss": loss,
                             "inner_oof_auc": auc, "inner_oof_ap": ap, **m})
                print(f"  {flab} fold{fold} {loss}: AUC={auc:.4f} AP={ap:.4f}", flush=True)

            winner = max(fold_auc, key=lambda s: (fold_auc[s] if fold_auc[s] == fold_auc[s] else -1))
            X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
            X_res, y_res = resample_xy(X_tr_ml, y_tr, sampler_name, cv["outer_seed"] + fold)
            clf = fit_rf_with_loss(X_res, y_res, winner, cv["outer_seed"] + fold, focal_gamma)
            scores_te = get_positive_scores(clf, X_te_ml)
            oof_for_th = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, winner, cv["inner_seed"], cv["inner_folds"], focal_gamma)
            th = youden_threshold(y_tr, oof_for_th)
            pred_te = (scores_te >= th).astype(int)
            metrics = compute_metrics(y_te, pred_te, scores_te)
            auc_te = roc_auc_score(y_te, scores_te)
            outer_rows.append({"feature_set": flab, "outer_fold": fold, "selected_loss": winner,
                                "outer_test_auc": auc_te, **metrics})
            print(f"  >> {flab} fold{fold} selected={winner} outer_test_auc={auc_te:.4f}", flush=True)
        print(f"{flab} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(out / "stage3_inner_oof_long.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(out / "stage3_outer_nested_auc.csv", index=False)
    print("\n=== DONE ===")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
