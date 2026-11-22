#!/usr/bin/env python3
"""Stage 4 (Nested CV): Hyperparameter grid comparison. Random Forest + the
target's locked-in Stage 2 sampler + Stage 3 loss (read from configs/recipes.yaml).

Grid: criterion {gini, entropy} x min_samples_leaf {1,5,15} x max_depth {None,12}
x max_features {sqrt,0.5} = 24 configs. For a wider, Optuna-based search instead
of this fixed grid, see stage4_optuna_search.py.

Usage:
    python src/stage4_hyperparameter_comparison.py --target ODD --data-dir data --output-dir results/odd_nested
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_feature_sets, get_recipe  # noqa: E402
from stage1_model_comparison import meta_target_column, select_cols  # noqa: E402
from stage3_loss_comparison import fit_rf_with_loss, resample_xy  # noqa: E402
from utils.models import compute_metrics, get_positive_scores, youden_threshold  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

GRID = [
    {"criterion": crit, "min_samples_leaf": leaf, "max_depth": depth, "max_features": feat}
    for crit, leaf, depth, feat in itertools.product(("gini", "entropy"), (1, 5, 15), (None, 12), ("sqrt", 0.5))
]


def cfg_tag(cfg: dict) -> str:
    depth = "None" if cfg["max_depth"] is None else str(cfg["max_depth"])
    feat = cfg["max_features"] if isinstance(cfg["max_features"], str) else str(cfg["max_features"])
    return f"{cfg['criterion']}_leaf{cfg['min_samples_leaf']}_depth{depth}_feat{feat}"


def build_rf_cfg(cfg: dict, seed: int, class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500, criterion=cfg["criterion"], min_samples_leaf=cfg["min_samples_leaf"],
        max_depth=cfg["max_depth"], max_features=cfg["max_features"], class_weight=class_weight,
        random_state=seed, n_jobs=-1,
    )


def fit_cfg_with_loss(X_res, y_res, loss, cfg, seed, focal_gamma):
    """Same as stage3_loss_comparison.fit_rf_with_loss but with cfg-controlled hyperparameters."""
    if loss == "BCE":
        clf = build_rf_cfg(cfg, seed); clf.fit(X_res, y_res); return clf
    if loss == "Weighted BCE":
        y = np.asarray(y_res)
        w = float((y == 0).sum() / max(int((y == 1).sum()), 1))
        clf = build_rf_cfg(cfg, seed, class_weight={0: 1.0, 1: w}); clf.fit(X_res, y_res); return clf
    if loss == "Focal":
        from stage3_loss_comparison import focal_sample_weight
        warm = build_rf_cfg(cfg, seed); warm.fit(X_res, y_res)
        p_warm = get_positive_scores(warm, X_res)
        sw = focal_sample_weight(y_res, p_warm, focal_gamma)
        clf = build_rf_cfg(cfg, seed); clf.fit(X_res, y_res, sample_weight=sw); return clf
    raise ValueError(loss)


def inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, cfg, inner_seed, inner_folds, focal_gamma):
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
    oof = np.zeros(len(y_tr), dtype=float)
    for fold_i, (itr, iva) in enumerate(skf.split(X_tr, y_tr)):
        Xi_tr, Xi_va, _ = fit_transform_ml_data(X_tr.iloc[itr], X_tr.iloc[iva], meta, target_col)
        Xi_res, yi_res = resample_xy(Xi_tr, y_tr.iloc[itr], sampler_name, inner_seed + fold_i)
        clf = fit_cfg_with_loss(Xi_res, yi_res, loss, cfg, inner_seed + fold_i, focal_gamma)
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
    sampler_name, loss, focal_gamma = recipe["sampler"], recipe["loss"], recipe["focal_gamma"]
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta_target_column(args.target)
    print(f"[{args.target}] Stage 4 fixed sampler={sampler_name}, loss={loss}", flush=True)

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
            for cfg in GRID:
                tag = cfg_tag(cfg)
                try:
                    oof = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, cfg, cv["inner_seed"], cv["inner_folds"], focal_gamma)
                    auc = roc_auc_score(y_tr, oof)
                    ap = average_precision_score(y_tr, oof)
                    th = youden_threshold(y_tr, oof)
                    pred = (oof >= th).astype(int)
                    m = compute_metrics(y_tr, pred, oof)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {flab} fold{fold} {tag} failed: {exc}", flush=True)
                    traceback.print_exc()
                    auc, ap, m = np.nan, np.nan, {"Accuracy": np.nan, "F1-score": np.nan}
                fold_auc[tag] = auc
                rows.append({"feature_set": flab, "outer_fold": fold, "config": tag, **cfg,
                             "inner_oof_auc": auc, "inner_oof_ap": ap, **m})
            best_tag = max(fold_auc, key=lambda s: (fold_auc[s] if fold_auc[s] == fold_auc[s] else -1))
            best_cfg = next(c for c in GRID if cfg_tag(c) == best_tag)
            print(f"  {flab} fold{fold}: best={best_tag} AUC={fold_auc[best_tag]:.4f}", flush=True)

            X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
            X_res, y_res = resample_xy(X_tr_ml, y_tr, sampler_name, cv["outer_seed"] + fold)
            clf = fit_cfg_with_loss(X_res, y_res, loss, best_cfg, cv["outer_seed"] + fold, focal_gamma)
            scores_te = get_positive_scores(clf, X_te_ml)
            oof_for_th = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, best_cfg, cv["inner_seed"], cv["inner_folds"], focal_gamma)
            th = youden_threshold(y_tr, oof_for_th)
            pred_te = (scores_te >= th).astype(int)
            metrics = compute_metrics(y_te, pred_te, scores_te)
            auc_te = roc_auc_score(y_te, scores_te)
            outer_rows.append({"feature_set": flab, "outer_fold": fold, "selected_config": best_tag,
                                "outer_test_auc": auc_te, **metrics})
            print(f"  >> {flab} fold{fold} selected={best_tag} outer_test_auc={auc_te:.4f}", flush=True)
        print(f"{flab} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(out / "stage4_inner_oof_long.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(out / "stage4_outer_nested_auc.csv", index=False)
    print("\n=== DONE ===")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
