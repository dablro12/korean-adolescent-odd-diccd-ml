#!/usr/bin/env python3
"""Stage 4' (Nested CV): Optuna hyperparameter search, replacing the manual
24-config grid in stage4_hyperparameter_comparison.py with a wider, partly
continuous search (TPE sampler). Same outer/inner CV scaffold; only the
candidate-generation method differs. Combined feature set only (the primary
reported line) to keep runtime reasonable; this is a robustness check, not a
replacement for the main grid comparison.

For both targets in this project, the wider search did not materially beat the
manual grid (ODD: tied AUC, worse F1; DICCD: +0.005 AUC with no single stable
config across folds) -- evidence the manual grid was already close to optimal,
not that it must be replaced. See README.md.

Usage:
    python src/stage4_optuna_search.py --target ODD --data-dir data --output-dir results/odd_nested --n-trials 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_recipe  # noqa: E402
from stage1_model_comparison import meta_target_column, select_cols  # noqa: E402
from stage3_loss_comparison import resample_xy  # noqa: E402
from stage4_hyperparameter_comparison import fit_cfg_with_loss  # noqa: E402
from utils.models import compute_metrics, get_positive_scores, youden_threshold  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

GROUP_MAP_ALL = ["other_features", "adolescent_features", "adult_features"]


def sample_cfg(trial: "optuna.Trial") -> dict:
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
    depth_choice = trial.suggest_categorical("depth_choice", ["none", "int"])
    max_depth = None if depth_choice == "none" else trial.suggest_int("max_depth_val", 3, 30)
    feat_choice = trial.suggest_categorical("feat_choice", ["sqrt", "log2", "float"])
    max_features = trial.suggest_float("max_features_val", 0.1, 1.0) if feat_choice == "float" else feat_choice
    return {"criterion": criterion, "min_samples_leaf": min_samples_leaf, "max_depth": max_depth, "max_features": max_features}


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
    ap.add_argument("--n-trials", default=40, type=int)
    args = ap.parse_args()

    out = args.output_dir or (REPO_ROOT / "results" / f"{args.target.lower()}_nested")
    out.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = data_paths(args.target, args.data_dir)
    if not csv_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Expected data at {csv_path}/{meta_path}. See data/README.md.")

    cv = get_cv_config()
    recipe = get_recipe(args.target)
    sampler_name, loss, focal_gamma = recipe["sampler"], recipe["loss"], recipe["focal_gamma"]
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta_target_column(args.target)
    cols = select_cols(meta, GROUP_MAP_ALL, target_col)
    Xf, y_full = df[cols], df[target_col]

    outer_cv = StratifiedKFold(n_splits=cv["outer_folds"], shuffle=True, random_state=cv["outer_seed"])
    outer_rows = []
    t0 = time.time()

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(Xf, y_full), start=1):
        X_tr, X_te = Xf.iloc[tr_idx], Xf.iloc[te_idx]
        y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

        def objective(trial: "optuna.Trial") -> float:
            cfg = sample_cfg(trial)
            try:
                oof = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, cfg, cv["inner_seed"], cv["inner_folds"], focal_gamma)
                return roc_auc_score(y_tr, oof)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! fold{fold} trial{trial.number} failed: {exc}", flush=True)
                traceback.print_exc()
                return 0.0

        sampler = optuna.samplers.TPESampler(seed=cv["outer_seed"] + fold)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        p = study.best_params
        best_cfg = {
            "criterion": p["criterion"], "min_samples_leaf": p["min_samples_leaf"],
            "max_depth": None if p["depth_choice"] == "none" else p["max_depth_val"],
            "max_features": p["max_features_val"] if p["feat_choice"] == "float" else p["feat_choice"],
        }
        print(f"  fold{fold} best inner-OOF AUC={study.best_value:.4f} cfg={best_cfg}", flush=True)

        X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
        X_res, y_res = resample_xy(X_tr_ml, y_tr, sampler_name, cv["outer_seed"] + fold)
        clf = fit_cfg_with_loss(X_res, y_res, loss, best_cfg, cv["outer_seed"] + fold, focal_gamma)
        scores_te = get_positive_scores(clf, X_te_ml)
        oof_for_th = inner_oof_scores(X_tr, y_tr, meta, target_col, sampler_name, loss, best_cfg, cv["inner_seed"], cv["inner_folds"], focal_gamma)
        th = youden_threshold(y_tr, oof_for_th)
        pred_te = (scores_te >= th).astype(int)
        metrics = compute_metrics(y_te, pred_te, scores_te)
        auc_te = roc_auc_score(y_te, scores_te)
        outer_rows.append({"outer_fold": fold, "best_cfg": json.dumps(best_cfg),
                            "inner_oof_auc": study.best_value, "outer_test_auc": auc_te, **metrics})
        print(f"  >> fold{fold} outer_test_auc={auc_te:.4f}", flush=True)

    print(f"done in {time.time()-t0:.0f}s", flush=True)
    outer_df = pd.DataFrame(outer_rows)
    outer_df.to_csv(out / "stage4_optuna_outer_nested_auc.csv", index=False)

    def ci(vals):
        n = len(vals); m = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        t = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        return m, m - t * se, m + t * se

    print("\n=== DONE ===")
    for metric in ["Accuracy", "F1-score", "AUC Score"]:
        if metric in outer_df:
            m, lo, hi = ci(outer_df[metric].to_numpy())
            print(f"{metric}: {m:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print("Best configs per fold:", outer_df["best_cfg"].tolist())


if __name__ == "__main__":
    main()
