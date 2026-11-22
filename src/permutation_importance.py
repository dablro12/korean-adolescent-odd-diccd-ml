#!/usr/bin/env python3
"""Permutation importance (delta-AUC), computed on the held-out outer-test fold
only (never on train), using the target's locked-in recipe (configs/recipes.yaml).
5-fold StratifiedKFold(seed=42), n_repeats=8, scoring="roc_auc" -- the same
convention as the original ADHD-included analysis's importance_seed42().

Usage:
    python src/permutation_importance.py --target ODD --feature-set all --data-dir data --output-dir results/odd_nested
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recipe import data_paths, get_cv_config, get_feature_sets, get_recipe  # noqa: E402
from stage1_model_comparison import meta_target_column, select_cols  # noqa: E402
from stage3_loss_comparison import fit_rf_with_loss, resample_xy  # noqa: E402
from utils.exp15_publication_labels import display_feature, load_labels  # noqa: E402
from utils.preprocess import fit_transform_ml_data  # noqa: E402

LABELS = load_labels()


def pretty(name: str) -> str:
    raw = name.split("__", 1)[-1] if "__" in name else name
    return display_feature(raw, LABELS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["ODD", "DICCD"])
    ap.add_argument("--feature-set", default="all", choices=["all", "adolescent", "parent"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--n-repeats", default=8, type=int)
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
    spec = feature_sets[args.feature_set]
    flab = spec["label"]
    cols = select_cols(meta, spec["groups"], target_col)
    Xf, y_full = df[cols], df[target_col]

    skf = StratifiedKFold(n_splits=cv["outer_folds"], shuffle=True, random_state=cv["outer_seed"])
    fold_series = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(Xf, y_full), start=1):
        X_tr, X_te = Xf.iloc[tr_idx], Xf.iloc[te_idx]
        y_tr, y_te = y_full.iloc[tr_idx].to_numpy(dtype=int), y_full.iloc[te_idx].to_numpy(dtype=int)
        X_tr_ml, X_te_ml, _ = fit_transform_ml_data(X_tr, X_te, meta, target_col)
        X_res, y_res = resample_xy(X_tr_ml, pd.Series(y_tr, index=X_tr_ml.index), sampler_name, cv["outer_seed"] + fold)
        clf = fit_rf_with_loss(X_res, y_res, loss, cv["outer_seed"] + fold, focal_gamma)

        # Permutation importance on the held-out outer-test fold ONLY -- never on train.
        pi = permutation_importance(clf, X_te_ml, y_te, scoring="roc_auc", n_repeats=args.n_repeats,
                                     random_state=cv["outer_seed"], n_jobs=-1)
        fold_series.append(pd.Series(pi.importances_mean, index=list(X_te_ml.columns), name=f"fold{fold}"))
        print(f"[{args.target}][{flab}] fold{fold} permutation importance done", flush=True)

    mat = pd.concat(fold_series, axis=1).fillna(0.0)
    df_out = pd.DataFrame({
        "feature": mat.index,
        "label": [pretty(c) for c in mat.index],
        "mean_delta_auc": mat.mean(axis=1).values,
        "sd_delta_auc": mat.std(axis=1, ddof=1).values,
    }).sort_values("mean_delta_auc", ascending=False).reset_index(drop=True)

    assert not df_out["label"].str.contains("adhd", case=False).any(), "ADHD must not appear -- check metadata"

    path = out / f"permutation_{args.target.lower()}_{args.feature_set}.csv"
    df_out.to_csv(path, index=False)
    print("\n=== DONE ===")
    print(df_out.head(10).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
