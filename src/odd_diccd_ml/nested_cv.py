"""Nested 5x5 stratified cross-validation: 4-stage sequential search, Optuna check, and locked-recipe evaluation."""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from .config import FEATURE_SET_LABEL, TARGET_LABEL, Config
from .data import prepare_xy
from .models import (
    build_stage1_pipeline,
    compute_metrics,
    fit_rf_with_loss,
    get_positive_scores,
    rank_consistent_winner,
    resample_xy,
    youden_threshold,
)
from .preprocess import fit_transform_fold

BASE_RF_HPARAMS = {"criterion": "gini", "min_samples_leaf": 1, "max_depth": None, "max_features": "sqrt"}
INTERP_GRID = np.linspace(0.0, 1.0, 201)


def cfg_tag(hp: dict) -> str:
    d = "None" if hp["max_depth"] is None else str(hp["max_depth"])
    return f"{hp['criterion']}_leaf{hp['min_samples_leaf']}_depth{d}_feat{hp['max_features']}"


def inner_oof_rf(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cfg: Config,
    sampler_name: str,
    loss: str,
    hparams: dict,
    focal_gamma: Optional[float],
    n_estimators: int = 500,
) -> np.ndarray:
    """Compute 5-fold inner out-of-fold positive-class probabilities strictly inside X_tr."""
    skf = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.inner_seed)
    oof = np.zeros(len(y_tr), dtype=float)
    for fold_i, (i_tr, i_val) in enumerate(skf.split(X_tr, y_tr)):
        X_itr, X_ival, _ = fit_transform_fold(X_tr.iloc[i_tr], X_tr.iloc[i_val], cfg)
        y_itr = y_tr.iloc[i_tr]
        X_res, y_res = resample_xy(X_itr, y_itr, sampler_name, cfg.inner_seed + fold_i)
        clf = fit_rf_with_loss(
            X_res, y_res, loss, hparams, cfg.inner_seed + fold_i,
            focal_gamma=focal_gamma, n_estimators=n_estimators,
        )
        oof[i_val] = get_positive_scores(clf, X_ival)
    return oof


def _interp_curves(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, d in scores_df.groupby("outer_fold"):
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        rows.append(pd.DataFrame({
            "kind": "roc", "outer_fold": k, "x": INTERP_GRID,
            "y": np.interp(INTERP_GRID, fpr, tpr, left=0.0, right=1.0),
        }))
        prec, rec, _ = precision_recall_curve(d["y_true"], d["y_score"])
        order = np.argsort(rec)
        r_s, p_s = rec[order], prec[order]
        rows.append(pd.DataFrame({
            "kind": "pr", "outer_fold": k, "x": INTERP_GRID,
            "y": np.interp(INTERP_GRID, r_s, p_s, left=p_s[0], right=p_s[-1]),
        }))
    return pd.concat(rows, ignore_index=True)


def run_locked_cv(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    feature_sets: Optional[Iterable[str]] = None,
    n_estimators_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Evaluate the target's locked-in recipe across feature sets under nested 5x5 CV."""
    cohort = cohort.lower()
    rec = cfg.recipe(cohort)
    n_est = int(n_estimators_override or rec["n_estimators"])
    out = cfg.cohort_out_dir(cohort) / "locked_cv"
    out.mkdir(parents=True, exist_ok=True)

    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.outer_seed)
    fold_rows, summary_rows, cm_rows, curve_parts, score_parts = [], [], [], [], []

    for fs in (feature_sets or cfg.feature_sets):
        X_all, y_all = prepare_xy(df, cfg, cohort, fs)
        fs_fold_rows, fs_scores = [], []
        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

            oof = inner_oof_rf(
                X_tr, y_tr, cfg, rec["sampler"], rec["loss"],
                rec["hyperparameters"], rec["focal_gamma"], n_estimators=n_est,
            )
            th = youden_threshold(y_tr, oof)

            X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
            X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
            clf = fit_rf_with_loss(
                X_res, y_res, rec["loss"], rec["hyperparameters"],
                cfg.outer_seed + fold, focal_gamma=rec["focal_gamma"], n_estimators=n_est,
            )
            scores_te = get_positive_scores(clf, X_te_ml)
            pred_te = (scores_te >= th).astype(int)
            m = compute_metrics(y_te, pred_te, scores_te)

            row = {
                "cohort": cohort,
                "target_label": TARGET_LABEL[cohort],
                "feature_set": fs,
                "feature_set_label": FEATURE_SET_LABEL[fs],
                "outer_fold": fold,
                "recipe": f"RF({n_est}) | {rec['sampler']} | {rec['loss']} | {cfg_tag(rec['hyperparameters'])}",
                "youden_threshold": th,
                **m,
            }
            fold_rows.append(row)
            fs_fold_rows.append(row)
            fs_scores.append(pd.DataFrame({
                "cohort": cohort, "feature_set": fs, "outer_fold": fold,
                "row_idx": te_idx, "y_true": y_te.to_numpy(), "y_score": scores_te, "y_pred": pred_te,
            }))
            if verbose:
                print(
                    f"[{TARGET_LABEL[cohort]}][{FEATURE_SET_LABEL[fs]}] fold{fold} "
                    f"AUC={m['auc']:.4f} F1={m['f1']:.4f} AP={m['average_precision']:.4f}",
                    flush=True,
                )

        fdf = pd.DataFrame(fs_fold_rows)
        sdf = pd.concat(fs_scores, ignore_index=True)
        score_parts.append(sdf)
        curve_parts.append(_interp_curves(sdf).assign(cohort=cohort, feature_set=fs))

        tot = fdf[["tn", "fp", "fn", "tp"]].sum()
        cm_rows.append({
            "cohort": cohort, "feature_set": fs, "feature_set_label": FEATURE_SET_LABEL[fs],
            "tn": int(tot["tn"]), "fp": int(tot["fp"]), "fn": int(tot["fn"]), "tp": int(tot["tp"]),
            "sensitivity": float(tot["tp"] / max(tot["tp"] + tot["fn"], 1)),
            "specificity": float(tot["tn"] / max(tot["tn"] + tot["fp"], 1)),
        })

        srow = {
            "cohort": cohort,
            "target_label": TARGET_LABEL[cohort],
            "feature_set": fs,
            "feature_set_label": FEATURE_SET_LABEL[fs],
            "n": int(len(y_all)),
            "n_cases": int(y_all.sum()),
            "case_proportion": float(y_all.mean()),
            "recipe": fdf.iloc[0]["recipe"],
        }
        for metric in ("auc", "average_precision", "accuracy", "precision", "recall", "f1", "specificity"):
            srow[f"{metric}_mean"] = float(fdf[metric].mean())
            srow[f"{metric}_sd"] = float(fdf[metric].std(ddof=1))
        summary_rows.append(srow)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curve_parts, ignore_index=True)
    cm_df = pd.DataFrame(cm_rows)
    scores_df = pd.concat(score_parts, ignore_index=True)

    fold_df.to_csv(out / "outer_fold_metrics.csv", index=False)
    summary_df.to_csv(out / "performance_summary.csv", index=False)
    curves_df.round(6).to_csv(out / "roc_pr_curves_interpolated.csv", index=False)
    cm_df.to_csv(out / "confusion_pooled.csv", index=False)
    scores_df.to_csv(out / "outer_test_scores.csv", index=False)
    return {
        "outer_fold_metrics": fold_df,
        "performance_summary": summary_df,
        "roc_pr_curves": curves_df,
        "confusion_pooled": cm_df,
        "outer_test_scores": scores_df,
    }


def run_stage_search(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    stages: Iterable[int] = (1, 2, 3, 4),
    feature_sets: Optional[Iterable[str]] = None,
    run_optuna: bool = False,
    verbose: bool = True,
) -> None:
    """Run the 4-stage sequential nested-CV search (Model -> Sampler -> Loss -> Hyperparameter)."""
    cohort = cohort.lower()
    rec = cfg.recipe(cohort)
    out = cfg.cohort_out_dir(cohort) / "stage_search"
    out.mkdir(parents=True, exist_ok=True)
    fsets = list(feature_sets or cfg.feature_sets)
    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.outer_seed)
    stage_set = set(stages)

    # Stage 1: 8-classifier comparison (no resampling)
    if 1 in stage_set:
        rows, outer_rows = [], []
        for fs in fsets:
            X_all, y_all = prepare_xy(df, cfg, cohort, fs)
            for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
                f_metrics, f_oof = {}, {}
                skf = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.inner_seed)
                for mname, scale in cfg.models.items():
                    oof = np.zeros(len(y_tr), dtype=float)
                    for fold_i, (i_tr, i_val) in enumerate(skf.split(X_tr, y_tr)):
                        X_itr, X_ival, _ = fit_transform_fold(X_tr.iloc[i_tr], X_tr.iloc[i_val], cfg)
                        pipe = build_stage1_pipeline(mname, scale, cfg.inner_seed + fold_i)
                        pipe.fit(X_itr, y_tr.iloc[i_tr])
                        oof[i_val] = get_positive_scores(pipe, X_ival)
                    th = youden_threshold(y_tr, oof)
                    m = compute_metrics(y_tr, (oof >= th).astype(int), oof)
                    f_metrics[mname] = (m["auc"], m["f1"], m["average_precision"])
                    f_oof[mname] = oof
                    rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "model": mname, **m})
                winner, _ = rank_consistent_winner(f_metrics)
                X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
                pipe = build_stage1_pipeline(winner, cfg.models[winner], cfg.outer_seed + fold)
                pipe.fit(X_tr_ml, y_tr)
                scores_te = get_positive_scores(pipe, X_te_ml)
                th = youden_threshold(y_tr, f_oof[winner])
                m_te = compute_metrics(y_te, (scores_te >= th).astype(int), scores_te)
                outer_rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "selected_model": winner, **m_te})
                if verbose:
                    print(f"[Stage 1][{cohort.upper()}][{fs}] fold{fold} winner={winner} outer_AUC={m_te['auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "stage1_inner_oof.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(out / "stage1_outer_test.csv", index=False)

    # Stage 2: 8-sampler comparison (Random Forest fixed)
    if 2 in stage_set:
        rows, outer_rows = [], []
        for fs in fsets:
            X_all, y_all = prepare_xy(df, cfg, cohort, fs)
            for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
                f_metrics, f_oof = {}, {}
                for sname in cfg.samplers:
                    oof = inner_oof_rf(X_tr, y_tr, cfg, sname, "BCE", BASE_RF_HPARAMS, None, n_estimators=500)
                    th = youden_threshold(y_tr, oof)
                    m = compute_metrics(y_tr, (oof >= th).astype(int), oof)
                    f_metrics[sname] = (m["auc"], m["f1"], m["average_precision"])
                    f_oof[sname] = oof
                    rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "sampler": sname, **m})
                winner, _ = rank_consistent_winner(f_metrics, exclude=cfg.unstable_cluster_samplers)
                X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
                X_res, y_res = resample_xy(X_tr_ml, y_tr, winner, cfg.outer_seed + fold)
                clf = fit_rf_with_loss(X_res, y_res, "BCE", BASE_RF_HPARAMS, cfg.outer_seed + fold, None, n_estimators=500)
                scores_te = get_positive_scores(clf, X_te_ml)
                th = youden_threshold(y_tr, f_oof[winner])
                m_te = compute_metrics(y_te, (scores_te >= th).astype(int), scores_te)
                outer_rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "selected_sampler": winner, **m_te})
                if verbose:
                    print(f"[Stage 2][{cohort.upper()}][{fs}] fold{fold} winner={winner} outer_AUC={m_te['auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "stage2_inner_oof.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(out / "stage2_outer_test.csv", index=False)

    # Stage 3: Loss comparison (locked sampler fixed)
    if 3 in stage_set:
        rows, outer_rows = [], []
        for fs in fsets:
            X_all, y_all = prepare_xy(df, cfg, cohort, fs)
            for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
                f_metrics, f_oof = {}, {}
                for lname in cfg.losses:
                    oof = inner_oof_rf(X_tr, y_tr, cfg, rec["sampler"], lname, BASE_RF_HPARAMS, cfg.focal_gamma, n_estimators=500)
                    th = youden_threshold(y_tr, oof)
                    m = compute_metrics(y_tr, (oof >= th).astype(int), oof)
                    f_metrics[lname] = (m["auc"], m["f1"], m["average_precision"])
                    f_oof[lname] = oof
                    rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "loss": lname, **m})
                winner, _ = rank_consistent_winner(f_metrics)
                X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
                X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
                clf = fit_rf_with_loss(X_res, y_res, winner, BASE_RF_HPARAMS, cfg.outer_seed + fold, cfg.focal_gamma, n_estimators=500)
                scores_te = get_positive_scores(clf, X_te_ml)
                th = youden_threshold(y_tr, f_oof[winner])
                m_te = compute_metrics(y_te, (scores_te >= th).astype(int), scores_te)
                outer_rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "selected_loss": winner, **m_te})
                if verbose:
                    print(f"[Stage 3][{cohort.upper()}][{fs}] fold{fold} winner={winner} outer_AUC={m_te['auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "stage3_inner_oof.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(out / "stage3_outer_test.csv", index=False)

    # Stage 4: 24-candidate hyperparameter grid (locked sampler + loss fixed)
    if 4 in stage_set:
        g = cfg.hparam_grid
        grid: List[dict] = [
            {"criterion": cr, "min_samples_leaf": lf, "max_depth": dp, "max_features": mf}
            for cr, lf, dp, mf in itertools.product(g["criterion"], g["min_samples_leaf"], g["max_depth"], g["max_features"])
        ]
        rows, outer_rows = [], []
        for fs in fsets:
            X_all, y_all = prepare_xy(df, cfg, cohort, fs)
            for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
                f_metrics, f_oof = {}, {}
                for hp in grid:
                    tag = cfg_tag(hp)
                    oof = inner_oof_rf(X_tr, y_tr, cfg, rec["sampler"], rec["loss"], hp, rec["focal_gamma"], n_estimators=int(rec["n_estimators"]))
                    th = youden_threshold(y_tr, oof)
                    m = compute_metrics(y_tr, (oof >= th).astype(int), oof)
                    f_metrics[tag] = (m["auc"], m["f1"], m["average_precision"])
                    f_oof[tag] = oof
                    rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "config": tag, **hp, **m})
                winner, _ = rank_consistent_winner(f_metrics)
                best_hp = next(hp for hp in grid if cfg_tag(hp) == winner)
                X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
                X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
                clf = fit_rf_with_loss(X_res, y_res, rec["loss"], best_hp, cfg.outer_seed + fold, rec["focal_gamma"], n_estimators=int(rec["n_estimators"]))
                scores_te = get_positive_scores(clf, X_te_ml)
                th = youden_threshold(y_tr, f_oof[winner])
                m_te = compute_metrics(y_te, (scores_te >= th).astype(int), scores_te)
                outer_rows.append({"feature_set": FEATURE_SET_LABEL[fs], "outer_fold": fold, "selected_config": winner, **m_te})
                if verbose:
                    print(f"[Stage 4][{cohort.upper()}][{fs}] fold{fold} winner={winner} outer_AUC={m_te['auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "stage4_inner_oof.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(out / "stage4_outer_test.csv", index=False)

    if run_optuna:
        run_optuna_check(cfg, df, cohort, n_trials=cfg.optuna_trials, verbose=verbose)


def run_optuna_check(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    n_trials: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Wider Optuna TPE hyperparameter check on the Combined feature set."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cohort = cohort.lower()
    rec = cfg.recipe(cohort)
    trials = int(n_trials or cfg.optuna_trials)
    out = cfg.cohort_out_dir(cohort) / "stage_search"
    out.mkdir(parents=True, exist_ok=True)

    X_all, y_all = prepare_xy(df, cfg, cohort, "combined")
    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.outer_seed)
    outer_rows = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

        def objective(trial: optuna.Trial) -> float:
            use_none = trial.suggest_categorical("max_depth_is_none", [True, False])
            depth = None if use_none else trial.suggest_int("max_depth_int", 3, 30)
            mf_kind = trial.suggest_categorical("max_features_kind", ["sqrt", "log2", "frac"])
            mf = trial.suggest_float("max_features_frac", 0.1, 0.7) if mf_kind == "frac" else mf_kind
            hp = {
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
                "max_depth": depth,
                "max_features": mf,
            }
            oof = inner_oof_rf(X_tr, y_tr, cfg, rec["sampler"], rec["loss"], hp, rec["focal_gamma"],
                               n_estimators=trial.suggest_categorical("n_estimators", [300, 500, 800]))
            trial.set_user_attr("oof", oof)
            trial.set_user_attr("hp", hp)
            return float(roc_auc_score(y_tr, oof))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed + fold))
        study.optimize(objective, n_trials=trials)
        best = study.best_trial
        best_hp = best.user_attrs["hp"]
        best_n = int(best.params["n_estimators"])
        oof_best = best.user_attrs["oof"]

        X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
        X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
        clf = fit_rf_with_loss(X_res, y_res, rec["loss"], best_hp, cfg.outer_seed + fold, rec["focal_gamma"], n_estimators=best_n)
        scores_te = get_positive_scores(clf, X_te_ml)
        th = youden_threshold(y_tr, oof_best)
        m_te = compute_metrics(y_te, (scores_te >= th).astype(int), scores_te)
        outer_rows.append({"outer_fold": fold, "best_inner_auc": float(best.value), "n_estimators": best_n, **best_hp, **m_te})
        if verbose:
            print(f"[Optuna][{cohort.upper()}] fold{fold} outer_AUC={m_te['auc']:.4f} F1={m_te['f1']:.4f}", flush=True)

    res_df = pd.DataFrame(outer_rows)
    res_df.to_csv(out / "stage4_optuna_outer_test.csv", index=False)
    return res_df
