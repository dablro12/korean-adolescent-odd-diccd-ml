"""Supplementary robustness checks: Platt/isotonic calibration, paired DeLong/bootstrap tests, and no-perpetration sensitivity."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import FEATURE_SET_LABEL, TARGET_LABEL, Config
from .data import prepare_xy
from .models import compute_metrics, fit_rf_with_loss, get_positive_scores, resample_xy, youden_threshold
from .nested_cv import inner_oof_rf
from .preprocess import fit_transform_fold


def _ece_10bin(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def _delong_two_roc(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    """Fast Sun & Xu (2014) paired DeLong z-test for two correlated ROC curves."""
    y = np.asarray(y_true, dtype=int)
    pos = y == 1
    neg = y == 0
    m, n = int(pos.sum()), int(neg.sum())

    def _structural(scores: np.ndarray):
        sp, sn = scores[pos], scores[neg]
        cmp = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :]).astype(float)
        auc = float(cmp.mean())
        v10 = cmp.mean(axis=1)
        v01 = cmp.mean(axis=0)
        return auc, v10, v01

    auc1, v10_1, v01_1 = _structural(np.asarray(p1))
    auc2, v10_2, v01_2 = _structural(np.asarray(p2))
    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)
    s = s10 / m + s01 / n
    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var_diff <= 1e-15:
        return float(auc1 - auc2), 1.0
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(auc1 - auc2), p


def run_robustness(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    n_estimators_override: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run inner-fold Platt/isotonic calibration, paired DeLong/bootstrap tests, and no-perpetration sensitivity."""
    cohort = cohort.lower()
    rec = cfg.recipe(cohort)
    n_est = int(n_estimators_override or rec["n_estimators"])
    out = cfg.cohort_out_dir(cohort) / "robustness"
    out.mkdir(parents=True, exist_ok=True)

    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.outer_seed)
    pooled_scores: Dict[str, pd.DataFrame] = {}
    calib_rows = []

    for fs in cfg.feature_sets:
        X_all, y_all = prepare_xy(df, cfg, cohort, fs)
        parts = []
        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

            oof = inner_oof_rf(X_tr, y_tr, cfg, rec["sampler"], rec["loss"], rec["hyperparameters"], rec["focal_gamma"], n_estimators=n_est)
            X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
            X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
            clf = fit_rf_with_loss(X_res, y_res, rec["loss"], rec["hyperparameters"], cfg.outer_seed + fold, rec["focal_gamma"], n_estimators=n_est)
            raw_te = get_positive_scores(clf, X_te_ml)

            # Platt scaling fit strictly on inner OOF log-odds
            eps = 1e-6
            logit_oof = np.log(np.clip(oof, eps, 1 - eps) / (1 - np.clip(oof, eps, 1 - eps))).reshape(-1, 1)
            logit_te = np.log(np.clip(raw_te, eps, 1 - eps) / (1 - np.clip(raw_te, eps, 1 - eps))).reshape(-1, 1)
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit_oof, y_tr.to_numpy())
            platt_te = platt.predict_proba(logit_te)[:, 1]

            # Isotonic regression fit strictly on inner OOF probabilities
            iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_tr.to_numpy())
            iso_te = iso.transform(raw_te)

            parts.append(pd.DataFrame({
                "row_idx": te_idx, "outer_fold": fold, "y_true": y_te.to_numpy(),
                "raw": raw_te, "platt": platt_te, "isotonic": iso_te,
            }))

        p_df = pd.concat(parts, ignore_index=True).sort_values("row_idx").reset_index(drop=True)
        pooled_scores[fs] = p_df
        yt = p_df["y_true"].to_numpy()
        for scheme in ("raw", "platt", "isotonic"):
            pr = p_df[scheme].to_numpy()
            calib_rows.append({
                "cohort": cohort,
                "target_label": TARGET_LABEL[cohort],
                "feature_set": FEATURE_SET_LABEL[fs],
                "scheme": scheme,
                "pooled_auc": float(roc_auc_score(yt, pr)),
                "brier_score": float(brier_score_loss(yt, pr)),
                "ece_10bin": _ece_10bin(yt, pr),
                "mean_pred_risk": float(pr.mean()),
                "observed_prevalence": float(yt.mean()),
            })

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(out / "calibration_summary.csv", index=False)

    # Paired DeLong + stratified bootstrap tests against Combined
    comb = pooled_scores["combined"]
    yt = comb["y_true"].to_numpy()
    p_comb = comb["raw"].to_numpy()
    n_boot = int(cfg.robustness["delong_bootstrap_b"])
    rng = np.random.default_rng(cfg.seed)
    pos_idx = np.where(yt == 1)[0]
    neg_idx = np.where(yt == 0)[0]

    delong_rows = []
    for comp_fs in ("adolescent", "parent"):
        p_sub = pooled_scores[comp_fs]["raw"].to_numpy()
        delta, p_delong = _delong_two_roc(yt, p_comb, p_sub)
        diffs = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                                  rng.choice(neg_idx, len(neg_idx), replace=True)])
            diffs[b] = roc_auc_score(yt[idx], p_comb[idx]) - roc_auc_score(yt[idx], p_sub[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        delong_rows.append({
            "cohort": cohort,
            "comparison": f"Combined vs {FEATURE_SET_LABEL[comp_fs]}",
            "delta_auc": delta,
            "boot_ci95_low": float(lo),
            "boot_ci95_high": float(hi),
            "delong_p": p_delong,
        })
    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(out / "delong_feature_set_tests.csv", index=False)

    # Perpetration-excluded sensitivity (dropping SV3_code, CV2_code)
    perp_cols = list(cfg.robustness["perpetration_columns"])
    sens_rows = []
    for fs in ("combined", "adolescent"):
        X_noperp, y_all = prepare_xy(df, cfg, cohort, fs, drop_columns=perp_cols)
        fold_aucs, fold_f1s = [], []
        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_noperp, y_all), start=1):
            X_tr, X_te = X_noperp.iloc[tr_idx], X_noperp.iloc[te_idx]
            y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
            oof = inner_oof_rf(X_tr, y_tr, cfg, rec["sampler"], rec["loss"], rec["hyperparameters"], rec["focal_gamma"], n_estimators=n_est)
            th = youden_threshold(y_tr, oof)
            X_tr_ml, X_te_ml, _ = fit_transform_fold(X_tr, X_te, cfg)
            X_res, y_res = resample_xy(X_tr_ml, y_tr, rec["sampler"], cfg.outer_seed + fold)
            clf = fit_rf_with_loss(X_res, y_res, rec["loss"], rec["hyperparameters"], cfg.outer_seed + fold, rec["focal_gamma"], n_estimators=n_est)
            sc = get_positive_scores(clf, X_te_ml)
            m = compute_metrics(y_te, (sc >= th).astype(int), sc)
            fold_aucs.append(m["auc"])
            fold_f1s.append(m["f1"])
        sens_rows.append({
            "cohort": cohort,
            "feature_set": FEATURE_SET_LABEL[fs],
            "dropped_columns": ",".join(perp_cols),
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_sd": float(np.std(fold_aucs, ddof=1)),
            "f1_mean": float(np.mean(fold_f1s)),
            "f1_sd": float(np.std(fold_f1s, ddof=1)),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out / "perpetration_excluded_sensitivity.csv", index=False)
    return {"calibration": calib_df, "delong": delong_df, "perpetration_sensitivity": sens_df}
