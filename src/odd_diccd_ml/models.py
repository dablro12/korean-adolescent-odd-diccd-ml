"""Classifiers, train-only resamplers, loss/sample-weight schemes, Youden cut-off, and metrics."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTE, SVMSMOTE
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_stage1_pipeline(model_name: str, scale: bool, seed: int) -> Pipeline:
    """Build one of the 8 Stage-1 candidate classifiers (no resampling)."""
    steps = [("scaler", StandardScaler())] if scale else []
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif model_name == "Naive Bayes":
        clf = GaussianNB()
    elif model_name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=seed)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=seed)
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(random_state=seed)
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    elif model_name == "CatBoost":
        clf = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, loss_function="Logloss",
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    steps.append(("model", clf))
    return Pipeline(steps)


def build_sampler(method: str, y_train, seed: int):
    """Instantiate a 1:1 minority-to-majority resampler for a training partition."""
    if method == "None":
        return None
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    target = {1: n_neg}
    k = max(1, min(5, n_pos - 1))
    if method == "SMOTE":
        return SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k)
    if method == "ADASYN":
        return ADASYN(sampling_strategy=target, random_state=seed, n_neighbors=k)
    if method == "BorderlineSMOTE":
        return BorderlineSMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k)
    if method == "SVMSMOTE":
        return SVMSMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k)
    if method == "KMeansSMOTE":
        n_clusters = max(2, min(10, max(2, n_pos // 8)))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
        return KMeansSMOTE(
            sampling_strategy=target, random_state=seed, k_neighbors=k,
            kmeans_estimator=km, cluster_balance_threshold=0.01,
        )
    if method == "SMOTEENN":
        return SMOTEENN(
            sampling_strategy=target, random_state=seed,
            smote=SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k),
        )
    if method == "SMOTETomek":
        return SMOTETomek(
            sampling_strategy=target, random_state=seed,
            smote=SMOTE(sampling_strategy=target, random_state=seed, k_neighbors=k),
        )
    raise ValueError(f"Unknown sampler: {method}")


def resample_xy(X_tr: pd.DataFrame, y_tr: pd.Series, sampler_name: str, seed: int):
    sampler = build_sampler(sampler_name, y_tr, seed)
    if sampler is None:
        return X_tr, y_tr
    return sampler.fit_resample(X_tr, y_tr)


def focal_sample_weight(y_train, p_pos: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Two-step focal sample weights (class-balanced alpha * (1 - p_t)^gamma, mean-normalized)."""
    y = np.asarray(y_train)
    n = len(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    alpha_pos, alpha_neg = n_neg / n, n_pos / n
    p = np.clip(p_pos, 1e-6, 1.0 - 1e-6)
    p_t = np.where(y == 1, p, 1.0 - p)
    alpha_t = np.where(y == 1, alpha_pos, alpha_neg)
    sw = np.clip(alpha_t * ((1.0 - p_t) ** gamma), 1e-3, None)
    return sw * (n / sw.sum())


def build_rf(hparams: dict, seed: int, n_estimators: int = 500, class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        criterion=hparams["criterion"],
        min_samples_leaf=int(hparams["min_samples_leaf"]),
        max_depth=None if hparams["max_depth"] is None else int(hparams["max_depth"]),
        max_features=hparams["max_features"],
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )


def fit_rf_with_loss(
    X_res,
    y_res,
    loss: str,
    hparams: dict,
    seed: int,
    focal_gamma: Optional[float] = 2.0,
    n_estimators: int = 500,
) -> RandomForestClassifier:
    """Fit a single RandomForestClassifier with the specified loss and hyperparameters."""
    if loss == "BCE":
        clf = build_rf(hparams, seed, n_estimators=n_estimators)
        clf.fit(X_res, y_res)
        return clf
    if loss == "Weighted BCE":
        y = np.asarray(y_res)
        w = float((y == 0).sum() / max(int((y == 1).sum()), 1))
        clf = build_rf(hparams, seed, n_estimators=n_estimators, class_weight={0: 1.0, 1: w})
        clf.fit(X_res, y_res)
        return clf
    if loss == "Focal":
        gamma = 2.0 if focal_gamma is None else float(focal_gamma)
        warm = build_rf(hparams, seed, n_estimators=n_estimators)
        warm.fit(X_res, y_res)
        p_pos = get_positive_scores(warm, X_res)
        sw = focal_sample_weight(y_res, p_pos, gamma=gamma)
        clf = build_rf(hparams, seed, n_estimators=n_estimators)
        clf.fit(X_res, y_res, sample_weight=sw)
        return clf
    raise ValueError(f"Unknown loss: {loss}")


def get_positive_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise ValueError("Estimator must expose predict_proba or decision_function.")


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximizing Youden's J = Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    return {
        "auc": auc,
        "average_precision": ap,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def rank_consistent_winner(
    metrics_by_tag: Dict[str, Tuple[float, float, float]],
    exclude: Iterable[str] = (),
) -> Tuple[str, pd.DataFrame]:
    """Select candidate minimizing rank_sum across Inner-OOF AUC, F1, and AP (ties broken by AP, AUC)."""
    excl = set(exclude)
    rows = [{"candidate": k, "auc": v[0], "f1": v[1], "ap": v[2]} for k, v in metrics_by_tag.items() if k not in excl]
    df = pd.DataFrame(rows)
    df["auc_rank"] = df["auc"].rank(ascending=False, method="min")
    df["f1_rank"] = df["f1"].rank(ascending=False, method="min")
    df["ap_rank"] = df["ap"].rank(ascending=False, method="min")
    df["rank_sum"] = df["auc_rank"] + df["f1_rank"] + df["ap_rank"]
    df = df.sort_values(["rank_sum", "ap", "auc"], ascending=[True, False, False]).reset_index(drop=True)
    return str(df.iloc[0]["candidate"]), df
