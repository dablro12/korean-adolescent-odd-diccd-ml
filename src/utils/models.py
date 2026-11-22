import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd 
warnings.filterwarnings("ignore")

SEED = 42
CV_FOLDS = 5
N_BOOTSTRAP = 1000
GPU_DEVICE = "cuda:1"
CATBOOST_GPU_DEVICES = "1"

# SUICIDE target: 0 = Negative (no suicide), 1 = Positive (suicide)
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
TARGET_LABELS = (NEGATIVE_LABEL, POSITIVE_LABEL)

METRIC_COLUMNS = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "Recall",
    "F1-score",
    "AUC Score",
]

MODEL_CONFIGS = [
    ("Logistic Regression", True),
    ("Naive Bayes", True),
    ("Linear SVM", True),
    ("RBF SVM", True),
    ("Random Forest", False),
    ("Gradient Boosting", False),
    ("XGBoost", False),
    ("CatBoost", False),
]

TREE_MODELS = {
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "CatBoost",
}
LINEAR_MODELS = {"Logistic Regression", "Linear SVM"}


def class_balance_table(y_arr, label: str) -> pd.DataFrame:
    y_arr = np.asarray(y_arr)
    return pd.DataFrame(
        {
            "split": [label],
            "n": [len(y_arr)],
            "positive_n": [int((y_arr == POSITIVE_LABEL).sum())],
            "negative_n": [int((y_arr == NEGATIVE_LABEL).sum())],
            "positive_rate": [float((y_arr == POSITIVE_LABEL).mean())],
        }
    )


def _positive_class_index(model) -> int:
    """모델이 학습한 classes_에서 Positive(1) 열 인덱스를 반환한다."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    classes = getattr(clf, "classes_", None)
    if classes is not None and POSITIVE_LABEL in classes:
        return int(np.where(classes == POSITIVE_LABEL)[0][0])
    return 1


def get_positive_scores(model, X, positive_label: int = POSITIVE_LABEL) -> np.ndarray:
    """Positive class(1=자살)에 대한 예측 점수(확률 또는 정규화 decision score)를 반환한다."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return np.asarray(proba)
        pos_idx = _positive_class_index(model)
        return proba[:, pos_idx]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim > 1:
            pos_idx = _positive_class_index(model)
            scores = scores[:, pos_idx]
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return (model.predict(X) == positive_label).astype(float)


def compute_scale_pos_weight(y_train) -> float:
    """원본 train 분포 기준 scale_pos_weight = n_negative / n_positive."""
    y = np.asarray(y_train)
    n_pos = int((y == POSITIVE_LABEL).sum())
    n_neg = int((y == NEGATIVE_LABEL).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def _class_weight_dict(scale_pos_weight: float) -> Dict[int, float]:
    return {NEGATIVE_LABEL: 1.0, POSITIVE_LABEL: float(scale_pos_weight)}


def youden_threshold(y_true, y_score) -> float:
    """train CV OOF 점수에서 Youden index(J = TPR - FPR)가 최대인 임계값."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 0.5
    j_scores = tpr[valid] - fpr[valid]
    return float(thresholds[valid][int(np.argmax(j_scores))])


def compute_metrics(
    y_true,
    y_pred,
    y_score,
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[negative_label, positive_label]
    ).ravel()
    # Sensitivity/Recall: Positive(1) 탐지율 | Specificity: Negative(0) 탐지율
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "AUC Score": (
            roc_auc_score(y_true, y_score, labels=[negative_label, positive_label])
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
    }


def build_model_pipeline(model_name: str, needs_scale: bool = False) -> ImbPipeline:
    """SMOTE oversampling + classifier (2_ml_model비교.ipynb와 동일 설정)."""
    steps: List[Tuple[str, object]] = []
    if needs_scale:
        steps.append(("scaler", StandardScaler()))

    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=3000, random_state=SEED)
    elif model_name == "Naive Bayes":
        clf = GaussianNB()
    elif model_name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=SEED)
    elif model_name == "RBF SVM":
        clf = SVC(kernel="rbf", probability=True, random_state=SEED)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.9,
            random_state=SEED,
        )
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            device=GPU_DEVICE,
            tree_method="hist",
        )
    elif model_name == "CatBoost":
        clf = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=SEED,
            verbose=0,
            allow_writing_files=False,
            task_type="GPU",
            devices=CATBOOST_GPU_DEVICES,
        )
    else:
        raise ValueError(model_name)

    steps.extend([
        ("smote", SMOTE(random_state=SEED)),
        ("clf", clf),
    ])
    return ImbPipeline(steps)


def cv_oof_scores_and_threshold(
    pipeline: ImbPipeline,
    X_train_df,
    y_train_series,
) -> Tuple[np.ndarray, float]:
    """5-fold CV OOF 점수 + Youden 임계값 (train only)."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.zeros(len(y_train_series), dtype=float)

    for tr_idx, va_idx in skf.split(X_train_df, y_train_series):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train_df.iloc[tr_idx], y_train_series.iloc[tr_idx])
        oof_scores[va_idx] = get_positive_scores(fold_pipe, X_train_df.iloc[va_idx])

    threshold = youden_threshold(y_train_series, oof_scores)
    return oof_scores, threshold


def stratified_bootstrap_ci(
    y_true,
    y_score,
    threshold: float,
    metric_name: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_idx = np.flatnonzero(y_true == POSITIVE_LABEL)
    neg_idx = np.flatnonzero(y_true == NEGATIVE_LABEL)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan, np.nan

    boot_values = []
    for _ in range(n_bootstrap):
        boot_idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yt = y_true[boot_idx]
        ys = y_score[boot_idx]
        yp = np.where(ys >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
        boot_values.append(compute_metrics(yt, yp, ys)[metric_name])

    lower, upper = np.percentile(boot_values, [2.5, 97.5])
    return float(np.mean(boot_values)), float(lower), float(upper)
