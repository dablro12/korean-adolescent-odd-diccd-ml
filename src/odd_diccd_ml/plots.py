"""Publication-style diagnostic figures from summary tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURE_SETS = [("combined", "Combined"), ("adolescent", "Adolescent"), ("parent", "Parent")]
COLORS = {"combined": "#1f77b4", "adolescent": "#ff7f0e", "parent": "#2ca02c"}
BAR_ADOL, BAR_PAR = "#4C72B0", "#DD8452"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)


def roc_pr_ribbons(curves: pd.DataFrame, perf: pd.DataFrame, title_prefix: str = ""):
    """Mean ROC and PR curves over outer folds with +/-1 SD ribbons across feature sets."""
    p = perf.set_index("feature_set")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, kind in enumerate(["roc", "pr"]):
        ax = axes[i]
        for fs, fs_lab in FEATURE_SETS:
            if fs not in p.index:
                continue
            d = curves[(curves["kind"] == kind) & (curves["feature_set"] == fs)]
            g = d.groupby("x")["y"].agg(["mean", "std"]).reset_index()
            lo = (g["mean"] - g["std"].fillna(0)).clip(0, 1)
            hi = (g["mean"] + g["std"].fillna(0)).clip(0, 1)
            r = p.loc[fs]
            stat = (f"AUC {r.auc_mean:.3f} ± {r.auc_sd:.3f}" if kind == "roc"
                    else f"AP {r.average_precision_mean:.3f} ± {r.average_precision_sd:.3f}")
            ax.fill_between(g["x"], lo, hi, color=COLORS[fs], alpha=0.15, lw=0)
            ax.plot(g["x"], g["mean"], color=COLORS[fs], lw=2, label=f"{fs_lab} ({stat})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
            ax.set_xlabel("1 - Specificity")
            ax.set_ylabel("Sensitivity")
            ax.set_title(f"{title_prefix}ROC Curve (5 outer folds)".strip(), fontsize=12, fontweight="bold")
        else:
            base = float(p.iloc[0]["case_proportion"])
            ax.axhline(base, ls="--", color="grey", lw=1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{title_prefix}Precision-Recall Curve (5 outer folds)".strip(), fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=8.5, frameon=False)
        _style(ax)
    fig.tight_layout()
    return fig


def confusion_grid(cm: pd.DataFrame, title_prefix: str = ""):
    """Outer-test confusion matrices pooled over folds across Combined, Adolescent, and Parent sets."""
    c = cm.set_index("feature_set")
    present = [(fs, lab) for fs, lab in FEATURE_SETS if fs in c.index]
    fig, axes = plt.subplots(1, len(present), figsize=(3.7 * len(present), 3.5), squeeze=False)
    for j, (fs, fs_lab) in enumerate(present):
        ax = axes[0, j]
        r = c.loc[fs]
        m = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        pct = m / np.maximum(m.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for b in range(2):
                ax.text(
                    b, a, f"{int(m[a, b])}\n({pct[a, b] * 100:.1f}%)",
                    ha="center", va="center",
                    color="white" if pct[a, b] > 0.5 else "black", fontsize=10,
                )
        ax.set_xticks([0, 1], ["Pred. control", "Pred. case"])
        ax.set_yticks([0, 1], ["Control", "Case"])
        ax.set_title(f"{title_prefix}{fs_lab}".strip(), fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def importance_bars(
    summary_df: pd.DataFrame,
    value_col: str = "mean_abs_shap",
    sd_col: str = "sd_abs_shap",
    xlabel: str = "mean |SHAP| (mean ± SD over outer folds)",
    top_k: int = 10,
):
    """Horizontal top-k importance bar plots across Combined, Adolescent, and Parent feature sets."""
    present = [(fs, lab) for fs, lab in FEATURE_SETS if fs in set(summary_df["feature_set"])]
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.2), squeeze=False)
    for j, (fs, fs_lab) in enumerate(present):
        ax = axes[0, j]
        sub = summary_df[summary_df["feature_set"] == fs].nsmallest(top_k, "rank").iloc[::-1]
        cols = [BAR_PAR if dom == "Parent" else BAR_ADOL for dom in sub["domain"]]
        ax.barh(
            sub["label"], sub[value_col], xerr=sub[sd_col], color=cols, alpha=0.88,
            error_kw=dict(ecolor="#7f8c8d", capsize=2),
        )
        ax.set_title(fs_lab, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9.5)
        _style(ax)
    fig.tight_layout()
    return fig


def association_forest(assoc_df: pd.DataFrame, title: str = "Covariate-Adjusted Odds Ratios (95% CI)"):
    """Log-scale forest plot of adjusted odds ratios from association_analysis."""
    t = assoc_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 0.34 * len(t) + 1.2))
    x = t["adjusted_or"].to_numpy()
    lo = np.maximum(x - t["adjusted_lo"].to_numpy(), 0.0)
    hi = np.maximum(t["adjusted_hi"].to_numpy() - x, 0.0)
    colors = [BAR_PAR if lv == "Parent-level" else BAR_ADOL for lv in t["Level"]]
    for i in range(len(t)):
        ax.errorbar(x[i], i, xerr=[[lo[i]], [hi[i]]], fmt="o", color=colors[i], ecolor="#7f8c8d", capsize=3)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(t)), t["Variable"], fontsize=9)
    ax.set_xlabel("Adjusted Odds Ratio (log scale, 95% CI)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    _style(ax)
    fig.tight_layout()
    return fig
