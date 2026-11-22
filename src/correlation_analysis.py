#!/usr/bin/env python3
"""Spearman correlation of each Table-3-style variable against the target
(= phi coefficient for binary x binary pairs), plus an annotated heatmap.

Usage:
    python src/correlation_analysis.py --target ODD --data-dir data --output-dir results/odd_nested
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from association_analysis import EXPOSURES, build_exposure_frame  # noqa: E402
from recipe import data_paths  # noqa: E402
from stage1_model_comparison import meta_target_column  # noqa: E402
from utils.figure_style import use_times_style  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["ODD", "DICCD"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    args = ap.parse_args()

    out = args.output_dir or (REPO_ROOT / "results" / f"{args.target.lower()}_nested")
    out.mkdir(parents=True, exist_ok=True)
    csv_path, _ = data_paths(args.target, args.data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected data at {csv_path}. See data/README.md.")

    df = pd.read_csv(csv_path)
    target_col = meta_target_column(args.target)
    frame = build_exposure_frame(df, target_col, age_series=None)

    rows = []
    for label, col, _contrast, _drop in EXPOSURES:
        use = frame[["y", col]].dropna()
        rho, p = spearmanr(use["y"], use[col])
        rows.append({"Variable": label, "rho": rho, "p": p, "n": len(use)})
    out_df = pd.DataFrame(rows).sort_values("rho", key=np.abs, ascending=False)
    out_df.to_csv(out / f"correlation_{args.target.lower()}.csv", index=False)

    use_times_style()
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(out_df) + 1))
    colors = ["#c0392b" if r > 0 else "#2980b9" for r in out_df["rho"]]
    ax.barh(out_df["Variable"][::-1], out_df["rho"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho")
    ax.set_title(f"{args.target}: variable correlations (Spearman)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"correlation_{args.target.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== DONE ===")
    print(out_df.to_string(index=False))
    print(f"Wrote {out}/correlation_{args.target.lower()}.csv and .png")


if __name__ == "__main__":
    main()
