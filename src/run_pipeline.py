#!/usr/bin/env python3
"""Runs the full pipeline for one target: Stage 1-4 -> Permutation -> SHAP ->
Association -> Correlation. Each stage reads its locked-in recipe from
configs/recipes.yaml rather than assuming the other target's choices.

Usage:
    python src/run_pipeline.py --target ODD --data-dir data --output-dir results/odd_nested
    python src/run_pipeline.py --target DICCD --data-dir data --output-dir results/diccd_nested
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(script: str, target: str, data_dir: Path, out_dir: Path, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "src" / script), "--target", target,
           "--data-dir", str(data_dir), "--output-dir", str(out_dir)]
    if extra:
        cmd.extend(extra)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["ODD", "DICCD"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--skip-grid-search", action="store_true", help="Skip stage4 grid search and only run the Optuna variant (faster).")
    args = ap.parse_args()

    out = args.output_dir or (REPO_ROOT / "results" / f"{args.target.lower()}_nested")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run("stage1_model_comparison.py", args.target, args.data_dir, out)
    run("stage2_sampler_comparison.py", args.target, args.data_dir, out)
    run("stage3_loss_comparison.py", args.target, args.data_dir, out)
    if not args.skip_grid_search:
        run("stage4_hyperparameter_comparison.py", args.target, args.data_dir, out)
    run("stage4_optuna_search.py", args.target, args.data_dir, out, extra=["--n-trials", "40"])
    for fs in ("all", "adolescent", "parent"):
        run("permutation_importance.py", args.target, args.data_dir, out, extra=["--feature-set", fs])
        run("shap_importance.py", args.target, args.data_dir, out, extra=["--feature-set", fs])
    run("association_analysis.py", args.target, args.data_dir, out)
    run("correlation_analysis.py", args.target, args.data_dir, out)

    print(f"\n=== {args.target} pipeline complete in {time.time()-t0:.0f}s ===")
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
