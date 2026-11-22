"""Command-line interface.

    cd src
    python -m odd_diccd_ml locked-cv    --target odd             # step 1: 5x5 nested CV with locked recipe
    python -m odd_diccd_ml stage-search --target odd             # optional: full 4-stage sequential search
    python -m odd_diccd_ml importance   --target odd             # step 2: outer-test permutation (dAUC) + SHAP
    python -m odd_diccd_ml association  --target odd             # step 3: adjusted ORs (BH-FDR) + Spearman rho
    python -m odd_diccd_ml robustness   --target odd             # step 4: Platt/isotonic, DeLong, no-perpetration
    python -m odd_diccd_ml all                                   # steps 1-4 for both ODD and DICCD
    python -m odd_diccd_ml synthetic    --out-dir ../data/synthetic   # schema-identical synthetic data

Common options (before the sub-command): --data-dir, --results-dir.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from .config import load_config

warnings.filterwarnings("ignore", category=FutureWarning)


def cmd_locked_cv(cfg, a):
    from .data import load_cohort
    from .nested_cv import run_locked_cv
    df = load_cohort(cfg, a.target)
    res = run_locked_cv(cfg, df, a.target, feature_sets=a.feature_sets, n_estimators_override=a.n_estimators, verbose=not a.quiet)
    print(res["performance_summary"][["feature_set_label", "auc_mean", "auc_sd", "f1_mean", "f1_sd", "average_precision_mean"]].round(4).to_string(index=False))


def cmd_stage_search(cfg, a):
    from .data import load_cohort
    from .nested_cv import run_stage_search
    df = load_cohort(cfg, a.target)
    run_stage_search(cfg, df, a.target, stages=a.stages, feature_sets=a.feature_sets, run_optuna=a.optuna, verbose=not a.quiet)


def cmd_importance(cfg, a):
    from .data import load_cohort
    from .importance import run_importance
    df = load_cohort(cfg, a.target)
    res = run_importance(
        cfg, df, a.target, feature_sets=a.feature_sets,
        n_repeats=a.n_repeats, no_shap=a.no_shap, n_estimators_override=a.n_estimators,
        verbose=not a.quiet,
    )
    comb = res["permutation_summary"]
    comb = comb[comb["feature_set"] == (a.feature_sets[0] if a.feature_sets else "combined")].head(10)
    print(comb[["rank", "label", "domain", "delta_auc_mean", "delta_auc_sd"]].round(4).to_string(index=False))


def cmd_association(cfg, a):
    from .association import run_association
    from .data import load_cohort
    df = load_cohort(cfg, a.target)
    assoc_df, _ = run_association(cfg, df, a.target, raw_xlsx=a.raw_xlsx, exposures_file=a.exposures_file)
    cols = ["Level", "Variable", "Contrast", "SHAP Rank (C/A/P)", "N", "N_cases", "Crude OR (95% CI)", "Adjusted OR (95% CI)", "p", "q (FDR)"]
    print(assoc_df[cols].to_string(index=False))


def cmd_robustness(cfg, a):
    from .data import load_cohort
    from .robustness import run_robustness
    df = load_cohort(cfg, a.target)
    res = run_robustness(cfg, df, a.target, n_estimators_override=a.n_estimators)
    print(res["delong"].round(4).to_string(index=False))


def cmd_synthetic(cfg, a):
    from .synthetic import write_synthetic_bundle
    paths = write_synthetic_bundle(cfg, a.out_dir, n=a.n, seed=a.seed)
    for k, p in paths.items():
        print(f"{k}: {p}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m odd_diccd_ml",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    def target_args(sp):
        sp.add_argument("--target", default="odd", choices=["odd", "diccd", "ODD", "DICCD"])
        sp.add_argument("--feature-sets", nargs="+", choices=["combined", "adolescent", "parent"])

    s = sub.add_parser("locked-cv", help="evaluate the locked per-target recipe under nested 5x5 CV")
    target_args(s)
    s.add_argument("--n-estimators", type=int, default=None)
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("stage-search", help="run the 4-stage sequential search (Model -> Sampler -> Loss -> HParam)")
    target_args(s)
    s.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    s.add_argument("--optuna", action="store_true", help="also run the 40-trial Optuna TPE check")
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("importance", help="outer-test permutation importance (delta-AUC) and SHAP")
    target_args(s)
    s.add_argument("--n-repeats", type=int, default=None)
    s.add_argument("--no-shap", action="store_true")
    s.add_argument("--n-estimators", type=int, default=None)
    s.add_argument("--quiet", action="store_true")

    s = sub.add_parser("association", help="crude & covariate-adjusted ORs (BH-FDR) + Spearman correlations")
    s.add_argument("--target", default="odd", choices=["odd", "diccd", "ODD", "DICCD"])
    s.add_argument("--raw-xlsx", type=Path, default=None)
    s.add_argument("--exposures-file", type=Path, default=None)

    s = sub.add_parser("robustness", help="Platt/isotonic calibration, paired DeLong tests, no-perpetration check")
    s.add_argument("--target", default="odd", choices=["odd", "diccd", "ODD", "DICCD"])
    s.add_argument("--n-estimators", type=int, default=None)

    s = sub.add_parser("all", help="locked-cv + importance + association + robustness for both ODD and DICCD")
    s.add_argument("--include-stage-search", action="store_true")

    s = sub.add_parser("synthetic", help="write schema-identical SYNTHETIC data (demo/testing only)")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--seed", type=int, default=42)

    a = p.parse_args(argv)
    cfg = load_config(data_dir=a.data_dir, results_dir=a.results_dir)
    t0 = time.time()

    if a.cmd == "all":
        for tgt in ("odd", "diccd"):
            ns = argparse.Namespace(
                target=tgt, feature_sets=None, n_estimators=None, quiet=False,
                n_repeats=None, no_shap=False, raw_xlsx=None, exposures_file=None,
                stages=[1, 2, 3, 4], optuna=True,
            )
            if a.include_stage_search:
                cmd_stage_search(cfg, ns)
            cmd_locked_cv(cfg, ns)
            cmd_importance(cfg, ns)
            cmd_association(cfg, ns)
            cmd_robustness(cfg, ns)
    else:
        {
            "locked-cv": cmd_locked_cv,
            "stage-search": cmd_stage_search,
            "importance": cmd_importance,
            "association": cmd_association,
            "robustness": cmd_robustness,
            "synthetic": cmd_synthetic,
        }[a.cmd](cfg, a)

    print(json.dumps({"command": a.cmd, "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
