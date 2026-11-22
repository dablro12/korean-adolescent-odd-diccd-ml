"""Covariate-adjusted logistic regression odds ratios (BH-FDR) and Spearman correlations."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from .config import Config

# Locked per-target Top-10 SHAP feature sets (used when standalone association runs before SHAP)
LOCKED_TOP10_FEATURES: Dict[str, Dict[str, List[str]]] = {
    "odd": {
        "Combined": [
            "continuous__SAS1_total", "binary__CV1_code", "continuous__PHQ_total", "continuous__GAD_total",
            "ordinal__ST1_4", "ordinal__ksQ15_t", "binary__CV2_code", "binary__ksQ16_t",
            "ordinal_parent__FA_M", "binary__SV3_code",
        ],
        "Adolescent": [
            "continuous__SAS1_total", "ordinal__ST1_4", "binary__CV1_code", "ordinal__ksQ15_t",
            "binary__CV2_code", "ordinal__PHA1", "ordinal__ST1_1", "ordinal__ST1_3",
            "binary__ksQ16_t", "ordinal__ST1_2",
        ],
        "Parent": [
            "continuous__PHQ_total", "continuous__GAD_total", "continuous__F6", "ordinal_parent__FA_M",
            "ordinal_parent__FA_F", "ordinal_parent__FT_F", "ordinal_parent__F4_M", "binary__PS_code",
            "ordinal_parent__F4_F", "nominal__F5F_3.0",
        ],
    },
    "diccd": {
        "Combined": [
            "continuous__SAS1_total", "binary__CV1_code", "binary__CV2_code", "binary__ksQ16_t",
            "binary__SV3_code", "continuous__GAD_total", "continuous__PHQ_total", "ordinal__ksQ15_t",
            "ordinal__ST1_4", "nominal__IGD_P_1.0",
        ],
        "Adolescent": [
            "continuous__SAS1_total", "binary__CV1_code", "binary__CV2_code", "binary__ksQ16_t",
            "binary__SV3_code", "ordinal__ksQ15_t", "ordinal__ST1_4", "nominal__IGD_P_1.0",
            "binary__SV2_code", "ordinal__PHA1",
        ],
        "Parent": [
            "continuous__GAD_total", "continuous__PHQ_total", "ordinal_parent__F4_M", "ordinal_parent__FA_M",
            "ordinal_parent__FA_F", "ordinal_parent__FT_F", "ordinal_parent__F4_F", "continuous__F6",
            "nominal__F5M_5.0", "nominal__F5M_4.0",
        ],
    },
}


def build_exposure_frame(df: pd.DataFrame, cfg: Config, cohort: str, age_series: Optional[pd.Series]) -> pd.DataFrame:
    target_col = cfg.target_col(cohort)
    grades = pd.to_numeric(df["ksQ15_t"], errors="coerce").where(lambda s: s.between(1, 5))
    activity = pd.to_numeric(df["PHA1"], errors="coerce").where(lambda s: s.between(1, 5))
    igd = pd.to_numeric(df["IGD_P"], errors="coerce")
    medu = pd.to_numeric(df["F4_M"], errors="coerce")
    fedu = pd.to_numeric(df["F4_F"], errors="coerce")
    malc = pd.to_numeric(df["FA_M"], errors="coerce").where(lambda s: s.between(1, 5))
    falc = pd.to_numeric(df["FA_F"], errors="coerce").where(lambda s: s.between(1, 5))
    msmoke = pd.to_numeric(df["FT_M"], errors="coerce").where(lambda s: s.between(1, 5))
    fsmoke = pd.to_numeric(df["FT_F"], errors="coerce").where(lambda s: s.between(1, 5))
    phone = pd.to_numeric(df["SAS1_total"], errors="coerce")
    income = pd.to_numeric(df["F6"], errors="coerce")
    phq = pd.to_numeric(df["PHQ_total"], errors="coerce")
    gad = pd.to_numeric(df["GAD_total"], errors="coerce")
    f5m = pd.to_numeric(df["F5M"], errors="coerce")
    f5f = pd.to_numeric(df["F5F"], errors="coerce")
    f3 = pd.to_numeric(df["F3"], errors="coerce")
    dong = pd.to_numeric(df["dong"], errors="coerce")

    def screen_ge3(col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce").where(lambda x: x.between(1, 5))
        return pd.Series(np.where(s.isna(), np.nan, (s >= 3).astype(float)), index=df.index)

    def known_occ(s: pd.Series, code: int) -> pd.Series:
        known = s.notna() & ~s.isin([9998, 9999])
        return pd.Series(np.where(known, (s == code).astype(float), np.nan), index=df.index)

    frame = pd.DataFrame({
        "y": pd.to_numeric(df[target_col], errors="coerce").astype(int),
        "male": (pd.to_numeric(df[cfg.sex_column], errors="coerce") == 1).astype(int),
        "urban": pd.Series(np.where(dong.isin([1, 2]), (dong == 1).astype(float), np.nan), index=df.index),
        "phone_z": (phone - phone.mean()) / phone.std(ddof=0),
        "cyber_victim": pd.to_numeric(df["CV1_code"], errors="coerce"),
        "cyber_perp": pd.to_numeric(df["CV2_code"], errors="coerce"),
        "bully_victim": pd.to_numeric(df["SV2_code"], errors="coerce"),
        "bully_perp": pd.to_numeric(df["SV3_code"], errors="coerce"),
        "sleep_prob": pd.to_numeric(df["CS_code"], errors="coerce"),
        "academic_decline": pd.to_numeric(df["ksQ16_t"], errors="coerce"),
        "best_friend": pd.to_numeric(df["ksQ22_t"], errors="coerce"),
        "peer_group": pd.to_numeric(df["ksQ24_t"], errors="coerce"),
        "grades": grades,
        "activity_days": activity,
        "video_screen": screen_ge3("ST1_1"),
        "sns_screen": screen_ge3("ST1_2"),
        "gaming_screen": screen_ge3("ST1_3"),
        "edu_screen": screen_ge3("ST1_4"),
        "game_high": pd.Series(np.where(igd.isin([0, 1]), (igd == 1).astype(float), np.nan), index=df.index),
        "phq_high": pd.Series(np.where(phq.isna(), np.nan, (phq >= 9).astype(float)), index=df.index),
        "gad_high": pd.Series(np.where(gad.isna(), np.nan, (gad >= 5).astype(float)), index=df.index),
        "income_high": pd.Series(np.where(income.isna(), np.nan, (income >= 500).astype(float)), index=df.index),
        "ps_prob": pd.to_numeric(df["PS_code"], errors="coerce"),
        "medu_college": pd.Series(np.where(medu.isin([1, 2, 3, 4]), (medu >= 3).astype(float), np.nan), index=df.index),
        "fedu_college": pd.Series(np.where(fedu.isin([1, 2, 3, 4]), (fedu >= 3).astype(float), np.nan), index=df.index),
        "malc_high": pd.Series(np.where(malc.isna(), np.nan, (malc >= 3).astype(float)), index=df.index),
        "falc_high": pd.Series(np.where(falc.isna(), np.nan, (falc >= 3).astype(float)), index=df.index),
        "msmoke_any": pd.Series(np.where(msmoke.isna(), np.nan, (msmoke >= 2).astype(float)), index=df.index),
        "fsmoke_any": pd.Series(np.where(fsmoke.isna(), np.nan, (fsmoke >= 2).astype(float)), index=df.index),
        "mother_missing": pd.to_numeric(df["mother_missing"], errors="coerce"),
        "father_missing": pd.to_numeric(df["father_missing"], errors="coerce"),
        "marital_nonfirst": pd.Series(np.where(f3.notna() & (f3 != 9999), (f3 != 2).astype(float), np.nan), index=df.index),
    }, index=df.index)

    for code in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        frame[f"mocc_{code}"] = known_occ(f5m, code)
        frame[f"focc_{code}"] = known_occ(f5f, code)

    if age_series is not None:
        frame["age"] = pd.to_numeric(age_series, errors="coerce").to_numpy()
    return frame


def _feature_registry(cfg: Config) -> Dict[str, Tuple[str, str, str, Optional[str], str]]:
    occ_cats = cfg.features["categories"]["F5M"]
    reg: Dict[str, Tuple[str, str, str, Optional[str], str]] = {
        "SAS1_total": ("Smartphone Addiction", "phone_z", "per 1 SD", None, "Adolescent-level"),
        "CV1_code": ("Cyberbullying Victimization", "cyber_victim", "yes vs no", None, "Adolescent-level"),
        "CV2_code": ("Cyberbullying Perpetration", "cyber_perp", "yes vs no", None, "Adolescent-level"),
        "SV2_code": ("Bullying Victimization", "bully_victim", "yes vs no", None, "Adolescent-level"),
        "SV3_code": ("Bullying Perpetration", "bully_perp", "yes vs no", None, "Adolescent-level"),
        "CS_code": ("Sleep Problem", "sleep_prob", "yes vs no", None, "Adolescent-level"),
        "ksQ16_t": ("Academic Decline", "academic_decline", "yes vs no", None, "Adolescent-level"),
        "ksQ15_t": ("Academic Performance", "grades", "per step toward a lower grade", None, "Adolescent-level"),
        "ksQ22_t": ("Best Friend", "best_friend", "yes vs no", None, "Adolescent-level"),
        "ksQ24_t": ("Peer Group", "peer_group", "yes vs no", None, "Adolescent-level"),
        "PHA1": ("Physical Activity", "activity_days", "per higher activity category (1-5)", None, "Adolescent-level"),
        "ST1_1": ("Video Screen Time", "video_screen", ">=3 h/day vs <3 h/day", None, "Adolescent-level"),
        "ST1_2": ("Social Media Screen Time", "sns_screen", ">=3 h/day vs <3 h/day", None, "Adolescent-level"),
        "ST1_3": ("Gaming Screen Time", "gaming_screen", ">=3 h/day vs <3 h/day", None, "Adolescent-level"),
        "ST1_4": ("Educational Screen Time", "edu_screen", ">=3 h/day vs <3 h/day", None, "Adolescent-level"),
        "IGD_P": ("Game Addiction (High-risk)", "game_high", "high-risk vs no", None, "Adolescent-level"),
        "dong": ("Residential Area (Urban)", "urban", "urban vs rural", None, "Adolescent-level"),
        "PHQ_total": ("Parental Depression", "phq_high", "PHQ-9 >=9 vs <9", "phq_high", "Parent-level"),
        "GAD_total": ("Parental Anxiety", "gad_high", "GAD-7 >=5 vs <5", "gad_high", "Parent-level"),
        "F6": ("Household Income", "income_high", ">=5 million KRW/month vs less", None, "Parent-level"),
        "PS_code": ("Parental Sleep Problem", "ps_prob", "yes vs no", None, "Parent-level"),
        "F4_M": ("Maternal Education", "medu_college", "college or higher vs high school or less", None, "Parent-level"),
        "F4_F": ("Paternal Education", "fedu_college", "college or higher vs high school or less", None, "Parent-level"),
        "FA_M": ("Maternal Alcohol Use", "malc_high", ">=2 times/month vs less", None, "Parent-level"),
        "FA_F": ("Paternal Alcohol Use", "falc_high", ">=2 times/month vs less", None, "Parent-level"),
        "FT_M": ("Maternal Smoking", "msmoke_any", "current smoking vs none", None, "Parent-level"),
        "FT_F": ("Paternal Smoking", "fsmoke_any", "current smoking vs none", None, "Parent-level"),
        "mother_missing": ("Maternal Absence", "mother_missing", "absent vs present", None, "Parent-level"),
        "father_missing": ("Paternal Absence", "father_missing", "absent vs present", None, "Parent-level"),
        "F3": ("Parental Marital Status", "marital_nonfirst", "other marital status vs married (first)", None, "Parent-level"),
    }
    for code in range(1, 13):
        role = occ_cats[str(code)]
        reg[f"F5M_{code}"] = (f"Maternal Occupation ({role})", f"mocc_{code}", f"{role.lower()} vs other known jobs", None, "Parent-level")
        reg[f"F5F_{code}"] = (f"Paternal Occupation ({role})", f"focc_{code}", f"{role.lower()} vs other known jobs", None, "Parent-level")
    return reg


def _resolve_encoded(encoded_col: str, reg: dict):
    raw = encoded_col.split("__", 1)[-1] if "__" in encoded_col else encoded_col
    if raw in reg:
        return reg[raw]
    if "_" in raw:
        prefix, tail = raw.rsplit("_", 1)
        try:
            c_int = int(float(tail))
            if f"{prefix}_{c_int}" in reg:
                return reg[f"{prefix}_{c_int}"]
            if prefix in reg:
                return reg[prefix]
        except ValueError:
            pass
    raise KeyError(f"Unmapped encoded feature: {encoded_col}")


def resolve_exposures(
    cfg: Config,
    cohort: str,
    exposures_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    cohort = cohort.lower()
    reg = _feature_registry(cfg)
    top_k = int(cfg.importance["top_k"])

    if exposures_file is not None:
        imp_df = pd.read_csv(exposures_file).head(top_k)
        specs, seen = [], set()
        for rank, (_, r) in enumerate(imp_df.iterrows(), start=1):
            lbl, col_k, contrast, drop_cov, lvl = _resolve_encoded(str(r["feature"]), reg)
            if lbl not in seen:
                seen.add(lbl)
                specs.append({"level": lvl, "label": lbl, "col": col_k, "contrast": contrast, "drop": drop_cov, "rank_str": str(rank)})
        return specs

    imp_dir = cfg.cohort_out_dir(cohort) / "importance"
    shap_files = [
        ("Combined", imp_dir / "shap_combined.csv"),
        ("Adolescent", imp_dir / "shap_adolescent.csv"),
        ("Parent", imp_dir / "shap_parent.csv"),
    ]
    if all(p.exists() for _, p in shap_files):
        top_by_fs = {fs_name: pd.read_csv(p).head(top_k)["feature"].astype(str).tolist() for fs_name, p in shap_files}
    else:
        top_by_fs = LOCKED_TOP10_FEATURES[cohort]

    unique_map: Dict[str, Dict[str, Any]] = {}
    for s_idx, fs_name in enumerate(("Combined", "Adolescent", "Parent")):
        for rank, feat in enumerate(top_by_fs[fs_name], start=1):
            lbl, col_k, contrast, drop_cov, lvl = _resolve_encoded(feat, reg)
            if lbl not in unique_map:
                unique_map[lbl] = {
                    "level": lvl, "label": lbl, "col": col_k,
                    "contrast": contrast, "drop": drop_cov,
                    "ranks": {}, "best_key": (rank, s_idx),
                }
            unique_map[lbl]["ranks"][fs_name] = min(rank, unique_map[lbl]["ranks"].get(fs_name, 99))
            unique_map[lbl]["best_key"] = min(unique_map[lbl]["best_key"], (rank, s_idx))

    ordered = sorted(unique_map.values(), key=lambda d: (d["level"] == "Parent-level", d["best_key"]))
    for item in ordered:
        rmap = item["ranks"]
        item["rank_str"] = "/".join(str(rmap.get(fs, "-")) for fs in ("Combined", "Adolescent", "Parent"))
    return ordered


def _fit_logit(frame: pd.DataFrame, exposure: str, drop_covariate: Optional[str], covariates: List[str]) -> dict:
    covs = [c for c in covariates if c != drop_covariate]
    use = frame[["y", exposure, *covs]].dropna()
    covs = [c for c in covs if use[c].nunique() >= 2]
    y = use["y"].to_numpy()

    def _fit_one(X_mat: pd.DataFrame):
        try:
            return sm.Logit(y, X_mat).fit(disp=False)
        except np.linalg.LinAlgError:
            return sm.Logit(y, X_mat).fit(method="bfgs", maxiter=500, disp=False)

    crude = _fit_one(sm.add_constant(use[[exposure]].astype(float), has_constant="add"))
    adj = _fit_one(sm.add_constant(use[[exposure, *covs]].astype(float), has_constant="add"))

    def _term(res):
        b, se = float(res.params[exposure]), float(res.bse[exposure])
        return {
            "or": float(np.exp(b)),
            "lo": float(np.exp(b - 1.96 * se)),
            "hi": float(np.exp(b + 1.96 * se)),
            "p": float(res.pvalues[exposure]),
        }

    return {"n": int(len(use)), "n_cases": int(y.sum()), "crude": _term(crude), "adjusted": _term(adj)}


def _fmt_or(t: dict) -> str:
    return f"{t['or']:.2f} ({t['lo']:.2f}-{t['hi']:.2f})"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def load_age_series(cfg: Config, n_rows: int, raw_xlsx: Optional[Path] = None) -> Tuple[Optional[pd.Series], List[str]]:
    xlsx = raw_xlsx if raw_xlsx is not None else cfg.age_xlsx
    if xlsx.exists():
        raw = pd.read_excel(xlsx, sheet_name=0)
        raw = raw.loc[raw["age"] >= 12].reset_index(drop=True)
        if len(raw) == n_rows:
            return raw["age"], ["male", "age", "phq_high", "gad_high"]
        warnings.warn(f"{xlsx} row count ({len(raw)}) != cohort CSV ({n_rows}); omitting age from covariates.")
    else:
        warnings.warn(f"Age export not found at {xlsx}; adjusting for sex and parental mental health only.")
    return None, ["male", "phq_high", "gad_high"]


def run_association(
    cfg: Config,
    df: pd.DataFrame,
    cohort: str,
    raw_xlsx: Optional[Path] = None,
    exposures_file: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run crude & covariate-adjusted logistic regression (BH-FDR) and Spearman correlations."""
    cohort = cohort.lower()
    out = cfg.cohort_out_dir(cohort) / "association"
    out.mkdir(parents=True, exist_ok=True)

    age_series, covariates = load_age_series(cfg, len(df), raw_xlsx=raw_xlsx)
    frame = build_exposure_frame(df, cfg, cohort, age_series)
    specs = resolve_exposures(cfg, cohort, exposures_file=exposures_file)

    fits = {sp["label"]: _fit_logit(frame, sp["col"], sp["drop"], covariates) for sp in specs}
    q_vals = dict(zip(fits, multipletests([fits[k]["adjusted"]["p"] for k in fits], method="fdr_bh")[1]))

    assoc_rows, corr_rows = [], []
    for sp in specs:
        lbl = sp["label"]
        f = fits[lbl]
        assoc_rows.append({
            "Level": sp["level"],
            "Variable": lbl,
            "Contrast": sp["contrast"],
            "SHAP Rank (C/A/P)": sp["rank_str"],
            "N": f["n"],
            "N_cases": f["n_cases"],
            "Crude OR (95% CI)": _fmt_or(f["crude"]),
            "Adjusted OR (95% CI)": _fmt_or(f["adjusted"]),
            "p": _fmt_p(f["adjusted"]["p"]),
            "q (FDR)": _fmt_p(q_vals[lbl]),
            "crude_or": f["crude"]["or"],
            "adjusted_or": f["adjusted"]["or"],
            "adjusted_lo": f["adjusted"]["lo"],
            "adjusted_hi": f["adjusted"]["hi"],
            "adjusted_p_raw": f["adjusted"]["p"],
            "q_fdr_raw": float(q_vals[lbl]),
            "covariates": ",".join(c for c in covariates if c != sp["drop"]),
        })
        use = frame[["y", sp["col"]]].dropna()
        rho, p_rho = spearmanr(use["y"], use[sp["col"]])
        corr_rows.append({"Level": sp["level"], "Variable": lbl, "rho": float(rho), "p": float(p_rho), "n": int(len(use))})

    assoc_df = pd.DataFrame(assoc_rows)
    assert not assoc_df["Variable"].str.contains("adhd", case=False).any(), "ADHD must never enter association table"
    corr_df = pd.DataFrame(corr_rows).sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)

    assoc_df.to_csv(out / f"association_{cohort}.csv", index=False)
    corr_df.to_csv(out / f"correlation_{cohort}.csv", index=False)
    return assoc_df, corr_df
