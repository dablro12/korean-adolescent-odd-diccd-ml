"""Configuration loading (configs/experiment.yaml, configs/features.yaml)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

FEATURE_SET_LABEL = {"combined": "Combined", "adolescent": "Adolescent", "parent": "Parent"}
TARGET_LABEL = {"odd": "ODD", "diccd": "DICCD"}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    seed: int
    data_dir: Path
    results_dir: Path
    age_xlsx_name: str
    cohorts: Dict[str, dict]
    feature_sets: List[str]
    outer_folds: int
    inner_folds: int
    outer_seed: int
    inner_seed: int
    models: Dict[str, bool]
    samplers: List[str]
    unstable_cluster_samplers: List[str]
    losses: List[str]
    focal_gamma: float
    hparam_grid: dict
    optuna_trials: int
    locked_recipes: Dict[str, dict]
    importance: dict
    robustness: dict
    features: dict = field(repr=False)

    @property
    def sex_column(self) -> str:
        return self.features["sex_column"]

    @property
    def age_xlsx(self) -> Path:
        return self.data_dir / self.age_xlsx_name

    def target_col(self, cohort: str) -> str:
        return self.cohorts[cohort.lower()]["target"]

    def cohort_csv(self, cohort: str) -> Path:
        return self.data_dir / self.cohorts[cohort.lower()]["csv"]

    def cohort_out_dir(self, cohort: str) -> Path:
        return self.results_dir / cohort.lower()

    def recipe(self, cohort: str) -> dict:
        return self.locked_recipes[cohort.lower()]

    def feature_columns(self, feature_set: str) -> List[str]:
        groups = self.features["groups"]
        cols: List[str] = []
        for g in self.features["feature_sets"][feature_set]:
            cols.extend(groups[g])
        return cols

    @property
    def all_feature_columns(self) -> List[str]:
        return self.feature_columns("combined")

    def type_lists(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self.features["types"].items()}

    def domain_of(self, raw_or_encoded: str) -> str:
        raw = raw_or_encoded.split("__", 1)[-1] if "__" in raw_or_encoded else raw_or_encoded
        parent_set = set(self.features["groups"]["parent"])
        if raw in parent_set:
            return "Parent"
        for p in sorted(parent_set, key=len, reverse=True):
            if raw.startswith(p + "_"):
                return "Parent"
        return "Adolescent"

    def label(self, raw_or_encoded: str) -> str:
        """Resolve a raw variable or one-hot encoded feature name to its publication label."""
        pretty = raw_or_encoded.split("__", 1)[-1] if "__" in raw_or_encoded else str(raw_or_encoded)
        labels = self.features["labels"]
        cats = self.features.get("categories") or {}
        if pretty in labels:
            return labels[pretty]["label"]
        for stem in sorted(labels.keys(), key=len, reverse=True):
            if pretty.startswith(stem + "_"):
                suffix = pretty[len(stem) + 1:]
                base = labels[stem]["label"]
                if suffix in {"code", "total"}:
                    return base
                stem_cats = cats.get(stem) or {}
                if suffix in stem_cats:
                    return f"{base} ({stem_cats[suffix]})"
                try:
                    code_f = float(suffix)
                    if code_f == int(code_f):
                        k_int = str(int(code_f))
                        if k_int in stem_cats:
                            return f"{base} ({stem_cats[k_int]})"
                        if code_f == 9999.0:
                            return f"{base} (Unknown/DK)"
                        if code_f == 9998.0:
                            return f"{base} (Other/None)"
                except ValueError:
                    pass
                if re.fullmatch(r"-?\d+(?:\.\d+)?", suffix):
                    return f"{base} ({suffix})"
                return f"{base} [{suffix}]"
        return pretty


def load_config(
    experiment_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Config:
    exp = _load_yaml(Path(experiment_path or CONFIG_DIR / "experiment.yaml"))
    feats = _load_yaml(Path(features_path or CONFIG_DIR / "features.yaml"))

    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    paths = exp["paths"]
    cv = exp["cv"]
    return Config(
        seed=int(exp["seed"]),
        data_dir=Path(data_dir) if data_dir else _resolve(paths["data_dir"]),
        results_dir=Path(results_dir) if results_dir else _resolve(paths["results_dir"]),
        age_xlsx_name=str(paths["age_xlsx"]),
        cohorts=exp["cohorts"],
        feature_sets=list(exp["feature_sets"]),
        outer_folds=int(cv["outer_folds"]),
        inner_folds=int(cv["inner_folds"]),
        outer_seed=int(cv["outer_seed"]),
        inner_seed=int(cv["inner_seed"]),
        models={k: bool(v) for k, v in exp["models"].items()},
        samplers=list(exp["samplers"]),
        unstable_cluster_samplers=list(exp["unstable_cluster_samplers"]),
        losses=list(exp["losses"]),
        focal_gamma=float(exp["focal_gamma"]),
        hparam_grid=exp["hparam_grid"],
        optuna_trials=int(exp["optuna"]["n_trials"]),
        locked_recipes=exp["locked_recipes"],
        importance=exp["importance"],
        robustness=exp["robustness"],
        features=feats,
    )
