"""Loads the per-target locked-in recipe (configs/recipes.yaml) shared by every stage script."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_PATH = REPO_ROOT / "configs" / "recipes.yaml"


def load_recipes(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or RECIPES_PATH).read_text(encoding="utf-8"))


def get_recipe(target: str, path: Path | None = None) -> dict[str, Any]:
    target = target.upper()
    recipes = load_recipes(path)
    if target not in ("ODD", "DICCD"):
        raise ValueError(f"target must be 'ODD' or 'DICCD', got {target!r}")
    return recipes[target]


def get_cv_config(path: Path | None = None) -> dict[str, Any]:
    return load_recipes(path)["cv"]


def get_feature_sets(path: Path | None = None) -> dict[str, Any]:
    return load_recipes(path)["feature_sets"]


DATA_FILES = {
    "ODD": {"csv": "adolescents_data_ODD_all.csv", "metadata": "adolescents_data_odd_metadata_noADHD.json"},
    "DICCD": {"csv": "adolescents_data_DICCD_all.csv", "metadata": "adolescents_data_diccd_metadata_noADHD.json"},
}


def data_paths(target: str, data_dir: Path) -> tuple[Path, Path]:
    """Returns (csv_path, metadata_path) for a target under data_dir. See data/README.md
    for how to obtain this data (it is not included in this repository)."""
    target = target.upper()
    spec = DATA_FILES[target]
    return data_dir / spec["csv"], data_dir / spec["metadata"]
