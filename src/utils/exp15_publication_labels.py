from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_LABELS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "exp15_publication_labels.json"
)

DEFAULT_TYPO_MAP = {
    "Depresive": "Depressive",
    "depresive": "depressive",
}

_SMALL = {"of", "and", "or", "the", "a", "an", "in", "on", "for", "to", "vs"}


def curate_display_name(
    mapping_word: str, typo_map: dict[str, str] | None = None
) -> str:
    text = str(mapping_word).strip()
    text = re.sub(r"/\s*\(주석\).*", "", text).strip()
    text = re.sub(r"/\s*\([^)]*주석[^)]*\).*", "", text).strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    text = text.rstrip(",").strip()
    for bad, good in (typo_map or DEFAULT_TYPO_MAP).items():
        text = text.replace(bad, good)
    words = re.split(r"(\s+)", text)
    out = []
    for i, w in enumerate(words):
        if w.isspace():
            out.append(w)
            continue
        low = w.lower()
        if i > 0 and low in _SMALL:
            out.append(low)
        else:
            out.append(w[:1].upper() + w[1:] if w else w)
    return "".join(out).strip()


def load_labels(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_LABELS_PATH
    return json.loads(Path(p).read_text(encoding="utf-8"))


def display_gender(code: str, labels: dict | None = None) -> str:
    lab = labels or load_labels()
    return lab["gender"][code]


def display_feature_type(code: str, labels: dict | None = None) -> str:
    lab = labels or load_labels()
    return lab["feature_type"][code]


def _pretty(raw: str) -> str:
    name = str(raw)
    if "__" in name:
        name = name.split("__", 1)[-1]
    return name


def _category_english(stem: str, suffix: str, labels: dict) -> str | None:
    """Resolve one-hot code → English category label (Exp-15 paper style)."""
    cats = (labels.get("categories") or {}).get(stem) or {}
    if suffix in cats:
        return str(cats[suffix])
    try:
        code = float(suffix)
    except (TypeError, ValueError):
        return None
    keys = {str(code)}
    if code == int(code):
        keys.add(str(int(code)))
        keys.add(f"{int(code)}.0")
    for key in keys:
        if key in cats:
            return str(cats[key])
    if code == 9999.0:
        return "Unknown/DK"
    if code == 9998.0:
        return "Other/None"
    return None


def display_feature(raw_name: str, labels: dict | None = None) -> str:
    """Map encoded column → publication label.

    One-hot categories use Exp-15 English style:
    ``Paternal Occupation (Service Worker)``, not ``Paternal Occupation [4.0]``.
    """
    lab = labels or load_labels()
    pretty = _pretty(raw_name)
    features = lab.get("features", {})
    aliases = lab.get("aliases", {})

    if pretty in features:
        return features[pretty]["display_name"]
    if pretty in aliases and aliases[pretty] in features:
        return features[aliases[pretty]]["display_name"]

    # Prefer known stems by longest key match (one-hot: STEM_cat)
    stem = None
    suffix = None
    for key in sorted(features.keys(), key=len, reverse=True):
        if pretty == key:
            return features[key]["display_name"]
        if pretty.startswith(key + "_"):
            stem, suffix = key, pretty[len(key) + 1 :]
            break
    if stem is not None:
        base = features[stem]["display_name"]
        # Binary/ordinal *_code / *_total suffixes → stem display only
        if suffix in {"code", "total"}:
            return base
        cat_en = _category_english(stem, suffix, lab)
        if cat_en:
            return f"{base} ({cat_en})"
        if re.fullmatch(r"-?\d+(?:\.\d+)?", str(suffix)):
            return f"{base} ({suffix})"
        return f"{base} [{suffix}]"

    key2 = aliases.get(pretty)
    if key2 and key2 in features:
        return features[key2]["display_name"]

    return pretty


def remap_legacy_population(text: str, labels: dict | None = None) -> str:
    lab = labels or load_labels()
    t = str(text).strip()
    mapping = {
        "All": lab["gender"]["all"],
        "All Adolescents": lab["gender"]["all"],
        "All Adolescents (Male/Female)": lab["gender"]["all"],
        "Male": lab["gender"]["man"],
        "Male Adolescents": lab["gender"]["man"],
        "Female": lab["gender"]["lady"],
        "Female Adolescents": lab["gender"]["lady"],
    }
    return mapping.get(t, t)


def remap_legacy_feature_set(text: str, labels: dict | None = None) -> str:
    lab = labels or load_labels()
    t = str(text).strip()
    mapping = {
        "Combined": lab["feature_type"]["all"],
        "All Features": lab["feature_type"]["all"],
        "Adolescent": lab["feature_type"]["adolescent"],
        "Parent": lab["feature_type"]["parent"],
    }
    return mapping.get(t, t)
