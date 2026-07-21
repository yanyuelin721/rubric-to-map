# -*- coding: utf-8 -*-
"""
utils.py

Shared table, path, and score utilities for Rubric-to-Map.

Key features:
- project-root resolution (CLI > env var > auto-detect)
- robust ID column detection (supports Chinese column name "图片名")
- Excel/CSV loading helpers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


DEFAULT_ID_ALIASES: Tuple[str, ...] = ("图片名", "image_id", "image_name")
BASE_SCORE_COLS: Tuple[str, ...] = (
    "Q1_character",
    "Q2_enclosure",
    "Q3_walkability",
    "Q4_nature",
    "Q5_facade",
)
DEFAULT_SCORE_COLS: Tuple[str, ...] = (*BASE_SCORE_COLS, "Q_overall")


def compute_overall_score(scores: Mapping[str, object]) -> float:
    """Return the arithmetic mean of Q1–Q5; Q_overall is never an independent input."""
    return float(sum(float(scores[col]) for col in BASE_SCORE_COLS) / len(BASE_SCORE_COLS))


def add_derived_overall(
    df: pd.DataFrame,
    *,
    suffix: str = "",
    target_col: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Add or overwrite a Q_overall column derived from Q1–Q5 columns."""
    out = df.copy() if copy else df
    source_cols = [f"{col}{suffix}" for col in BASE_SCORE_COLS]
    missing = [col for col in source_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Cannot derive Q_overall; missing columns: {missing}")
    target = target_col or f"Q_overall{suffix}"
    numeric = out[source_cols].apply(pd.to_numeric, errors="raise")
    out[target] = numeric.mean(axis=1)
    return out


def get_project_root(cli_base_dir: Optional[str] = None) -> Path:
    """
    Resolve project root directory.

    Priority:
    1) CLI --base-dir
    2) env STREETLLM_BASE_DIR
    3) auto-detect: parent of this file's parent (repo_root/src/utils.py -> repo_root)
    """
    if cli_base_dir:
        return Path(cli_base_dir).expanduser().resolve()

    env = os.getenv("STREETLLM_BASE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    """
    Read an Excel or CSV/TSV file into a DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix in (".csv", ".tsv"):
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)

    raise ValueError(f"Unsupported file type: {path.suffix} (expected .xlsx/.xls/.csv/.tsv)")


def find_id_column(df: pd.DataFrame, id_col: Optional[str] = None,
                   aliases: Sequence[str] = DEFAULT_ID_ALIASES) -> str:
    """
    Find an ID column in df. If id_col is provided, it must exist.
    Otherwise, choose the first existing column from aliases.
    """
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found. Available columns: {list(df.columns)[:30]}")
        return id_col

    for c in aliases:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find an ID column. Tried: {list(aliases)}. "
        f"Available columns: {list(df.columns)[:30]}"
    )


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def must_have_columns(df: pd.DataFrame, cols: Iterable[str], table_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{table_name}] Missing required columns: {missing}. "
            f"Available columns (first 30): {list(df.columns)[:30]}"
        )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
