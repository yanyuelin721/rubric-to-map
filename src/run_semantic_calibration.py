# -*- coding: utf-8 -*-
"""
run_semantic_calibration.py

Purpose
-------
Fit a linear backend calibrator for each dimension:

    human ≈ a0 + a1 * llm + Σ a_i * semantic_group_i

Steps
-----
1) Aggregate raw LLM scores to one row per image (median across runs)
2) Merge with semantic group features (G1–G5) and human reference scores
3) Fit least-squares regression per dimension
4) Apply calibration to all images (LLM + semantic)
5) Save calibrated scores and a coefficient/metric summary

Example
-------
python src/run_semantic_calibration.py --dim all
python src/run_semantic_calibration.py --dim Q3_walkability --clip 0 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from utils import (
    DEFAULT_SCORE_COLS,
    find_id_column,
    get_project_root,
    must_have_columns,
    normalize_id_series,
    read_table,
)

SEM_COLS = [
    "G1_character",
    "G2_enclosure",
    "G3_walkability",
    "G4_nature",
    "G5_facade_activity",
]

SEM_FEATURES_BY_DIM: Dict[str, List[str]] = {
    "Q1_character": ["G1_character", "G5_facade_activity"],
    "Q2_enclosure": ["G2_enclosure", "G1_character"],
    "Q3_walkability": ["G3_walkability", "G2_enclosure", "G4_nature"],
    "Q4_nature": ["G4_nature"],
    "Q5_facade": ["G1_character", "G5_facade_activity"],
    "Q_overall": ["G1_character", "G2_enclosure", "G3_walkability", "G4_nature", "G5_facade_activity"],
}


def compute_leniency(h: np.ndarray, e: np.ndarray, max_score: float = 10.0) -> float:
    h_hat = h / max_score
    e_hat = e / max_score
    return float(np.nanmean(e_hat - h_hat))


def fit_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def load_human(primary: Path, fallback: Path | None, id_col: str | None) -> pd.DataFrame:
    path = primary if primary.exists() else fallback
    if path is None or not path.exists():
        raise FileNotFoundError(f"Human score file not found: {primary} (and fallback {fallback})")
    df = read_table(path)
    idc = find_id_column(df, id_col=id_col)
    must_have_columns(df, [idc, *DEFAULT_SCORE_COLS], "HUMAN")
    df = df[[idc, *DEFAULT_SCORE_COLS]].copy()
    df[idc] = normalize_id_series(df[idc])
    return df.rename(columns={idc: "image_id"})


def load_llm_median(llm_path: Path, id_col: str | None) -> pd.DataFrame:
    df = read_table(llm_path)
    idc = find_id_column(df, id_col=id_col)
    must_have_columns(df, [idc, *DEFAULT_SCORE_COLS], "LLM_RAW")
    df = df[[idc, *DEFAULT_SCORE_COLS]].copy()
    df[idc] = normalize_id_series(df[idc])
    med = df.groupby(idc)[list(DEFAULT_SCORE_COLS)].median().reset_index()
    med = med.rename(columns={c: f"{c}_llm" for c in DEFAULT_SCORE_COLS})
    return med.rename(columns={idc: "image_id"})


def load_sem(sem_path: Path, id_col: str | None) -> pd.DataFrame:
    df = read_table(sem_path)
    idc = find_id_column(df, id_col=id_col)
    cols = [c for c in SEM_COLS if c in df.columns]
    if not cols:
        raise ValueError(f"Semantic table missing expected columns: {SEM_COLS}")
    df = df[[idc, *cols]].copy()
    df[idc] = normalize_id_series(df[idc])
    return df.rename(columns={idc: "image_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend calibration using semantic group features.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--llm", type=str, default="sample_data/sample_cre_scores_raw.xlsx")
    parser.add_argument("--human", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--human-fallback", type=str, default="sample_data/sample_human_ratings_fallback.xlsx")
    parser.add_argument("--feats", type=str, default="sample_data/sample_semantic_features.xlsx")
    parser.add_argument("--out-scores", type=str, default="sample_outputs/sample_calibrated_scores.xlsx")
    parser.add_argument("--out-summary", type=str, default="sample_outputs/sample_calibration_summary.xlsx")
    parser.add_argument("--dim", type=str, default="all")
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--max-score", type=float, default=10.0)
    parser.add_argument("--clip", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    llm_path = _resolve(args.llm)
    human_primary = _resolve(args.human)
    human_fallback = _resolve(args.human_fallback)
    sem_path = _resolve(args.feats)
    out_scores = _resolve(args.out_scores); out_scores.parent.mkdir(parents=True, exist_ok=True)
    out_summary = _resolve(args.out_summary); out_summary.parent.mkdir(parents=True, exist_ok=True)

    df_human = load_human(human_primary, human_fallback, id_col=args.id_col)
    df_llm = load_llm_median(llm_path, id_col=args.id_col)
    df_sem = load_sem(sem_path, id_col=args.id_col)

    df_all = df_llm.merge(df_sem, on="image_id", how="left")
    df_calib = df_all.merge(df_human, on="image_id", how="inner")

    if args.dim == "all":
        dims = list(DEFAULT_SCORE_COLS)
    else:
        if args.dim not in DEFAULT_SCORE_COLS:
            raise ValueError(f"Unknown dim: {args.dim}")
        dims = [args.dim]

    summary_rows = []
    clip_rng = tuple(args.clip) if args.clip is not None else None

    for dim in dims:
        llm_col = f"{dim}_llm"
        human_col = dim
        sem_feats = [c for c in SEM_FEATURES_BY_DIM.get(dim, []) if c in df_calib.columns]
        cols = [llm_col] + sem_feats
        df_fit = df_calib[[human_col] + cols].dropna()
        if len(df_fit) < 10:
            print(f"[{dim}] Not enough valid samples (n={len(df_fit)}), skip.")
            continue

        y = df_fit[human_col].astype(float).values
        X = df_fit[cols].astype(float).values

        r_before, p_before = pearsonr(y, df_fit[llm_col].astype(float).values)
        len_before = compute_leniency(y, df_fit[llm_col].astype(float).values, max_score=args.max_score)

        beta = fit_lstsq(X, y)
        intercept, coefs = float(beta[0]), beta[1:]
        y_hat = intercept + X @ coefs
        if clip_rng is not None:
            y_hat = np.clip(y_hat, clip_rng[0], clip_rng[1])

        r_after, p_after = pearsonr(y, y_hat)
        len_after = compute_leniency(y, y_hat, max_score=args.max_score)

        # apply to all
        X_all = df_all[cols].astype(float).values
        y_all = intercept + X_all @ coefs
        if clip_rng is not None:
            y_all = np.clip(y_all, clip_rng[0], clip_rng[1])
        df_all[f"{dim}_calibrated"] = y_all

        row = {
            "dimension": dim,
            "n_samples": int(len(df_fit)),
            "features_used": ",".join(cols),
            "intercept": intercept,
            "r_before": float(r_before),
            "p_before": float(p_before),
            "r_after": float(r_after),
            "p_after": float(p_after),
            "leniency_before_norm": float(len_before),
            "leniency_after_norm": float(len_after),
        }
        for name, c in zip(cols, coefs):
            row[f"coef_{name}"] = float(c)
        summary_rows.append(row)

        print(f"[{dim}] r_before={r_before:.3f} -> r_after={r_after:.3f}; len={len_before:.3f} -> {len_after:.3f}")

    df_all.to_excel(out_scores, index=False)
    print("Saved calibrated scores ->", out_scores)

    if summary_rows:
        pd.DataFrame(summary_rows).to_excel(out_summary, index=False)
        print("Saved calibration summary ->", out_summary)
    else:
        print("No summary produced (no dimension calibrated).")


if __name__ == "__main__":
    main()
