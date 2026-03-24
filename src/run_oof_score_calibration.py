# -*- coding: utf-8 -*-
"""
run_oof_score_calibration.py

Purpose
-------
Out-of-fold (OOF) calibration for raw LLM scores using semantic group features (G1–G5).
This script follows a 5-fold stratified scheme (by default stratified on human Q_overall quantile bins).

Example
-------
python src/run_oof_score_calibration.py --n-splits 5 --seed 42 --n-bins 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

from utils import (
    DEFAULT_SCORE_COLS,
    find_id_column,
    get_project_root,
    must_have_columns,
    normalize_id_series,
    read_table,
)

SEM_RENAME = {
    "G1_character": "G1",
    "G2_enclosure": "G2",
    "G3_walkability": "G3",
    "G4_nature": "G4",
    "G5_facade_activity": "G5",
}

FEATURE_SUBSET: Dict[str, List[str]] = {
    "Q1_character": ["G1", "G5"],
    "Q2_enclosure": ["G2", "G1"],
    "Q3_walkability": ["G3", "G2", "G4"],
    "Q4_nature": ["G4"],
    "Q5_facade": ["G1", "G5"],
    "Q_overall": ["G1", "G2", "G3", "G4", "G5"],
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pr, pr_p = pearsonr(y_true, y_pred)
    sr, sr_p = spearmanr(y_true, y_pred)
    leniency = float(np.nanmean(y_pred - y_true))
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pr_p),
        "spearman_rho": float(sr),
        "spearman_p": float(sr_p),
        "leniency_mean(pred-true)": float(leniency),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF calibration using semantic group features.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--human", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--human-fallback", type=str, default="sample_data/sample_human_ratings_fallback.xlsx")
    parser.add_argument("--feats", type=str, default="sample_data/sample_semantic_features.xlsx")
    parser.add_argument("--llm", type=str, default="sample_data/sample_sre_scores_raw.xlsx")
    parser.add_argument("--out-pred", type=str, default="sample_outputs/sample_oof_predictions.xlsx")
    parser.add_argument("--out-summary", type=str, default="sample_outputs/sample_oof_summary.xlsx")
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--clip", type=float, nargs=2, default=(0.0, 10.0), metavar=("MIN", "MAX"))
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    human_primary = _resolve(args.human)
    human_fallback = _resolve(args.human_fallback)
    feat_path = _resolve(args.feats)
    llm_path = _resolve(args.llm)
    out_pred = _resolve(args.out_pred); out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_summary = _resolve(args.out_summary); out_summary.parent.mkdir(parents=True, exist_ok=True)

    # Load tables
    human_path = human_primary if human_primary.exists() else human_fallback
    if not human_path.exists():
        raise FileNotFoundError("Human score file not found (primary and fallback).")

    human = read_table(human_path)
    feats = read_table(feat_path)
    llm = read_table(llm_path)

    hk = find_id_column(human, id_col=args.id_col)
    fk = find_id_column(feats, id_col=args.id_col)
    lk = find_id_column(llm, id_col=args.id_col)

    must_have_columns(human, [hk, *DEFAULT_SCORE_COLS], "HUMAN")
    must_have_columns(feats, [fk, *SEM_RENAME.keys()], "FEATURES")
    must_have_columns(llm, [lk, *DEFAULT_SCORE_COLS], "LLM_RAW")

    # Normalize IDs
    human[hk] = normalize_id_series(human[hk])
    feats[fk] = normalize_id_series(feats[fk])
    llm[lk] = normalize_id_series(llm[lk])

    # Prepare semantic features
    feats = feats[[fk, *SEM_RENAME.keys()]].rename(columns=SEM_RENAME).rename(columns={fk: "image_id"})

    # Aggregate LLM runs
    if args.agg == "mean":
        llm_agg = llm.groupby(lk, as_index=False)[list(DEFAULT_SCORE_COLS)].mean()
    else:
        llm_agg = llm.groupby(lk, as_index=False)[list(DEFAULT_SCORE_COLS)].median()
    llm_agg = llm_agg.rename(columns={lk: "image_id"})
    llm_agg = llm_agg.rename(columns={d: f"{d}_LLM" for d in DEFAULT_SCORE_COLS})

    # Human rename
    human = human[[hk, *DEFAULT_SCORE_COLS]].rename(columns={hk: "image_id"})
    human = human.rename(columns={d: f"{d}_human" for d in DEFAULT_SCORE_COLS})

    # Merge
    df = llm_agg.merge(feats, on="image_id", how="inner").merge(human, on="image_id", how="inner")
    if df.empty:
        raise ValueError("No samples after merge. Please check image IDs.")
    print("Merged N =", len(df))

    # Stratification bins on human Q_overall
    df["_bin"] = pd.qcut(df["Q_overall_human"], args.n_bins, labels=False, duplicates="drop")

    # If the dataset is small (e.g., the included sample data), adjust n_splits automatically.
    class_counts = df["_bin"].value_counts()
    min_class = int(class_counts.min())
    if args.n_splits > min_class:
        if min_class < 2:
            raise ValueError(
                "Not enough samples per stratification bin to run CV. "
                "Try reducing --n-bins or use a larger dataset."
            )
        print(f"⚠️ Adjusting n_splits from {args.n_splits} to {min_class} (limited by smallest bin).")
        args.n_splits = min_class

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for d in DEFAULT_SCORE_COLS:
        df[f"{d}_cal_OOF"] = np.nan

    clip_min, clip_max = args.clip

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df["_bin"]), start=1):
        tr = df.iloc[tr_idx]
        te = df.iloc[te_idx]
        for d in DEFAULT_SCORE_COLS:
            x_cols = [f"{d}_LLM"] + FEATURE_SUBSET[d]
            Xtr = tr[x_cols].astype(float).values
            ytr = tr[f"{d}_human"].astype(float).values
            Xte = te[x_cols].astype(float).values
            model = LinearRegression()
            model.fit(Xtr, ytr)
            pred = np.clip(model.predict(Xte), clip_min, clip_max)
            df.iloc[te_idx, df.columns.get_loc(f"{d}_cal_OOF")] = pred
        print(f"Fold {fold} done.")

    # Summary
    rows = []
    for d in DEFAULT_SCORE_COLS:
        y = df[f"{d}_human"].astype(float).values
        raw = df[f"{d}_LLM"].astype(float).values
        cal = df[f"{d}_cal_OOF"].astype(float).values
        m_raw = compute_metrics(y, raw)
        m_cal = compute_metrics(y, cal)
        rows.append({
            "dimension": d,
            "n": int(len(df)),
            "pearson_r_raw": m_raw["pearson_r"],
            "pearson_r_cal_OOF": m_cal["pearson_r"],
            "spearman_rho_raw": m_raw["spearman_rho"],
            "spearman_rho_cal_OOF": m_cal["spearman_rho"],
            "leniency_raw": m_raw["leniency_mean(pred-true)"],
            "leniency_cal_OOF": m_cal["leniency_mean(pred-true)"],
            "delta_pearson_r": m_cal["pearson_r"] - m_raw["pearson_r"],
            "delta_leniency": m_cal["leniency_mean(pred-true)"] - m_raw["leniency_mean(pred-true)"],
        })

    summary = pd.DataFrame(rows)

    keep_cols = (
        ["image_id", "G1", "G2", "G3", "G4", "G5"] +
        [f"{d}_LLM" for d in DEFAULT_SCORE_COLS] +
        [f"{d}_human" for d in DEFAULT_SCORE_COLS] +
        [f"{d}_cal_OOF" for d in DEFAULT_SCORE_COLS]
    )
    with pd.ExcelWriter(out_pred, engine="openpyxl") as w:
        df[keep_cols].to_excel(w, index=False, sheet_name="oof_predictions")
    with pd.ExcelWriter(out_summary, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")

    print("OOF predictions ->", out_pred)
    print("Summary        ->", out_summary)


if __name__ == "__main__":
    main()