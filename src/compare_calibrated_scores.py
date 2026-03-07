# -*- coding: utf-8 -*-
"""
compare_calibrated_scores.py

Purpose
-------
Compare raw (LLM) vs calibrated scores against human reference.

Input (default)
---------------
- sample_outputs/sample_calibrated_scores.xlsx
  expected columns: image_id (or 图片名) and for each dimension:
    {dim}_llm, {dim}_calibrated

- sample_data/sample_human_ratings.xlsx (or fallback sample_data/sample_human_ratings_fallback.xlsx)

Output (default)
----------------
- sample_outputs/sample_raw_vs_calibrated_comparison.xlsx

Example
-------
python src/compare_calibrated_scores.py --calib sample_outputs/sample_calibrated_scores.xlsx --human sample_data/sample_human_ratings.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils import (
    DEFAULT_SCORE_COLS,
    find_id_column,
    get_project_root,
    must_have_columns,
    normalize_id_series,
    read_table,
)


def compute_leniency(h_scores: np.ndarray, e_scores: np.ndarray, max_score: float = 10.0) -> float:
    h_hat = h_scores / max_score
    e_hat = e_scores / max_score
    return float(np.nanmean(e_hat - h_hat))


def load_human_scores(primary: Path, fallback: Path | None, id_col: str | None) -> tuple[pd.DataFrame, str, Path]:
    path = primary if primary.exists() else fallback
    if path is None or not path.exists():
        raise FileNotFoundError(f"Human score file not found: {primary} (and fallback {fallback})")
    df = read_table(path)
    idc = find_id_column(df, id_col=id_col)
    must_have_columns(df, [idc, *DEFAULT_SCORE_COLS], "HUMAN")
    df = df[[idc, *DEFAULT_SCORE_COLS]].copy()
    df[idc] = normalize_id_series(df[idc])
    return df, idc, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw vs calibrated scores.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--calib", type=str, default="sample_outputs/sample_calibrated_scores.xlsx")
    parser.add_argument("--human", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--human-fallback", type=str, default="sample_data/sample_human_ratings_fallback.xlsx")
    parser.add_argument("--out", type=str, default="sample_outputs/sample_raw_vs_calibrated_comparison.xlsx")
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--max-score", type=float, default=10.0)
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    calib_path = _resolve(args.calib)
    human_primary = _resolve(args.human)
    human_fallback = _resolve(args.human_fallback)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_human, hid, used_human = load_human_scores(human_primary, human_fallback, id_col=args.id_col)
    df_human = df_human.rename(columns={hid: "image_id"})

    df_cal = read_table(calib_path)
    cid = find_id_column(df_cal, id_col=args.id_col)
    needed = [cid]
    for dim in DEFAULT_SCORE_COLS:
        needed += [f"{dim}_llm", f"{dim}_calibrated"]
    must_have_columns(df_cal, needed, "CALIBRATED_SCORES")
    df_cal = df_cal[needed].copy()
    df_cal[cid] = normalize_id_series(df_cal[cid])
    df_cal = df_cal.rename(columns={cid: "image_id"})

    df = df_human.merge(df_cal, on="image_id", how="inner")
    if df.empty:
        raise ValueError("No matched images after merge. Please check ID consistency.")

    rows = []
    for dim in DEFAULT_SCORE_COLS:
        h = df[dim].astype(float)
        raw = df[f"{dim}_llm"].astype(float)
        cal = df[f"{dim}_calibrated"].astype(float)

        mask = h.notna() & raw.notna() & cal.notna()
        h = h[mask]; raw = raw[mask]; cal = cal[mask]
        if len(h) < 3:
            continue

        pr_raw, pr_raw_p = pearsonr(h, raw)
        sr_raw, sr_raw_p = spearmanr(h, raw)
        len_raw = compute_leniency(h.values, raw.values, max_score=args.max_score)

        pr_cal, pr_cal_p = pearsonr(h, cal)
        sr_cal, sr_cal_p = spearmanr(h, cal)
        len_cal = compute_leniency(h.values, cal.values, max_score=args.max_score)

        rows.append({
            "dimension": dim,
            "n": int(len(h)),
            "pearson_raw": float(pr_raw),
            "pearson_raw_p": float(pr_raw_p),
            "pearson_calib": float(pr_cal),
            "pearson_calib_p": float(pr_cal_p),
            "spearman_raw": float(sr_raw),
            "spearman_raw_p": float(sr_raw_p),
            "spearman_calib": float(sr_cal),
            "spearman_calib_p": float(sr_cal_p),
            "leniency_raw_norm(E-H)": float(len_raw),
            "leniency_calib_norm(E-H)": float(len_cal),
            "delta_pearson": float(pr_cal - pr_raw),
            "delta_leniency": float(len_cal - len_raw),
        })

    summary = pd.DataFrame(rows)

    with pd.ExcelWriter(out_path) as writer:
        summary.to_excel(writer, sheet_name="compare_raw_vs_calib", index=False)
        df.to_excel(writer, sheet_name="merged_detail", index=False)

    print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    main()
