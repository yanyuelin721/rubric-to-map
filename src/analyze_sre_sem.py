# -*- coding: utf-8 -*-
"""
analyze_sre_sem.py

Purpose
-------
Analyze SRE-Sem raw VLM scores (prompt includes semantic features) vs human reference.

Example
-------
python src/analyze_sre_sem.py \
  --llm sample_data/sample_sre_sem_scores_raw.xlsx \
  --human sample_data/sample_human_ratings.xlsx \
  --out sample_outputs/sample_sre_sem_reliability_summary.xlsx
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
    parser = argparse.ArgumentParser(description="Analyze SRE-Sem raw VLM scores (prompt includes semantic features) vs human reference.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--llm", type=str, default="sample_data/sample_sre_sem_scores_raw.xlsx")
    parser.add_argument("--human", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--human-fallback", type=str, default="sample_data/sample_human_ratings_fallback.xlsx")
    parser.add_argument("--out", type=str, default="sample_outputs/sample_sre_sem_reliability_summary.xlsx")
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--max-score", type=float, default=10.0)
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    llm_path = _resolve(args.llm)
    human_primary = _resolve(args.human)
    human_fallback = _resolve(args.human_fallback)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_llm = read_table(llm_path)
    llm_id = find_id_column(df_llm, id_col=args.id_col)
    must_have_columns(df_llm, [llm_id, "run_id", *DEFAULT_SCORE_COLS], "LLM_RAW")
    df_llm = df_llm[[llm_id, "run_id", *DEFAULT_SCORE_COLS]].copy()
    df_llm[llm_id] = normalize_id_series(df_llm[llm_id])
    print(f"Loaded LLM results: {llm_path} (rows={len(df_llm)})")

    df_human, human_id, used_human = load_human_scores(human_primary, human_fallback, id_col=args.id_col)
    print(f"Loaded human scores: {used_human} (rows={len(df_human)})")

    df_llm = df_llm.rename(columns={llm_id: "image_id"})
    df_human = df_human.rename(columns={human_id: "image_id"})
    df = pd.merge(df_llm, df_human, on="image_id", how="inner", suffixes=("_llm", "_human"))
    if df.empty:
        raise ValueError("No matched images after merge. Please check ID consistency.")
    print(f"Merged samples (including all runs): {len(df)}")

    # A) Agreement by run_id
    rows_by_run = []
    run_ids = sorted(df["run_id"].dropna().unique().tolist())
    for dim in DEFAULT_SCORE_COLS:
        h_col = f"{dim}_human"
        e_col = f"{dim}_llm"
        for run in run_ids:
            sub = df[df["run_id"] == run]
            x = sub[h_col].astype(float)
            y = sub[e_col].astype(float)
            mask = x.notna() & y.notna()
            x = x[mask]; y = y[mask]
            if len(x) < 3:
                continue
            pr, pr_p = pearsonr(x, y)
            sr, sr_p = spearmanr(x, y)
            leniency = compute_leniency(x.values, y.values, max_score=args.max_score)
            rows_by_run.append({
                "dimension": dim,
                "run_id": run,
                "n": int(len(x)),
                "pearson_r": float(pr),
                "pearson_p": float(pr_p),
                "spearman_rho": float(sr),
                "spearman_p": float(sr_p),
                "leniency_mean(E-H)_norm": float(leniency),
            })
    df_by_run = pd.DataFrame(rows_by_run)

    # B) Agreement across all runs
    rows_all = []
    for dim in DEFAULT_SCORE_COLS:
        h_col = f"{dim}_human"
        e_col = f"{dim}_llm"
        x = df[h_col].astype(float)
        y = df[e_col].astype(float)
        mask = x.notna() & y.notna()
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            continue
        pr, pr_p = pearsonr(x, y)
        sr, sr_p = spearmanr(x, y)
        leniency = compute_leniency(x.values, y.values, max_score=args.max_score)
        rows_all.append({
            "dimension": dim,
            "n": int(len(x)),
            "pearson_r": float(pr),
            "pearson_p": float(pr_p),
            "spearman_rho": float(sr),
            "spearman_p": float(sr_p),
            "leniency_mean(E-H)_norm": float(leniency),
        })
    df_all_runs = pd.DataFrame(rows_all)

    # C) Stability across runs (per-image SD and CV)
    grp = df_llm.groupby("image_id")[list(DEFAULT_SCORE_COLS)]
    sd = grp.std(ddof=0)
    mean = grp.mean()
    mean_safe = mean.replace(0, np.nan)
    cv_vals = sd.values / mean_safe.values
    cv = pd.DataFrame(cv_vals, index=sd.index, columns=[c + "_cv" for c in DEFAULT_SCORE_COLS])
    df_stab_per_image = pd.concat([sd.add_suffix("_sd"), cv], axis=1).reset_index()

    rows_stab = []
    for dim in DEFAULT_SCORE_COLS:
        sd_col = dim + "_sd"
        cv_col = dim + "_cv"
        rows_stab.append({
            "dimension": dim,
            "n_images": int(df_stab_per_image[sd_col].notna().sum()),
            "sd_mean": float(df_stab_per_image[sd_col].mean()),
            "sd_median": float(df_stab_per_image[sd_col].median()),
            "cv_mean": float(df_stab_per_image[cv_col].mean()),
            "cv_median": float(df_stab_per_image[cv_col].median()),
        })
    df_stab_summary = pd.DataFrame(rows_stab)

    with pd.ExcelWriter(out_path) as writer:
        df_by_run.to_excel(writer, sheet_name="sem_by_run", index=False)
        df_all_runs.to_excel(writer, sheet_name="sem_all_runs", index=False)
        df_stab_per_image.to_excel(writer, sheet_name="sem_stab_img", index=False)
        df_stab_summary.to_excel(writer, sheet_name="sem_stab_sum", index=False)

    print(f"Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
