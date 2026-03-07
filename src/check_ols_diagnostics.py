# -*- coding: utf-8 -*-
"""
check_ols_diagnostics.py

Purpose
-------
Run basic OLS diagnostics for the relationship between semantic group features (G1–G5)
and a target human score (default: Q_overall).

Outputs
-------
- Correlation matrix (xlsx)
- VIF table (xlsx)
- Residual diagnostic plots (png)
- Cook's distance table (xlsx)

Example
-------
python src/check_ols_diagnostics.py \
  --features sample_data/sample_semantic_features.xlsx \
  --scores sample_data/sample_human_ratings.xlsx \
  --target Q_overall
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import find_id_column, get_project_root, must_have_columns, normalize_id_series, read_table


FEATURE_COLS = [
    "G1_character",
    "G2_enclosure",
    "G3_walkability",
    "G4_nature",
    "G5_facade_activity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="OLS diagnostics for semantic features.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--features", type=str, default="sample_data/sample_semantic_features.xlsx")
    parser.add_argument("--scores", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--target", type=str, default="Q_overall")
    parser.add_argument("--out-dir", type=str, default="sample_outputs/diagnostics")
    parser.add_argument("--fig-dir", type=str, default="figures/diagnostics")
    parser.add_argument("--id-col", type=str, default=None)
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    feature_path = _resolve(args.features)
    score_path = _resolve(args.scores)
    out_dir = _resolve(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = _resolve(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    feat_df = read_table(feature_path)
    score_df = read_table(score_path)

    fid = find_id_column(feat_df, id_col=args.id_col)
    sid = find_id_column(score_df, id_col=args.id_col)

    must_have_columns(feat_df, [fid, *FEATURE_COLS], "FEATURES")
    must_have_columns(score_df, [sid, args.target], "SCORES")

    feat_df[fid] = normalize_id_series(feat_df[fid])
    score_df[sid] = normalize_id_series(score_df[sid])

    df = pd.merge(
        feat_df[[fid, *FEATURE_COLS]],
        score_df[[sid, args.target]],
        left_on=fid, right_on=sid, how="inner"
    )
    if df.empty:
        raise ValueError("No samples after merge. Please check image IDs.")

    X = df[FEATURE_COLS].astype(float)
    y = df[args.target].astype(float)

    # 1) Correlation matrix
    corr = X.corr()
    corr_out = out_dir / "ols_corr_matrix.xlsx"
    corr.to_excel(corr_out)

    # 2) VIF
    vif_df = pd.DataFrame({"variable": FEATURE_COLS})
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_out = out_dir / "ols_vif.xlsx"
    vif_df.to_excel(vif_out, index=False)

    # 3) Fit OLS
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(model.summary())

    # 4) Residual plots
    resid = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, resid, alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(fig_dir / "ols_resid_vs_fitted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(fig_dir / "ols_resid_hist.png", dpi=150)
    plt.close()

    fig = sm.qqplot(resid, line="45", fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.savefig(fig_dir / "ols_resid_qqplot.png", dpi=150)
    plt.close(fig)

    # 5) Cook's distance
    infl = model.get_influence()
    cooks_d, _ = infl.cooks_distance
    leverage = infl.hat_matrix_diag
    student_resid = infl.resid_studentized_external

    cooks_df = pd.DataFrame({
        "image_id": df[fid],
        "cooks_d": cooks_d,
        "leverage": leverage,
        "studentized_resid": student_resid,
    })
    cooks_out = out_dir / "ols_cooks_distance.xlsx"
    cooks_df.to_excel(cooks_out, index=False)

    n = len(df)
    threshold = 4 / n
    n_large = int((cooks_df["cooks_d"] > threshold).sum())
    print(f"n={n}, Cook's threshold 4/n≈{threshold:.4f}, points above threshold={n_large}")
    print("Saved:", corr_out, vif_out, cooks_out, fig_dir)


if __name__ == "__main__":
    main()
