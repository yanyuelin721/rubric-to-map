# -*- coding: utf-8 -*-
"""
check_key_mismatch.py

Purpose
-------
Quickly inspect ID mismatches across:
- human score table
- semantic feature table
- llm score table

Example
-------
python src/check_key_mismatch.py \
  --human sample_data/sample_human_ratings.xlsx \
  --feats sample_data/sample_semantic_features.xlsx \
  --llm sample_data/sample_sre_scores_raw.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import find_id_column, get_project_root, normalize_id_series, read_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ID mismatches across tables.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--human", type=str, default="sample_data/sample_human_ratings.xlsx")
    parser.add_argument("--feats", type=str, default="sample_data/sample_semantic_features.xlsx")
    parser.add_argument("--llm", type=str, default="sample_data/sample_sre_scores_raw.xlsx")
    parser.add_argument("--id-col", type=str, default=None)
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    human_path = _resolve(args.human)
    feat_path = _resolve(args.feats)
    llm_path = _resolve(args.llm)

    human = read_table(human_path)
    feat = read_table(feat_path)
    llm = read_table(llm_path)

    hk = find_id_column(human, id_col=args.id_col)
    fk = find_id_column(feat, id_col=args.id_col)
    lk = find_id_column(llm, id_col=args.id_col)

    for df, k in ((human, hk), (feat, fk), (llm, lk)):
        df[k] = normalize_id_series(df[k])

    hs = set(human[hk].unique())
    fs = set(feat[fk].unique())
    ls = set(llm[lk].unique())

    print("unique counts (human, feats, llm):", len(hs), len(fs), len(ls))
    print("human not in feats:", sorted(list(hs - fs))[:50])
    print("human not in llm :", sorted(list(hs - ls))[:50])
    print("feats not in human:", sorted(list(fs - hs))[:50])
    print("llm not in human :", sorted(list(ls - hs))[:50])

    inter = hs & fs & ls
    union = hs | fs | ls
    print("intersection:", len(inter))
    print("union:", len(union))
    print("missing_from_intersection:", len(union) - len(inter))


if __name__ == "__main__":
    main()
