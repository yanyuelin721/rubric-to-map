# -*- coding: utf-8 -*-
"""
build_semantic_groups.py

Purpose
-------
Create semantic feature groups (G1–G5) from a per-image semantic label table.
This is a lightweight, transparent grouping step used as backend calibration features.

Input (default)
---------------
- data/semantic_label_table.xlsx (one row per image; semantic labels are columns)

Outputs (default)
-----------------
- outputs/semantic_pruned.xlsx (pruned semantic table)
- outputs/semantic_features.xlsx (G1–G5 grouped feature table)

Notes
-----
- The semantic label columns depend on your segmentation/tagger output.
- This script drops all-zero columns and a small set of indoor/object-like labels (editable).

Example
-------
python src/build_semantic_groups.py --input data/semantic_label_table.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import find_id_column, get_project_root, normalize_id_series, read_table


DEFAULT_INDOOR_LIKE = [
    "bed", "cabinet", "table", "shelf", "pooltable", "pillow", "sink",
    "towel", "bottle", "bag", "ball", "plate", "monitor", "crtscreen",
    "blanket", "pot", "barrel", "tent", "computer", "food",
    "dishwasher", "microwave", "oven", "sofa", "coffeetable",
]


GROUPS = {
    "G1_character": [
        "tower", "sculpture", "fountain", "flag", "stage", "tradename",
        "signboard", "poster", "bulletinboard", "clock", "painting",
    ],
    "G2_enclosure": [
        "building", "house", "skyscraper", "wall", "column", "bridge",
        "road", "sky",
    ],
    "G3_walkability": [
        "sidewalk", "path", "dirttrack", "stairs", "stairway", "step",
        "bench", "seat", "chair", "stool", "railing", "fence", "bannister",
        "pole", "streetlight", "lamp", "light", "trafficlight",
        "awning", "booth", "base", "ashcan", "plaything",
    ],
    "G4_nature": [
        "tree", "grass", "plant", "flower", "palm", "field", "hill",
        "sand", "mountain", "earth", "water", "river", "sea", "rock",
    ],
    "G5_facade_activity": [
        "windowpane", "door", "glass", "person", "apparel",
    ],
}


def validate_exclusive_groups() -> None:
    """Fail fast if any retained semantic label is assigned to more than one group."""
    owners: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for group_name, labels in GROUPS.items():
        for label in labels:
            if label in owners:
                duplicates.setdefault(label, [owners[label]]).append(group_name)
            else:
                owners[label] = group_name
    if duplicates:
        raise ValueError(f"Semantic groups must be mutually exclusive; duplicates: {duplicates}")


def main() -> None:
    validate_exclusive_groups()
    parser = argparse.ArgumentParser(description="Construct semantic group features (G1–G5).")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--input", type=str, default="data/semantic_label_table.xlsx")
    parser.add_argument("--out-pruned", type=str, default="outputs/semantic_pruned.xlsx")
    parser.add_argument("--out-groups", type=str, default="outputs/semantic_features.xlsx")
    parser.add_argument("--id-col", type=str, default=None)
    args = parser.parse_args()

    root = get_project_root(args.base_dir)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    in_path = _resolve(args.input)
    out_pruned = _resolve(args.out_pruned)
    out_groups = _resolve(args.out_groups)
    out_pruned.parent.mkdir(parents=True, exist_ok=True)
    out_groups.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(in_path)
    id_col = find_id_column(df, id_col=args.id_col)
    df[id_col] = normalize_id_series(df[id_col])

    # 1) Drop all-zero columns (excluding ID)
    cols_to_drop_zero = []
    for col in df.columns:
        if col == id_col:
            continue
        if (df[col].fillna(0) == 0).all():
            cols_to_drop_zero.append(col)

    df_pruned = df.drop(columns=cols_to_drop_zero)

    # 2) Drop indoor/object-like labels (editable)
    cols_to_drop_indoor = [c for c in DEFAULT_INDOOR_LIKE if c in df_pruned.columns]
    if cols_to_drop_indoor:
        df_pruned = df_pruned.drop(columns=cols_to_drop_indoor)

    df_pruned.to_excel(out_pruned, index=False)
    print(f"Saved pruned semantic table: {out_pruned}")

    # 3) Compute grouped features
    group_df = df_pruned[[id_col]].copy()
    for gname, labels in GROUPS.items():
        valid = [c for c in labels if c in df_pruned.columns]
        if not valid:
            group_df[gname] = 0.0
        else:
            group_df[gname] = df_pruned[valid].sum(axis=1)

    group_df = group_df.rename(columns={id_col: "image_id"})
    group_df.to_excel(out_groups, index=False)
    print(f"Saved grouped features (G1–G5): {out_groups}")


if __name__ == "__main__":
    main()
