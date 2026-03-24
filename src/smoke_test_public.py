# -*- coding: utf-8 -*-
"""
smoke_test_public.py

Run a lightweight smoke test for the public reproducibility package.
This script executes a minimal set of commands that should work with the
included public sample tables.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMANDS = [
    [sys.executable, 'src/build_semantic_groups.py', '--input', 'sample_data/sample_semantic_label_table.xlsx', '--out-pruned', 'sample_outputs/sample_semantic_pruned.xlsx', '--out-groups', 'sample_outputs/sample_semantic_features_from_labels.xlsx'],
    [sys.executable, 'src/check_key_mismatch.py', '--human', 'sample_data/sample_human_ratings.xlsx', '--feats', 'sample_data/sample_semantic_features.xlsx', '--llm', 'sample_data/sample_sre_scores_raw.xlsx'],
    [sys.executable, 'src/analyze_sre.py', '--llm', 'sample_data/sample_sre_scores_raw.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx', '--out', 'sample_outputs/sample_sre_reliability_summary.xlsx'],
    [sys.executable, 'src/analyze_sre_sem.py', '--llm', 'sample_data/sample_sre_sem_scores_raw.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx', '--out', 'sample_outputs/sample_sre_sem_reliability_summary.xlsx'],
    [sys.executable, 'src/analyze_sre_sem_level.py', '--llm', 'sample_data/sample_sre_sem_level_scores_raw.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx', '--out', 'sample_outputs/sample_sre_sem_level_reliability_summary.xlsx'],
    [sys.executable, 'src/run_semantic_calibration.py', '--llm', 'sample_data/sample_sre_scores_raw.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx', '--feats', 'sample_data/sample_semantic_features.xlsx'],
    [sys.executable, 'src/run_oof_score_calibration.py', '--llm', 'sample_data/sample_sre_scores_raw.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx', '--feats', 'sample_data/sample_semantic_features.xlsx'],
    [sys.executable, 'src/compare_calibrated_scores.py', '--calib', 'sample_outputs/sample_calibrated_scores.xlsx', '--human', 'sample_data/sample_human_ratings.xlsx'],
    [sys.executable, 'src/check_ols_diagnostics.py', '--features', 'sample_data/sample_semantic_features.xlsx', '--scores', 'sample_data/sample_human_ratings.xlsx', '--target', 'Q_overall'],
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for cmd in COMMANDS:
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, cwd=root, check=True)
    print('All public smoke tests passed.')


if __name__ == '__main__':
    main()
