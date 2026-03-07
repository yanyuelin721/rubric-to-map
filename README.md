# Street Quality Reproducibility Package

This repository is a public reproducibility package for a street-quality assessment study based on rubric-constrained VLM scoring, semantic-feature-assisted prompting, and backend score calibration.

## Public release scope
This repository includes only the materials that are safe to share publicly:
- analysis scripts
- calibration scripts
- LLM scoring scripts
- prompt/rubric text
- configuration templates
- small sample input tables
- small sample output tables

This repository does not include:
- original street-view images
- platform-restricted raw image collections
- API keys or private credentials
- the full private project dataset

## Repository structure
- `src/`: executable Python scripts
- `configs/`: lightweight configuration templates and naming conventions
- `rubric/`: combined scoring rubric and prompt note
- `sample_data/`: small demonstration input tables and a placeholder folder for shareable sample images
- `sample_outputs/`: example output files generated from the public workflow
- `figures/`: placeholder folder for diagnostic figures and visual outputs

## Main scripts
### LLM scoring
- `src/run_cre_scoring.py`
- `src/run_cre_sem_scoring.py`
- `src/run_cre_sem_level_scoring.py`
- `src/run_q3_only_scoring.py`

### Analysis and calibration
- `src/run_oof_score_calibration.py`
- `src/run_semantic_calibration.py`
- `src/build_semantic_groups.py`
- `src/analyze_cre.py`
- `src/analyze_cre_sem.py`
- `src/analyze_cre_sem_level.py`
- `src/compare_calibrated_scores.py`
- `src/check_key_mismatch.py`
- `src/check_ols_diagnostics.py`

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run the LLM scoring scripts, set your API key first:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

A local `.env` file is also supported when `python-dotenv` is installed.

## Minimal runnable public demo
The commands below can be executed with the public sample tables included in this repository.

```bash
python src/build_semantic_groups.py --input sample_data/sample_semantic_label_table.xlsx --out-pruned sample_outputs/sample_semantic_pruned.xlsx --out-groups sample_outputs/sample_semantic_features_from_labels.xlsx
python src/check_key_mismatch.py --human sample_data/sample_human_ratings.xlsx --feats sample_data/sample_semantic_features.xlsx --llm sample_data/sample_cre_scores_raw.xlsx
python src/analyze_cre.py --llm sample_data/sample_cre_scores_raw.xlsx --human sample_data/sample_human_ratings.xlsx --out sample_outputs/sample_cre_reliability_summary.xlsx
python src/analyze_cre_sem.py --llm sample_data/sample_cre_sem_scores_raw.xlsx --human sample_data/sample_human_ratings.xlsx --out sample_outputs/sample_cre_sem_reliability_summary.xlsx
python src/analyze_cre_sem_level.py --llm sample_data/sample_cre_sem_level_scores_raw.xlsx --human sample_data/sample_human_ratings.xlsx --out sample_outputs/sample_cre_sem_level_reliability_summary.xlsx
python src/run_semantic_calibration.py --llm sample_data/sample_cre_scores_raw.xlsx --human sample_data/sample_human_ratings.xlsx --feats sample_data/sample_semantic_features.xlsx
python src/run_oof_score_calibration.py --llm sample_data/sample_cre_scores_raw.xlsx --human sample_data/sample_human_ratings.xlsx --feats sample_data/sample_semantic_features.xlsx
python src/compare_calibrated_scores.py --calib sample_outputs/sample_calibrated_scores.xlsx --human sample_data/sample_human_ratings.xlsx
python src/check_ols_diagnostics.py --features sample_data/sample_semantic_features.xlsx --scores sample_data/sample_human_ratings.xlsx --target Q_overall
```

## Structural examples requiring images and API access
The public package does not include shareable sample street-view images. The commands below are retained to document the expected workflow, but they require image inputs and API access.

```bash
python src/run_cre_scoring.py --images sample_data/sample_images --out sample_outputs/sample_cre_scores_raw.xlsx
python src/run_cre_sem_scoring.py --images sample_data/sample_images --sem-feats sample_data/sample_semantic_features.xlsx --out sample_outputs/sample_cre_sem_scores_raw.xlsx
python src/run_cre_sem_level_scoring.py --images sample_data/sample_images --sem-feats sample_data/sample_semantic_features.xlsx --out sample_outputs/sample_cre_sem_level_scores_raw.xlsx
python src/run_q3_only_scoring.py --images sample_data/sample_images --out sample_outputs/sample_q3_only_scores_raw.xlsx
```

## Sample table note
The sample tables and outputs are small illustrative examples intended to show file structure, column conventions, and expected outputs. They should not be interpreted as the full private dataset, and they are not guaranteed to form a single end-to-end mini-pipeline.

## ID column conventions
Scripts automatically detect image identifiers from the following column names:
- `image_id`
- `image_name`
- `图片名`

Use `--id-col` if you want to force a specific ID column.

## Prompt and rubric note
In the original project, the scoring prompt was directly derived from the evaluation rubric. For that reason, the exact prompts remain embedded in the LLM scoring scripts, and a combined note is also provided in:
- `rubric/scoring_rubric_and_prompt.md`

The preserved prompt wording is Chinese and study-context-specific. Reuse in other cities, languages, or scoring settings may require prompt adaptation.

## Configuration note
The YAML files in `configs/` are lightweight public templates for naming conventions and reference settings. In the current public package, most scripts are driven directly by CLI arguments rather than auto-loading those YAML files.

## Placeholder directories
Two directories are intentionally kept in the public package as placeholders:
- `sample_data/sample_images/` is reserved for future shareable example images.
- `figures/` is reserved for future diagnostic plots and visual outputs.

Each placeholder directory contains a short `README.md` explaining its purpose.

## Notes for public reuse
This package is designed to be understandable to external readers. Compared with the private working version, it uses:
- English file names
- consistent script naming
- CLI arguments instead of hard-coded local paths
- resumable outputs where relevant
- small shareable sample tables instead of the full private data

## Citation
If you use this repository, please cite the associated paper or repository release in the format required by your venue.

Author ORCID: https://orcid.org/0009-0006-2078-1299

## License
MIT License.