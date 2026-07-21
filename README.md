# Rubric-to-Map

Rubric-to-Map provides Python scripts for rubric-guided vision-language-model scoring, semantic feature construction, score calibration, agreement analysis, and diagnostic checks for urban visual-quality studies.

## Contents

- `src/`: scoring, analysis, calibration, and diagnostic scripts
- `configs/`: reference settings and column conventions
- `rubric/`: scoring rubric and prompt documentation
- `requirements.txt`: Python dependencies

Images, rating tables, semantic-label tables, model outputs, and study results are not distributed. Supply your own files through the command-line arguments described below.

## Method definitions

The current implementation uses the following definitions consistently:

- The model returns five dimension scores: `Q1_character`, `Q2_enclosure`, `Q3_walkability`, `Q4_nature`, and `Q5_facade`.
- `Q_overall` is calculated as the arithmetic mean of the five dimension scores.
- Each retained semantic label belongs to at most one of the five semantic groups.
- The three main scoring modes default to three runs per image; inferential summaries average repeated runs within image.
- `SRE-Sem-Level` maps semantic proportions to four qualitative levels using cut points of 0.05, 0.20, and 0.40. The numeric proportions are not included in that prompt.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set an OpenRouter API key before running the model-scoring scripts:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Alternatively, copy `.env.example` to `.env` and add the key there.

## Expected inputs

Tabular inputs may be Excel (`.xlsx` or `.xls`), CSV, or TSV files. Image identifiers are detected from `image_id`, `image_name`, or `图片名`; use `--id-col` to specify another column.

Human and model score tables use the five dimension columns listed above. Raw model tables also include `run_id`. Semantic feature tables use:

- `G1_character`
- `G2_enclosure`
- `G3_walkability`
- `G4_nature`
- `G5_facade_activity`

## Main commands

Create grouped semantic features:

```bash
python src/build_semantic_groups.py \
  --input data/semantic_label_table.xlsx \
  --out-pruned outputs/semantic_pruned.xlsx \
  --out-groups outputs/semantic_features.xlsx
```

Run the three scoring variants:

```bash
python src/run_sre_scoring.py \
  --images data/images \
  --runs-per-image 3 \
  --out outputs/sre_scores_raw.xlsx

python src/run_sre_sem_scoring.py \
  --images data/images \
  --sem-feats data/semantic_features.xlsx \
  --runs-per-image 3 \
  --out outputs/sre_sem_scores_raw.xlsx

python src/run_sre_sem_level_scoring.py \
  --images data/images \
  --sem-feats data/semantic_features.xlsx \
  --runs-per-image 3 \
  --out outputs/sre_sem_level_scores_raw.xlsx
```

Analyze scores and run calibration:

```bash
python src/analyze_sre.py \
  --llm data/sre_scores_raw.xlsx \
  --human data/human_ratings.xlsx \
  --out outputs/sre_analysis.xlsx

python src/run_semantic_calibration.py \
  --llm data/sre_scores_raw.xlsx \
  --human data/human_ratings.xlsx \
  --feats data/semantic_features.xlsx \
  --out-scores outputs/calibrated_scores.xlsx \
  --out-summary outputs/calibration_summary.xlsx

python src/run_oof_score_calibration.py \
  --llm data/sre_scores_raw.xlsx \
  --human data/human_ratings.xlsx \
  --feats data/semantic_features.xlsx \
  --out-pred outputs/oof_predictions.xlsx \
  --out-summary outputs/oof_summary.xlsx
```

Use `python src/<script>.py --help` for the complete options of any script.

## Version note

This source-only release contains the current, harmonized implementation of the method. It is not a byte-for-byte snapshot of every script used during the study and cannot by itself regenerate the reported numerical results without the original study inputs.

## Citation

Repository: https://github.com/yanyuelin721/rubric-to-map

Author ORCID: https://orcid.org/0009-0006-2078-1299

## License

MIT License.
