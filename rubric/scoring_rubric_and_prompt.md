# Scoring rubric and prompt note

In this project, the effective prompt text used for inference was directly derived from the scoring rubric.
For that reason, the exact operational prompts are embedded inside the public LLM scoring scripts rather than maintained as a separate prompt file.

Relevant scripts:
- `src/run_sre_scoring.py`
- `src/run_sre_sem_scoring.py`
- `src/run_sre_sem_level_scoring.py`
- `src/run_q3_only_scoring.py`

The released workflow applies three operational rules consistently:

- The model returns only `Q1_character`, `Q2_enclosure`, `Q3_walkability`, `Q4_nature`, and `Q5_facade` on a 0–10 scale.
- `Q_overall` is derived downstream as the arithmetic mean of those five dimension scores; it is not elicited or calibrated independently.
- The main scoring modes use three runs per image. Overall inferential statistics are computed after averaging runs within image, while run-level and stability summaries are retained separately.
- The `SRE-Sem-Level` variant uses four qualitative semantic priors (`very little`, `a little`, `a lot`, and `a great deal`) and does not expose the underlying numeric proportions to the model.

Semantic groups used for calibration are mutually exclusive: each retained semantic label can contribute to at most one of `G1_character`, `G2_enclosure`, `G3_walkability`, `G4_nature`, or `G5_facade_activity`.
