# -*- coding: utf-8 -*-
"""
run_sre_scoring.py

Run rubric-constrained visual quality scoring on a folder of street-view images
using an OpenRouter-compatible multimodal model.

This public version removes hard-coded local paths and supports:
- CLI path configuration
- resume from an existing output file
- parallel requests
- repeated runs per image
- optional progress bar

Notes
-----
The exact evaluation prompt is embedded in this script because the original
project used the rubric text directly as the inference prompt.

Example
-------
python src/run_sre_scoring.py \
  --images sample_data/sample_images \
  --out sample_outputs/sample_sre_scores_raw.xlsx \
  --runs-per-image 1 --workers 3 --progress-bar
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from llm_utils import (
    collect_done_keys,
    encode_image_to_data_url,
    list_image_paths,
    load_existing_output,
    post_openrouter_json,
)
from utils import get_project_root

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_RUNS_PER_IMAGE = 1
DEFAULT_OUT = "sample_outputs/sample_sre_scores_raw.xlsx"
DEFAULT_IMAGES = "sample_data/sample_images"

QS_RUBRIC_TEXT = """
You are an expert in urban design and environmental psychology. Based on the provided street-view images from the Wuhan Tiandi district (Wuhan, Hubei Province, China), evaluate the location's overall urban visual quality.

Please score the following 5 dimensions on a 0–10 scale, where 0 is extremely poor and 10 is excellent.
Do NOT restrict yourself to integers or .5 scores (e.g., 6.0 or 7.5). Instead, use two-decimal continuous scores whenever possible (e.g., 6.23, 7.84) to make the evaluation more fine-grained.

[Q1_character: Distinctiveness & Sense of Place]
- Low (0–2): No salient identity cues; generic/ordinary/chaotic; not recognizable as Wuhan Tiandi; vague spatial structure without clear nodes/landmarks/theme.
- Mid (3–5): One or two memorable elements (building/sculpture/signage), but weak overall coherence; reads as a typical mixed-use area; Wuhan Tiandi vibe not clear.
- High (6–8): Coherent and recognizable character (e.g., historic + modern frontage, riverfront ambience); multiple nodes/signature elements; strong sense of place.
- Very high (9–10): Highly iconic; city-level landmarks or uniquely recognizable composition; clear axes/nodes; strong thematic narrative.

[Q2_enclosure: Spatial Enclosure & Openness]
- Low (0–2): Extremely narrow or extremely exposed; highly imbalanced height-to-width; strong oppression/emptiness; views blocked by walls/hoardings/barriers.
- Mid (3–5): Generally acceptable proportions but locally too tight/open; partial continuity; many obstructions; ambiguous edges.
- High (6–8): Well-balanced; clear street walls from buildings/trees; legible sightlines; comfortable enclosure with some sky/river/distant view.
- Very high (9–10): Rhythmic sequence from inner streets to plazas/riverfront; exceptionally clear spatial skeleton; well-organized corridors/nodes/edges.

[Q3_walkability: Pedestrian Scale & Walking Comfort]
- Low (0–2): Sidewalk absent/occupied; strong pedestrian-vehicle mixing; unsafe/uncomfortable; lacking seating/shade/accessibility.
- Mid (3–5): Sidewalk present but narrow/discontinuous/obstructed; average micro-design; limited amenities; mainly for passing through.
- High (6–8): Adequate width and continuity; separated from traffic; good paving; pedestrian-scale details; seating/shade/wayfinding/accessibility reasonably provided.
- Very high (9–10): Strong pedestrian priority; minimal conflicts; rich human-scale details; comprehensive amenities encouraging strolling and lingering.

[Q4_nature: Greenery, Environmental Quality & Cleanliness]
- Low (0–2): Sparse/deteriorated greenery; litter/stains/disorder; damaged facilities; messy cables/rough hoardings; “dirty and chaotic.”
- Mid (3–5): Some greenery but limited/discontinuous; moderate maintenance; local litter/damage/parking disorder; average water/river scenic quality if present.
- High (6–8): Abundant and well-maintained greenery; clean pavements; intact facilities; orderly parking/operations; comfortable environment.
- Very high (9–10): Highly integrated nature and urban space; rich layers; open river/water views; meticulous maintenance; excellent order and quality.

[Q5_facade: Facade Detail & Visual Complexity]
- Low (0–2): Extremely monotonous or chaotic; uncontrolled signage/temporary structures/cables; clashing materials/colors; too little or too noisy information.
- Mid (3–5): Basic elements exist but weak layering; ordinary or mildly messy palette; signage lacks control; moderate but not refined complexity.
- High (6–8): Balanced variation and clear layering; signage aligns with architecture; reasonable color control; appropriate information density and rhythm.
- Very high (9–10): Rich but ordered details; refined materials/colors; seamless integration of signage/shopfronts/architecture; strong aesthetics and recognizability.

After reading the rubric carefully, make your judgment and output ONLY the following JSON:

{
  "Q1_character": float,
  "Q2_enclosure": float,
  "Q3_walkability": float,
  "Q4_nature": float,
  "Q5_facade": float,
  "Q_overall": float
}
""".strip()


def call_llm_for_image(*, image_path: Path, model: str, temperature: float, timeout: int) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是一名城市设计与环境心理专家，负责根据街景图像进行多维度质量评价。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QS_RUBRIC_TEXT},
                {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
            ],
        },
    ]
    return post_openrouter_json(
        messages=messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
        title="streetLLM-SRE",
    )


def worker(job: Tuple[Path, int, str, float, int]) -> Dict[str, Any]:
    image_path, run_id, model, temperature, timeout = job
    image_id = image_path.name
    result: Dict[str, Any] = {"success": False, "image_id": image_id, "run_id": run_id, "row": None, "error": ""}
    try:
        payload = call_llm_for_image(image_path=image_path, model=model, temperature=temperature, timeout=timeout)
        row = {
            "image_id": image_id,
            "model": model,
            "run_id": run_id,
            "Q1_character": float(payload.get("Q1_character")),
            "Q2_enclosure": float(payload.get("Q2_enclosure")),
            "Q3_walkability": float(payload.get("Q3_walkability")),
            "Q4_nature": float(payload.get("Q4_nature")),
            "Q5_facade": float(payload.get("Q5_facade")),
            "Q_overall": float(payload.get("Q_overall")),
            "comment": str(payload.get("comment", "")),
        }
        result["success"] = True
        result["row"] = row
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SRE scoring on a folder of images.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--runs-per-image", type=int, default=DEFAULT_RUNS_PER_IMAGE)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    root = get_project_root(args.base_dir)
    images_dir = (root / args.images).resolve() if not Path(args.images).is_absolute() else Path(args.images).resolve()
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(images_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"No image files were found in: {images_dir}")
        return

    df_out = load_existing_output(out_path)
    done_keys = collect_done_keys(df_out)
    if not df_out.empty:
        print(f"Found existing output file: {out_path} ({len(df_out)} rows). Resume mode is enabled.")

    jobs: List[Tuple[Path, int, str, float, int]] = []
    for image_path in image_paths:
        for run_id in range(1, max(1, args.runs_per_image) + 1):
            if (image_path.name, run_id) not in done_keys:
                jobs.append((image_path, run_id, args.model, args.temperature, args.timeout))

    if not jobs:
        print("No new jobs to run. All image/run combinations already exist in the output file.")
        return

    print(f"Queued {len(jobs)} jobs for {len(image_paths)} images (runs_per_image={max(1, args.runs_per_image)}).")
    buffer_rows: List[Dict[str, Any]] = []

    def flush_to_disk() -> None:
        nonlocal df_out, buffer_rows
        if not buffer_rows:
            return
        df_new = pd.DataFrame(buffer_rows)
        df_out = pd.concat([df_out, df_new], ignore_index=True) if not df_out.empty else df_new
        df_out.to_excel(out_path, index=False)
        buffer_rows = []

    pbar = tqdm(total=len(jobs), desc="SRE jobs", ncols=90) if args.progress_bar else None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_job = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(future_to_job):
            image_path, run_id, *_ = future_to_job[future]
            try:
                res = future.result()
            except Exception as exc:
                msg = f"[ERROR] {image_path.name} run {run_id}: {exc}"
            else:
                if res["success"]:
                    buffer_rows.append(res["row"])
                    msg = f"[OK] {image_path.name} run {run_id}"
                else:
                    msg = f"[FAILED] {image_path.name} run {run_id}: {res['error']}"

            if pbar is not None:
                pbar.write(msg)
                pbar.update(1)
            else:
                print(msg)

            if args.flush_every > 0 and len(buffer_rows) >= args.flush_every:
                flush_to_disk()

    flush_to_disk()
    if pbar is not None:
        pbar.close()
    print(f"Done. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
