# -*- coding: utf-8 -*-
"""
run_q3_only_scoring.py

Run a Q3-only inference variant that asks the model to score only the
walkability dimension (Q3_walkability).

This public version removes hard-coded local paths and keeps the exact prompt
embedded in the script, because in the original project the prompt text itself
was the experimental treatment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from llm_utils import collect_done_keys, encode_image_to_data_url, list_image_paths, load_existing_output, post_openrouter_json
from utils import get_project_root

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_RUNS_PER_IMAGE = 3
DEFAULT_OUT = "sample_outputs/sample_q3_only_scores_raw.xlsx"
DEFAULT_IMAGES = "sample_data/sample_images"

QS_RUBRIC_Q3 = """
You are an expert in urban design and environmental psychology. You only need to evaluate ONE thing:

> How does this streetscape perform on **Pedestrian Scale & Walking Comfort (Q3_walkability)**?

Please focus **only on Q3_walkability**. Do NOT consider other dimensions such as distinctiveness/character, enclosure, greenery/cleanliness, or facade aesthetics.
Even if the place looks overall attractive, do NOT inflate the Q3 score because of general visual quality.

[Q3_walkability: Pedestrian Scale & Walking Comfort (0–10)]

Pay special attention to the following aspects:

1) **Sidewalk presence and continuity**
   - Can you clearly see a sidewalk or dedicated pedestrian space?
   - Is it heavily occupied by vehicles, parking, or clutter?
   - Are pedestrians forced to walk along the edge of the vehicular carriageway (strong pedestrian–vehicle mixing)?

2) **Pedestrian–vehicle separation and safety**
   - Is there a clear physical or visual separation between pedestrians and vehicles (curb, planting strip, railing/bollards, etc.)?
   - Do vehicles frequently encroach on pedestrian space?

3) **Walking space scale and accessibility**
   - Is the sidewalk wide enough (e.g., for two people walking side by side or for multi-directional passing)?
   - Are corners, intersections, entrances/exits overly congested or blocked by obvious obstacles?

4) **Comfort-supporting amenities**
   - Is there shade (street trees, arcades, canopies), seating, and accessible ramps?
   - What is the paving quality—any damage, unevenness, or excessive level differences?

5) **Occlusion and robust inference**
   - In many street-view scenes, sidewalks may be partially occluded by trees, parked vehicles, planters, etc.
   - When occlusion occurs, make a **robust inference** based on the overall street structure:
     - If the carriageway width, building setbacks, and frontage layout strongly suggest that a continuous pedestrian space **should** exist, you may judge that a walkable pedestrian space is **likely present**, even if parts are blocked from view.
     - Conversely, if the road runs directly against buildings or walls, the space is extremely narrow, and there are no visible cues of pedestrian space, you should treat walkability as poor, even if some areas are occluded.
   - In short: **do not give 0–2 simply because “it is hard to see,” and do not give a very high score simply because “it probably should be there.”**
     Make a balanced judgment grounded in the visible overall spatial structure.

[Score anchors]

- **0–2 (Very poor):**
  - Sidewalks are essentially absent, or most pedestrians must walk next to moving traffic;
  - Severe pedestrian–vehicle mixing and frequent vehicle occupation of walking space;
  - Little to no shade, seating, or accessibility support; walking is clearly unsafe and uncomfortable.

- **3–5 (Fair):**
  - Sidewalks exist but are narrow, discontinuous, or frequently occupied (parking, stalls, obstacles);
  - Separation from traffic is unclear or locally broken, with some pedestrian–vehicle mixing;
  - Few amenities; the space is “walkable” in a minimal sense, but lingering quality is limited.

- **6–8 (Good):**
  - Sidewalks are appropriately wide and largely continuous, with clear separation from traffic;
  - Some parking/occlusion may exist but **does not seriously break overall continuity**;
  - Decent paving quality and a reasonable amount of shade, seating, wayfinding, or accessibility facilities; walking is comfortable and suitable for strolling.

- **9–10 (Excellent):**
  - Strong pedestrian priority (e.g., pedestrian streets or shared streets) with minimal conflicts;
  - Wide, continuous pedestrian space with rich human-scale design details (shopfront rhythm, window display scale, street furnishings);
  - Comprehensive shade, seating, and accessibility support that encourages staying, strolling, and social interaction.

Output ONLY the following JSON (no extra text):

{
  "Q3_walkability": float,
  "comment": "In 1–2 sentences, explain the main reasons for your walkability score."
}
""".strip()


def call_llm_for_image(*, image_path: Path, model: str, temperature: float, timeout: int) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是一名城市设计与环境心理学专家，现在只需要从街景图像中评估【人行尺度与步行舒适度】一个维度。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QS_RUBRIC_Q3},
                {"type": "text", "text": f"当前街景图片文件名：{image_path.name}。请只返回 Q3_walkability 和简要理由。"},
                {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
            ],
        },
    ]
    return post_openrouter_json(
        messages=messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
        title="streetLLM-Q3-only",
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
            "Q3_walkability": float(payload.get("Q3_walkability")),
            "comment": str(payload.get("comment", "")),
        }
        result["success"] = True
        result["row"] = row
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Q3-only scoring variant on a folder of images.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--runs-per-image", type=int, default=DEFAULT_RUNS_PER_IMAGE)
    parser.add_argument("--workers", type=int, default=4)
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

    pbar = tqdm(total=len(jobs), desc="Q3-only jobs", ncols=90) if args.progress_bar else None

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
