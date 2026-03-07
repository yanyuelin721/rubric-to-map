# -*- coding: utf-8 -*-
"""
run_cre_sem_scoring.py

Run the CRE-Sem variant: image-based scoring with additional numeric semantic
feature ratios (G1-G5) injected as auxiliary prompt information.

The prompt remains embedded in the script because the original project treated
prompt wording as part of the experimental design.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from llm_utils import (
    SEM_COLS,
    collect_done_keys,
    encode_image_to_data_url,
    list_image_paths,
    load_existing_output,
    post_openrouter_json,
    read_semantic_feature_map,
)
from utils import get_project_root

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_RUNS_PER_IMAGE = 1
DEFAULT_OUT = "sample_outputs/sample_cre_sem_scores_raw.xlsx"
DEFAULT_IMAGES = "sample_data/sample_images"
DEFAULT_SEM = "sample_data/sample_semantic_features.xlsx"

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


def build_semantic_text(image_id: str, sem_row: Dict[str, float]) -> str:
    if not sem_row:
        return (
            f"当前街景图片文件名：{image_id}。\n"
            f"未找到该图的语义要素统计表，请主要依据图像本身进行评估。"
        )

    g1 = float(sem_row.get("G1_character", 0.0))
    g2 = float(sem_row.get("G2_enclosure", 0.0))
    g3 = float(sem_row.get("G3_walkability", 0.0))
    g4 = float(sem_row.get("G4_nature", 0.0))
    g5 = float(sem_row.get("G5_facade_activity", 0.0))
    built = g1 + g2 + g5
    walk = g3
    nature = g4
    total = built + walk + nature

    if total > 0:
        ratio_text = (
            f"从相对构成来看，G1+G2+G5（建筑与街道界面要素）≈{built / total:.2f}，"
            f"G3（步行空间）≈{walk / total:.2f}，G4（自然要素）≈{nature / total:.2f}。"
            f"这些比例反映的是“建筑/界面–步行空间–自然要素”三类之间的客观视觉平衡，"
            f"通常比单个 G 值的绝对大小更能说明空间格局。"
        )
    else:
        ratio_text = "整体语义占比较低，请主要依据图像本身进行综合判断。"

    return (
        f"当前街景图片文件名：{image_id}。\n"
        f"根据语义分割统计，该视角下五个语义要素的像素占比分别为：\n"
        f"- G1_character：{g1:.3f}\n"
        f"- G2_enclosure：{g2:.3f}\n"
        f"- G3_walkability：{g3:.3f}\n"
        f"- G4_nature：{g4:.3f}\n"
        f"- G5_facade_activity：{g5:.3f}\n"
        f"{ratio_text}\n"
        f"请结合这些比例关系和图像细节来理解空间结构，但仍然以图像的整体视觉印象为主进行评分。"
    )


def call_llm_for_image(*, image_path: Path, sem_row: Optional[Dict[str, float]], model: str, temperature: float, timeout: int) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是一名城市设计与环境心理专家，负责根据街景图像和语义要素进行多维度质量评价。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QS_RUBRIC_TEXT},
                {"type": "text", "text": build_semantic_text(image_path.name, sem_row or {})},
                {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
            ],
        },
    ]
    return post_openrouter_json(
        messages=messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
        title="streetLLM-CRE-Sem",
    )


def worker(job: Tuple[Path, int, Optional[Dict[str, float]], str, float, int]) -> Dict[str, Any]:
    image_path, run_id, sem_row, model, temperature, timeout = job
    image_id = image_path.name
    result: Dict[str, Any] = {"success": False, "image_id": image_id, "run_id": run_id, "row": None, "error": ""}
    try:
        payload = call_llm_for_image(image_path=image_path, sem_row=sem_row, model=model, temperature=temperature, timeout=timeout)
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
    parser = argparse.ArgumentParser(description="Run the CRE-Sem scoring variant on a folder of images.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES)
    parser.add_argument("--sem-feats", type=str, default=DEFAULT_SEM)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--runs-per-image", type=int, default=DEFAULT_RUNS_PER_IMAGE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column name in the semantic feature table.")
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    root = get_project_root(args.base_dir)
    images_dir = (root / args.images).resolve() if not Path(args.images).is_absolute() else Path(args.images).resolve()
    sem_path = (root / args.sem_feats).resolve() if not Path(args.sem_feats).is_absolute() else Path(args.sem_feats).resolve()
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(images_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"No image files were found in: {images_dir}")
        return

    sem_map = read_semantic_feature_map(sem_path, id_col=args.id_col, sem_cols=SEM_COLS) if sem_path.exists() else {}
    if sem_path.exists():
        print(f"Loaded semantic feature rows for {len(sem_map)} images from: {sem_path}")
    else:
        print(f"Semantic feature file not found: {sem_path}. The script will continue with image-only prompts.")

    df_out = load_existing_output(out_path)
    done_keys = collect_done_keys(df_out)
    if not df_out.empty:
        print(f"Found existing output file: {out_path} ({len(df_out)} rows). Resume mode is enabled.")

    jobs: List[Tuple[Path, int, Optional[Dict[str, float]], str, float, int]] = []
    for image_path in image_paths:
        sem_row = sem_map.get(image_path.name)
        for run_id in range(1, max(1, args.runs_per_image) + 1):
            if (image_path.name, run_id) not in done_keys:
                jobs.append((image_path, run_id, sem_row, args.model, args.temperature, args.timeout))

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

    pbar = tqdm(total=len(jobs), desc="CRE-Sem jobs", ncols=90) if args.progress_bar else None

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
