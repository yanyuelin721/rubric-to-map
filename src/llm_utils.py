# -*- coding: utf-8 -*-
"""
llm_utils.py

Utility helpers for the public StreetLLM reproducibility package.
These functions are used by the OpenRouter-based scoring scripts.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from utils import find_id_column, normalize_id_series, read_table

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
SEM_COLS: Tuple[str, ...] = (
    "G1_character",
    "G2_enclosure",
    "G3_walkability",
    "G4_nature",
    "G5_facade_activity",
)


def list_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES],
        key=lambda p: p.name,
    )


def encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def post_openrouter_json(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    timeout: int,
    title: str,
    referer: str = "https://example.com/streetllm",
) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in your shell environment, or place it in a local .env file.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text[:300]}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc

    obj = extract_json_object(content)
    if obj is None:
        raise RuntimeError(f"Could not parse JSON from model output: {content[:300]}")
    return obj


def read_semantic_feature_map(sem_path: Path, id_col: Optional[str] = None,
                              sem_cols: Sequence[str] = SEM_COLS) -> Dict[str, Dict[str, float]]:
    if not sem_path.exists():
        return {}
    df = read_table(sem_path)
    idc = find_id_column(df, id_col=id_col)
    cols = [c for c in sem_cols if c in df.columns]
    if not cols:
        raise ValueError(f"Semantic feature table does not contain expected columns: {list(sem_cols)}")
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df[[idc, *cols]].iterrows():
        key = str(row[idc]).strip()
        out[key] = {c: float(row[c]) for c in cols}
    return out


def load_existing_output(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    return read_table(output_path)


def collect_done_keys(df_out: pd.DataFrame, id_aliases: Iterable[str] = ("image_id", "图片名", "image_name")) -> set:
    if df_out.empty or "run_id" not in df_out.columns:
        return set()
    id_col = None
    for c in id_aliases:
        if c in df_out.columns:
            id_col = c
            break
    if id_col is None:
        return set()
    ids = normalize_id_series(df_out[id_col])
    runs = pd.to_numeric(df_out["run_id"], errors="coerce")
    done = set()
    for img, run in zip(ids, runs):
        if pd.notna(run):
            done.add((str(img), int(run)))
    return done
