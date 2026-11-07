"""Shared helpers for the pipeline stages (paths, coin IDs, crops, JSONL, overrides)."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, Iterator, List

import cv2
import numpy as np
from dotenv import load_dotenv

# The one results run every stage works on (see docs/PLAN.md, Phase 0)
DEFAULT_RUN = "results/2025_11_16_15_40_15"
IMG_DIR = "images"
OVERRIDES_CSV = "overrides.csv"

load_dotenv()

Item = Dict[str, Any]


def coin_id(item: Item) -> str:
    return f"{item['pair_key']}_coin{item['coin_number']}"


def load_coins(run_dir: str) -> List[Item]:
    """Extraction entries of a run, with group_id if clustering has been run."""
    grouped = os.path.join(run_dir, "extraction_results_with_groups.json")
    path = (
        grouped
        if os.path.exists(grouped)
        else os.path.join(run_dir, "extraction_results.json")
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["coin_id"] = coin_id(item)
    return data


_image_cache: Dict[str, np.ndarray] = {}


def crop_side(item: Item, side: str, max_size: int | None = None) -> np.ndarray:
    """Crop 'front' or 'back' of a coin from the original scan (BGR)."""
    name = item[f"{side}_image"]
    if name not in _image_cache:
        if len(_image_cache) > 4:  # scans are large; keep only a few in memory
            _image_cache.clear()
        img = cv2.imread(os.path.join(IMG_DIR, name))
        if img is None:
            raise FileNotFoundError(os.path.join(IMG_DIR, name))
        _image_cache[name] = img
    rect = item[f"{side}_rect"]
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    roi = _image_cache[name][max(y, 0) : y + h, max(x, 0) : x + w]
    if max_size and max(roi.shape[:2]) > max_size:
        scale = max_size / max(roi.shape[:2])
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return roi


def read_jsonl(path: str) -> List[Item]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, rows: Iterable[Item]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_overrides(path: str = OVERRIDES_CSV) -> Iterator[Dict[str, str]]:
    """Rows of overrides.csv (coin_id, field, value, source, note); later rows win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def apply_overrides(rows_by_coin: Dict[str, Item], path: str = OVERRIDES_CSV) -> int:
    """Apply overrides in place. Returns how many were applied."""
    n = 0
    for row in iter_overrides(path):
        target = rows_by_coin.get(row["coin_id"])
        if target is None:
            continue
        value: Any = row["value"]
        if value == "":
            value = None
        elif row["field"] in ("denomination_value", "est_value_eur"):
            value = float(value)
        elif row["field"] in ("numista_type_id", "numista_issue_id"):
            value = int(value)
        elif row["field"] in ("front_is_obverse", "exclude"):
            value = value.strip().lower() in ("1", "true", "yes")
        target[row["field"]] = value
        target.setdefault("overridden", []).append(row["field"])
        n += 1
    return n
