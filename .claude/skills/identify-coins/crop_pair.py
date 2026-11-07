"""Render front+back of coins side by side so they can be looked at in one image.

Usage:
    uv run python .claude/skills/identify-coins/crop_pair.py <coin_id> [<coin_id> ...]
        [--run results/2025_11_16_15_40_15] [--out <dir>] [--size 512]
    uv run python .claude/skills/identify-coins/crop_pair.py --group 17   # all coins of a cluster

Prints one output path per coin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

DEFAULT_RUN = "results/2025_11_16_15_40_15"
IMG_DIR = "images"


def coin_id(item: dict) -> str:
    return f"{item['pair_key']}_coin{item['coin_number']}"


def crop(image_name: str, rect: dict[str, int], size: int) -> np.ndarray:
    img = cv2.imread(os.path.join(IMG_DIR, image_name))
    if img is None:
        raise FileNotFoundError(os.path.join(IMG_DIR, image_name))
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    roi = img[max(y, 0) : y + h, max(x, 0) : x + w]
    return cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coin_ids", nargs="*")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument(
        "--group", type=int, help="render every coin with this group_id"
    )
    parser.add_argument(
        "--out", default=None, help="output dir (default: <run>/review_crops)"
    )
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    grouped = os.path.join(args.run, "extraction_results_with_groups.json")
    json_path = (
        grouped
        if os.path.exists(grouped)
        else os.path.join(args.run, "extraction_results.json")
    )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {coin_id(item): item for item in data}

    if args.group is not None:
        wanted = [coin_id(i) for i in data if i.get("group_id") == args.group]
    else:
        wanted = args.coin_ids
    if not wanted:
        parser.error("give coin ids or --group")

    out_dir = args.out or os.path.join(args.run, "review_crops")
    os.makedirs(out_dir, exist_ok=True)

    missing = 0
    for cid in wanted:
        item = by_id.get(cid)
        if item is None:
            print(f"unknown coin_id: {cid}", file=sys.stderr)
            missing += 1
            continue
        front = crop(item["front_image"], item["front_rect"], args.size)
        back = crop(item["back_image"], item["back_rect"], args.size)
        gap = np.full((args.size, 8, 3), 255, dtype=np.uint8)
        out_path = os.path.join(out_dir, f"{cid}.jpg")
        cv2.imwrite(
            out_path, np.hstack([front, gap, back]), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        print(out_path)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
