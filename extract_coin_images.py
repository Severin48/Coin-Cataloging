from __future__ import annotations

import cv2
import numpy as np
import os
import re
from datetime import datetime
from tqdm import tqdm
import platform
import subprocess
import json
from collections import defaultdict
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

# Type aliases

# OpenCV images
NDArray = np.ndarray

# Rectangle as (x, y, w, h, area)
Rect = Tuple[int, int, int, int, float]

# Row sequence: list of (Rect, original_index)
RectIdx = Tuple[Rect, int]
RowSequence = List[RectIdx]


# Layout per filename
class Layout(TypedDict):
    sequences: List[RowSequence]


# Pair mapping: '..._v' → 'front', '..._r' → 'back'
ImagePair = Dict[str, str]  # keys: "front", "back"


# One JSON entry for an extracted coin
class ExtractionEntry(TypedDict):
    pair_key: str
    coin_number: int
    front_image: str
    back_image: str
    front_extracted: str
    back_extracted: str
    front_rect: Dict[str, int]
    back_rect: Dict[str, int]


def save_debug_image(image: NDArray, step_name: str, filename: str, folder: str) -> str:
    debug_folder = os.path.join(folder, "debug")
    os.makedirs(debug_folder, exist_ok=True)
    debug_path = os.path.join(
        debug_folder, f"{os.path.splitext(filename)[0]}_{step_name}.png"
    )
    cv2.imwrite(debug_path, image)
    return debug_path


def save_extracted_image(
    image: NDArray, coin_nr: int, side: str, filename: str, folder: str
) -> str:
    extracted_folder = os.path.join(folder, "extracted")
    os.makedirs(extracted_folder, exist_ok=True)
    # Adjusted naming as per user’s change to ensure paired sides are next to each other
    base_name = os.path.splitext(filename)[0][:-2]  # Remove '_v' or '_r'
    extracted_path = os.path.join(
        extracted_folder, f"{base_name}_coin{coin_nr}_{side}.png"
    )
    cv2.imwrite(extracted_path, image)
    return extracted_path


def draw_rows_debug_image(
    image: NDArray,
    num_segments: int,
    segment_height: float,
    selected_segments: Optional[Iterable[int]],
    filename: str,
    output_dir: str,
    save_debug: bool = False,
) -> None:
    img_copy = image.copy()
    overlay = img_copy.copy()
    image_height, image_width = img_copy.shape[:2]

    selected = set(selected_segments or [])

    for seg_idx in range(num_segments):
        y_start = int(seg_idx * segment_height)
        y_end = int((seg_idx + 1) * segment_height)
        if selected and seg_idx in selected:
            color = (0, 255, 0)  # Grün
        else:
            color = (255, 255, 255)  # Weiß

        cv2.rectangle(overlay, (0, y_start), (image_width, y_end), color, -1)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

    if save_debug:
        save_debug_image(img_copy, "rows", filename, output_dir)


def save_extraction_overview_image(
    image: NDArray, rectangles: Sequence[Rect], filename: str, folder: str
) -> str:
    overview_img = image.copy()

    for x, y, w, h, _area in rectangles:
        cv2.rectangle(overview_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    overview_path = os.path.join(
        folder, f"{os.path.splitext(filename)[0]}_extracted.png"
    )
    cv2.imwrite(overview_path, overview_img)
    return overview_path


def is_inside(inner: Rect, outer: Rect) -> bool:
    x1, y1, w1, h1, _ = inner
    x2, y2, w2, h2, _ = outer
    return x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2)


def filter_nested_rectangles(
    rectangles: Sequence[Rect], filename: str, verbose: bool = False
) -> List[Rect]:
    if verbose:
        print(f"[{filename}] Rectangles before filtering nested: {len(rectangles)}")
    filtered: List[Rect] = []
    for i, rect1 in enumerate(rectangles):
        keep = True
        for j, rect2 in enumerate(rectangles):
            if i != j and is_inside(rect1, rect2):
                keep = False
                if verbose:
                    print(f"[{filename}] Removing rectangle {rect1} inside {rect2}")
                break
        if keep:
            filtered.append(rect1)
    if verbose:
        print(f"[{filename}] Rectangles after filtering nested: {len(filtered)}")
    return filtered


def filter_non_squares(
    rectangles: Sequence[Rect], filename: str, verbose: bool = False
) -> List[Rect]:
    if verbose:
        print(
            f"[{filename}] Rectangles before filtering other shapes: {len(rectangles)}"
        )
    filtered: List[Rect] = []
    for rect in rectangles:
        _, _, w, h, _ = rect
        ratio = w / h if h != 0 else 0.0
        if ratio > 1.5 or ratio < 0.75:
            if verbose:
                print(
                    f"[{filename}] Removing rectangle {rect} due to not being square enough"
                )
        else:
            filtered.append(rect)
    if verbose:
        print(f"[{filename}] Rectangles after filtering other shapes: {len(filtered)}")
    return filtered


def open_image(image_path: str) -> None:
    if platform.system() == "Windows":
        os.startfile(image_path)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", image_path])
    else:  # Linux
        subprocess.call(["xdg-open", image_path])


def open_images(image_paths: Iterable[str]) -> None:
    for img in image_paths:
        open_image(img)


def pair_images(filenames: Sequence[str]) -> Dict[str, ImagePair]:
    pairs: DefaultDict[str, ImagePair] = defaultdict(dict)
    for filename in filenames:
        if not any(
            filename.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue
        # Example: k1_h2_s4_v.jpg or k1_h2_s4_r.jpg
        name_part, _ext = os.path.splitext(filename)
        if name_part.endswith("_v"):
            key = name_part[:-2]  # Remove '_v'
            pairs[key]["front"] = filename
        elif name_part.endswith("_r"):
            key = name_part[:-2]  # Remove '_r'
            pairs[key]["back"] = filename
    complete_pairs: Dict[str, ImagePair] = {
        k: v for k, v in pairs.items() if "front" in v and "back" in v
    }
    return complete_pairs


def assign_to_rows_by_center(
    rectangles: Sequence[Rect],
    image: NDArray,
    filename: str,
    debug_dir: str,
    min_gap_factor: float = 0.6,
    save_debug: bool = False,
) -> List[RowSequence]:
    """
    Group bounding boxes by their vertical centers into row sequences.
    """
    if not rectangles:
        return []

    centers: List[Tuple[float, int]] = [
        (rect[1] + rect[3] / 2, idx) for idx, rect in enumerate(rectangles)
    ]
    centers.sort(key=lambda x: x[0])

    heights: List[int] = [rect[3] for rect in rectangles]
    median_h: float = float(np.median(heights)) if heights else 0.0

    rows: List[List[int]] = []
    current: List[int] = [centers[0][1]]
    mean_y: float = centers[0][0]
    for cy, idx in centers[1:]:
        if abs(cy - mean_y) < median_h * min_gap_factor:
            current.append(idx)
            mean_y = float(
                np.mean([rectangles[i][1] + rectangles[i][3] / 2 for i in current])
            )
        else:
            rows.append(current)
            current = [idx]
            mean_y = cy
    rows.append(current)

    debug_img = image.copy()
    for row in rows:
        y_centers = [rectangles[i][1] + rectangles[i][3] / 2 for i in row]
        avg_y = int(sum(y_centers) / len(y_centers))
        cv2.line(debug_img, (0, avg_y), (debug_img.shape[1], avg_y), (0, 255, 0), 2)

    base = os.path.splitext(filename)[0]
    debug_path = os.path.join(debug_dir, f"{base}_rows.png")
    if save_debug:
        cv2.imwrite(debug_path, debug_img)

    sequences: List[RowSequence] = []
    for row in rows:
        seq: RowSequence = [(rectangles[i], i) for i in row]
        seq.sort(key=lambda item: item[0][0])
        sequences.append(seq)

    return sequences


def detect_coins(
    start: str = "",
    save_extracted: bool = False,
    save_debug: bool = False,
    verbose: bool = False,
) -> None:
    # Create a directory for results with a timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = f"results/{timestamp}"
    debug_dir = output_dir + "/debug"
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    image_dir = "images"

    # Collect all filenames
    filenames = sorted(os.listdir(image_dir))

    # Pair images into front and back
    pairs = pair_images(filenames)
    print(f"Total image pairs found: {len(pairs)}")

    def nat_key(k: str) -> Tuple[int, ...]:
        """Turn 'k2_h10_s7' → (2,10,7) so that 10>9 is sorted correctly."""
        m = re.match(r"k(\d+)_h(\d+)_s(\d+)", k)
        return tuple(map(int, m.groups())) if m else (float("inf"),)  # type: ignore[return-value]

    sorted_keys = sorted(pairs.keys(), key=nat_key)

    start_key: Optional[str] = None
    if start:
        if start.endswith(("_r.jpg", "_v.jpg")):
            start_key = start[:-6]
        else:
            start_key = start

    if start_key:
        skipped_keys = [k for k in sorted_keys if nat_key(k) < nat_key(start_key)]
        process_keys = [k for k in sorted_keys if nat_key(k) >= nat_key(start_key)]
    else:
        skipped_keys = []
        process_keys = sorted_keys

    if skipped_keys and verbose:
        print(
            f"Skipping {skipped_keys[0]} to {skipped_keys[-1]} "
            f"({len(skipped_keys)} pairs total)"
        )

    detections: Dict[str, List[Rect]] = {}
    layouts: Dict[str, Layout] = {}
    total_coins = 0

    all_extracted_rectangles: DefaultDict[str, List[Rect]] = defaultdict(list)

    # First pass: Detect coins in all images and store their rectangles
    for pair_key in tqdm(process_keys, desc="Processing image pairs"):
        pair = pairs[pair_key]
        for side, filename in pair.items():
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {filename}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            edges_img = cv2.Canny(blurred, 40, 80)
            if save_debug:
                save_debug_image(edges_img, "edges", filename, output_dir)

            kernel = np.ones((11, 11), np.uint8)
            edges_dilated = cv2.dilate(edges_img, kernel, iterations=1)
            if save_debug:
                save_debug_image(edges_dilated, "edges_dilated", filename, output_dir)

            contours, _ = cv2.findContours(
                edges_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if verbose:
                print(f"[{filename}] Initial contours detected: {len(contours)}")

            img_all_contours = img.copy()
            cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 2)
            if save_debug:
                save_debug_image(img_all_contours, "all_contours", filename, output_dir)

            filtered_contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) > 30000
            ]
            if verbose:
                print(
                    f"[{filename}] Contours after area filtering: {len(filtered_contours)}"
                )

            rectangles: List[Rect] = []
            detected_objects = 0

            for contour in filtered_contours:
                area = float(cv2.contourArea(contour))
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                rectangles.append((int(x), int(y), int(w), int(h), area))

            filtered_rectangles = filter_nested_rectangles(rectangles, filename)
            filtered_rectangles = filter_non_squares(filtered_rectangles, filename)

            img_kept = img.copy()
            img_removed = img.copy()

            for x, y, w, h, area in filtered_rectangles:
                detected_objects += 1
                cv2.rectangle(img_kept, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{detected_objects}"
                cv2.putText(
                    img_kept,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            for rect in rectangles:
                if rect not in filtered_rectangles:
                    x, y, w, h, area = rect
                    if verbose:
                        print(f"[{filename}] Area of removed rectangle: {area}")
                    cv2.rectangle(img_removed, (x, y), (x + w, y + h), (0, 0, 255), 2)

            total_coins += len(filtered_rectangles)

            if save_debug:
                save_debug_image(img_kept, "kept_rectangles", filename, debug_dir)
                save_debug_image(img_removed, "removed_rectangles", filename, debug_dir)

            output_contour_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}_contour.jpg"
            )
            if save_debug:
                cv2.imwrite(output_contour_path, img_kept)

            detections[filename] = filtered_rectangles

            assigned_sequences = assign_to_rows_by_center(
                filtered_rectangles, img, filename, debug_dir
            )

            layouts[filename] = {
                "sequences": assigned_sequences,
            }

    extraction_folder = output_dir
    extraction_results: List[ExtractionEntry] = []

    for pair_key in tqdm(process_keys, desc="Matching and extracting coins"):
        pair = pairs[pair_key]
        front_filename = pair["front"]
        back_filename = pair["back"]
        layout_front = layouts.get(front_filename)
        layout_back = layouts.get(back_filename)

        if not layout_front or not layout_back:
            print(
                f"No rectangles detected in front or back image for pair {pair_key}. Skipping."
            )
            continue

        front_image_path = os.path.join(image_dir, front_filename)
        back_image_path = os.path.join(image_dir, back_filename)
        img_front = cv2.imread(front_image_path)
        img_back = cv2.imread(back_image_path)
        if img_front is None or img_back is None:
            print(f"Failed to read images for pair {pair_key}. Skipping.")
            continue

        num_rows = min(len(layout_front["sequences"]), len(layout_back["sequences"]))
        idx = 0
        for row in range(num_rows):
            front_rects = layout_front["sequences"][row]
            back_rects = layout_back["sequences"][row]

            for front_rect, back_rect in zip(front_rects, back_rects):
                x_f, y_f, w_f, h_f, _af = front_rect[0]
                x_b, y_b, w_b, h_b, _ab = back_rect[0]

                extracted_front_path = ""
                extracted_back_path = ""

                coin_front = img_front[y_f : y_f + h_f, x_f : x_f + w_f]
                if save_extracted:
                    extracted_front_path = save_extracted_image(
                        coin_front, idx + 1, "front", front_filename, extraction_folder
                    )

                coin_back = img_back[y_b : y_b + h_b, x_b : x_b + w_b]
                if save_extracted:
                    extracted_back_path = save_extracted_image(
                        coin_back, idx + 1, "back", back_filename, extraction_folder
                    )

                extraction_results.append(
                    {
                        "pair_key": pair_key,
                        "coin_number": idx + 1,
                        "front_image": front_filename,
                        "back_image": back_filename,
                        "front_extracted": extracted_front_path,
                        "back_extracted": extracted_back_path,
                        "front_rect": {"x": x_f, "y": y_f, "w": w_f, "h": h_f},
                        "back_rect": {"x": x_b, "y": y_b, "w": w_b, "h": h_b},
                    }
                )
                idx += 1

                all_extracted_rectangles[front_filename].append(
                    (x_f, y_f, w_f, h_f, _af)
                )
                all_extracted_rectangles[back_filename].append(
                    (x_b, y_b, w_b, h_b, _ab)
                )

    for filename, rects in tqdm(
        all_extracted_rectangles.items(), desc="Saving extraction results"
    ):
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is not None and save_debug:
            save_extraction_overview_image(img, rects, filename, output_dir)

    extraction_json_path = os.path.join(output_dir, "extraction_results.json")
    with open(extraction_json_path, "w") as f:
        json.dump(extraction_results, f, indent=4)
    print(f"Extraction results saved to {extraction_json_path}")

    print(
        f"All matched coins have been extracted and saved to '{extraction_folder}/extracted/'."
    )
    print(f"Extracted {len(extraction_results)} pairs of front- and backside images.")
    print(f"Detected {total_coins // 2} coins in total.")
