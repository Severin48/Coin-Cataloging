import cv2
import numpy as np
import os
from datetime import datetime
from sys import maxsize
from tqdm import tqdm
import platform
import subprocess
import json
from collections import defaultdict


def save_debug_image(image, step_name, filename, folder):
    debug_folder = os.path.join(folder, 'debug')
    os.makedirs(debug_folder, exist_ok=True)
    debug_path = os.path.join(debug_folder, f"{os.path.splitext(filename)[0]}_{step_name}.jpg")
    cv2.imwrite(debug_path, image)
    return debug_path


def save_extracted_image(image, coin_nr, side, filename, folder):
    extracted_folder = os.path.join(folder, 'extracted')
    os.makedirs(extracted_folder, exist_ok=True)
    # Adjusted naming as per user’s change to ensure paired sides are next to each other
    base_name = os.path.splitext(filename)[0][:-2]  # Remove '_v' or '_r'
    extracted_path = os.path.join(extracted_folder, f"{base_name}_coin{coin_nr}_{side}.png")
    cv2.imwrite(extracted_path, image)
    return extracted_path

def draw_rows_debug_image(image, num_segments, segment_height, selected_segments, filename, output_dir):
    img_copy = image.copy()
    overlay = img_copy.copy()
    image_height, image_width = img_copy.shape[:2]

    for seg_idx in range(num_segments):
        y_start = int(seg_idx * segment_height)
        y_end = int((seg_idx + 1) * segment_height)
        if selected_segments and seg_idx in selected_segments:
            color = (0, 255, 0)  # Grün für relevante Segmente
        else:
            color = (255, 255, 255)  # Rot für Segmente mit Rechtecken, die nicht ausgewählt sind

        # Zeichne halbtransparentes Rechteck auf das Overlay
        cv2.rectangle(overlay, (0, y_start), (image_width, y_end), color, -1)

    # Überlagere das Overlay mit Transparenz auf das Originalbild
    alpha = 0.4  # Transparenzfaktor
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

    # Speichere das Debug-Bild
    debug_path = save_debug_image(img_copy, 'rows', filename, output_dir)


def save_extraction_overview_image(image, rectangles, filename, folder):
    overview_img = image.copy()

    for (x, y, w, h, area) in rectangles:
        cv2.rectangle(overview_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for extracted coins

    # Save the overview image in the main folder
    overview_path = os.path.join(folder, f"{os.path.splitext(filename)[0]}_extracted.png")
    cv2.imwrite(overview_path, overview_img)
    return overview_path


# Function to check if a rectangle is inside another
def is_inside(inner, outer):
    x1, y1, w1, h1, _ = inner
    x2, y2, w2, h2, _ = outer
    return x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2)


# Function to remove rectangles inside larger rectangles
def filter_nested_rectangles(rectangles, filename):
    print(f"[{filename}] Rectangles before filtering nested: {len(rectangles)}")
    filtered = []
    for i, rect1 in enumerate(rectangles):
        keep = True
        for j, rect2 in enumerate(rectangles):
            if i != j and is_inside(rect1, rect2):
                keep = False
                print(f"[{filename}] Removing rectangle {rect1} inside {rect2}")
                break
        if keep:
            filtered.append(rect1)
    print(f"[{filename}] Rectangles after filtering nested: {len(filtered)}")
    return filtered


def filter_non_squares(rectangles, filename):
    print(f"[{filename}] Rectangles before filtering other shapes: {len(rectangles)}")
    filtered = []
    for rect in rectangles:
        _, _, w, h, _ = rect
        ratio = w / h
        if ratio > 1.5 or ratio < 0.75:
            print(f"[{filename}] Removing rectangle {rect} due to not being square enough")
        else:
            filtered.append(rect)
    print(f"[{filename}] Rectangles after filtering other shapes: {len(filtered)}")
    return filtered


# Function to open images after processing
def open_image(image_path):
    if platform.system() == "Windows":
        os.startfile(image_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", image_path])
    else:  # Linux
        subprocess.call(["xdg-open", image_path])


def open_images(image_paths):
    for img in image_paths:
        open_image(img)


# Function to pair front and back images based on filename labels
def pair_images(filenames):
    pairs = defaultdict(dict)
    for filename in filenames:
        if not any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        # Example filename: k1_h2_s4_v.jpg or k1_h2_s4_r.jpg
        name_part, ext = os.path.splitext(filename)
        if name_part.endswith('_v'):
            key = name_part[:-2]  # Remove '_v'
            pairs[key]['front'] = filename
        elif name_part.endswith('_r'):
            key = name_part[:-2]  # Remove '_r'
            pairs[key]['back'] = filename
    # Filter out incomplete pairs
    complete_pairs = {k: v for k, v in pairs.items() if 'front' in v and 'back' in v}
    return complete_pairs


def assign_to_rows(rectangles, image, filename, output_dir):
    if not rectangles:
        return [], []

    print(f"[{filename}] Assigning rectangles to rows.")

    num_segments = 100
    segment_height = max(image.shape[0] / num_segments, 1.)

    # For each segment, find rectangles that overlap with the segment
    segments = [[] for _ in range(num_segments)]

    for idx, rect in enumerate(rectangles):
        x, y, w, h, _ = rect
        rect_top = y
        rect_bottom = y + h

        # Find which segments the rectangle overlaps
        start_segment = int(rect_top / segment_height)
        end_segment = int(rect_bottom / segment_height)

        # Clamp the values
        start_segment = max(0, min(num_segments - 1, start_segment))
        end_segment = max(0, min(num_segments - 1, end_segment))

        for seg_idx in range(start_segment, end_segment + 1):
            segments[seg_idx].append((rect, idx))

    # Now, for each segment, sort the rectangles by x (left to right)
    for seg_idx in range(num_segments):
        segments[seg_idx].sort(key=lambda item: item[0][0])  # item[0][0] is x of rectangle

    # Now, for each segment, group sequences starting with the same leftmost rectangle
    sequences_dict = {}  # key: leftmost rectangle index, value: list of sequences

    for seg_idx in range(num_segments):
        rects_in_segment = segments[seg_idx]
        if not rects_in_segment:
            continue

        # Get rectangle indices in order
        rect_indices = [idx for (rect, idx) in rects_in_segment]

        # Get the leftmost rectangle index
        leftmost_idx = rect_indices[0]

        # Add the sequence to the list for this leftmost rectangle
        if leftmost_idx not in sequences_dict:
            sequences_dict[leftmost_idx] = []

        sequences_dict[leftmost_idx].append((seg_idx, rect_indices))

    # Erstelle eine Menge, um alle Indizes zu sammeln, die irgendwo in den Sequenzen vorkommen
    used_indices = set()

    # Füge alle Indizes hinzu, die in den Sequenzen vorkommen (außer die ersten Indizes)
    for value in sequences_dict.values():
        for _, seq_list in value:
            used_indices.update(seq_list[1:])

    # Gehe durch die sequences_dict und lösche alle Einträge, deren Startindex in der used_indices-Menge ist
    keys_to_delete = [key for key in sequences_dict if key in used_indices]

    for key in keys_to_delete:
        del sequences_dict[key]

    # Now, find the longest sequence(s)
    max_lengths_dict = {}

    for leftmost_idx, seq_list in sequences_dict.items():
        max_lengths_dict[leftmost_idx] = 0
        for _, seq in seq_list:
            max_lengths_dict[leftmost_idx] = max(max_lengths_dict[leftmost_idx], len(seq))

    # From sequences with max_length, select the one in the middle
    if not max_lengths_dict:
        print(f"[{filename}] No sequences found.")
        return [], []

    relevant_segments = []
    relevant_sequences = []
    min_max_index = {}
    for leftmost_idx, seq_list in sequences_dict.items():
        min_max_index[leftmost_idx] = [maxsize, -maxsize]
        for seg_idx, seq in seq_list:
            if len(seq) == max_lengths_dict[leftmost_idx]:
                if seg_idx < min_max_index[leftmost_idx][0]:
                    min_max_index[leftmost_idx][0] = seg_idx
                if seg_idx > min_max_index[leftmost_idx][1]:
                    min_max_index[leftmost_idx][1] = seg_idx
        index = sum(min_max_index[leftmost_idx]) // 2
        relevant_sequences.append(segments[index])
        relevant_segments.append(index)

    relevant_sequences = [sorted(seq, key=lambda rect: rect[0][0]) for seq in relevant_sequences]

    # Sortiere die äußere Liste nach dem y-Wert des ersten Eintrags pro Liste
    relevant_sequences.sort(key=lambda seq: seq[0][0][1])

    print(f"[{filename}] Found {len(relevant_sequences)} rows.")
    draw_rows_debug_image(image, num_segments, segment_height, relevant_segments, filename, output_dir)

    return relevant_sequences, relevant_segments


def detect_coins():
    # Create a directory for results with a timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Set the directory for images (relative to the script)
    image_dir = 'images'
    image_formats = ('.jpg', '.jpeg', '.png', '.bmp')  # Common image formats

    # Collect all filenames
    filenames = os.listdir(image_dir)

    # Pair images into front and back
    pairs = pair_images(filenames)
    print(f"Total image pairs found: {len(pairs)}")

    # Dictionaries to store detections and layouts
    detections = {}
    layouts = {}
    total_coins = 0

    all_extracted_rectangles = defaultdict(list)

    # First pass: Detect coins in all images and store their rectangles
    for pair_key, pair in tqdm(pairs.items(), desc="Processing image pairs"):
        for side, filename in pair.items():
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {filename}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            # Edge detection
            edged = cv2.Canny(blurred, 40, 80)
            save_debug_image(edged, 'edged', filename, output_dir)

            # Dilate edges to close gaps
            kernel = np.ones((11, 11), np.uint8)
            edged_dilated = cv2.dilate(edged, kernel, iterations=1)
            save_debug_image(edged_dilated, 'edged_dilated', filename, output_dir)

            # Find contours in the edged image
            contours, _ = cv2.findContours(edged_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[{filename}] Initial contours detected: {len(contours)}")

            # Draw all contours for debugging
            img_all_contours = img.copy()
            cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 2)  # Blue contours
            save_debug_image(img_all_contours, 'all_contours', filename, output_dir)

            # Filter contours by area
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40000]
            print(f"[{filename}] Contours after area filtering: {len(filtered_contours)}")

            # Extract bounding rectangles from contours
            rectangles = []
            detected_objects = 0
            img_contour = img.copy()

            for contour in filtered_contours:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                rectangles.append((x, y, w, h, area))

            # Filter nested rectangles
            filtered_rectangles = filter_nested_rectangles(rectangles, filename)
            # Filter non-square rectangles
            filtered_rectangles = filter_non_squares(filtered_rectangles, filename)

            img_kept = img.copy()
            img_removed = img.copy()

            for (x, y, w, h, area) in filtered_rectangles:
                detected_objects += 1
                cv2.rectangle(img_kept, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for kept
                label = f"{detected_objects}"
                cv2.putText(img_kept, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # print(f"[{filename}] Detected rectangle {detected_objects}: x={x}, y={y}, w={w}, h={h}, area={area}")

            # Optionally, draw removed rectangles in red
            for rect in rectangles:
                if rect not in filtered_rectangles:
                    x, y, w, h, area = rect
                    print(f"[{filename}] Area of removed rectangle: {area}")
                    cv2.rectangle(img_removed, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for removed

            total_coins += len(filtered_rectangles)

            # Save images with rectangles
            save_debug_image(img_kept, 'kept_rectangles', filename, output_dir)
            save_debug_image(img_removed, 'removed_rectangles', filename, output_dir)

            output_contour_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_contour.jpg")
            cv2.imwrite(output_contour_path, img_kept)

            # Store the filtered rectangles
            detections[filename] = filtered_rectangles

            assigned_rectangles, segments = assign_to_rows(filtered_rectangles, img, filename, output_dir)

            # Store the layout
            layouts[filename] = {
                'sequences': assigned_rectangles,
                'segments': segments
            }

    # Now, process each pair to match coins and extract them
    extraction_folder = output_dir  # Use the same output directory
    extraction_results = []

    for pair_key, pair in tqdm(pairs.items(), desc="Matching and extracting coins"):
        front_filename = pair['front']
        back_filename = pair['back']
        layout_front = layouts.get(front_filename, [])
        layout_back = layouts.get(back_filename, [])

        if not layout_front or not layout_back:
            print(f"No rectangles detected in front or back image for pair {pair_key}. Skipping.")
            continue

        front_image_path = os.path.join(image_dir, front_filename)
        back_image_path = os.path.join(image_dir, back_filename)
        img_front = cv2.imread(front_image_path)
        img_back = cv2.imread(back_image_path)
        if img_front is None or img_back is None:
            print(f"Failed to read images for pair {pair_key}. Skipping.")
            continue

        num_rows = min(len(layout_front['sequences']), len(layout_back['sequences']))
        idx = 0
        for row in range(num_rows):
            front_rects = layout_front['sequences'][row]
            back_rects = layout_back['sequences'][row]

            for front_rect, back_rect in zip(front_rects, back_rects):
                x_f, y_f, w_f, h_f, _ = front_rect[0]
                x_b, y_b, w_b, h_b, _ = back_rect[0]

                # Extract front coin
                coin_front = img_front[y_f:y_f + h_f, x_f:x_f + w_f]
                extracted_front_path = save_extracted_image(coin_front, idx + 1, 'front', front_filename, extraction_folder)

                # Extract back coin
                coin_back = img_back[y_b:y_b + h_b, x_b:x_b + w_b]
                extracted_back_path = save_extracted_image(coin_back, idx + 1, 'back', back_filename, extraction_folder)

                # Store extraction info
                extraction_results.append({
                    'pair_key': pair_key,
                    'coin_number': idx + 1,
                    'front_image': front_filename,
                    'back_image': back_filename,
                    'front_extracted': extracted_front_path,
                    'back_extracted': extracted_back_path,
                    'front_rect': {'x': x_f, 'y': y_f, 'w': w_f, 'h': h_f},
                    'back_rect': {'x': x_b, 'y': y_b, 'w': w_b, 'h': h_b},
                })
                idx += 1

                # Collect extracted rectangles for the overview image
                all_extracted_rectangles[front_filename].append((x_f, y_f, w_f, h_f, _))
                all_extracted_rectangles[back_filename].append((x_b, y_b, w_b, h_b, _))

    for filename, rects in all_extracted_rectangles.items():
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is not None:
            save_extraction_overview_image(img, rects, filename, output_dir)

    # TODO: Warum 75 / 77 Paaren extrahiert??

    # Save extraction results to JSON
    extraction_json_path = os.path.join(output_dir, 'extraction_results.json')
    with open(extraction_json_path, 'w') as f:
        json.dump(extraction_results, f, indent=4)
    print(f"Extraction results saved to {extraction_json_path}")

    print(f"All matched coins have been extracted and saved to '{extraction_folder}/extracted/'.")
    print(f"Extracted {len(extraction_results)} pairs of front- and backside images.")
    print(f"Detected {total_coins//2} coins in total.")
