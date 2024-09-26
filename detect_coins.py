import cv2
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import platform
import subprocess
import json
from collections import defaultdict
from scipy.cluster.hierarchy import fclusterdata
import math


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
    extracted_path = os.path.join(extracted_folder, f"coin{coin_nr}_{base_name}_{side}.png")
    cv2.imwrite(extracted_path, image)
    return extracted_path

# TODO: Es fehlen noch ein paar (148/154 extracted)
# TODO: Falsches matching (wahrscheinlich wegen Rotation)


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


# Function to compute scale factor between two sets of rectangles
def compute_scale_factor(rects_front, rects_back):
    if not rects_front or not rects_back:
        return 1.0  # Default scale factor
    # Sort both lists by area
    sorted_front = sorted(rects_front, key=lambda x: x[4])
    sorted_back = sorted(rects_back, key=lambda x: x[4])
    # Compute area ratios
    ratios = []
    for f, b in zip(sorted_front, sorted_back):
        if b[4] == 0:
            continue
        ratios.append(f[4] / b[4])
    if not ratios:
        return 1.0
    # Use median to avoid outliers
    ratios = sorted(ratios)
    mid = len(ratios) // 2
    scale = ratios[mid] if len(ratios) % 2 != 0 else (ratios[mid - 1] + ratios[mid]) / 2
    return scale


# Function to assign coins to rows or columns based on their spatial arrangement
def assign_to_rows_or_columns(rectangles, filename):
    if not rectangles:
        return [], 'unknown'

    # Compute centers of rectangles
    centers = []
    for idx, rect in enumerate(rectangles):
        x, y, w, h, _ = rect
        center_x = x + w / 2
        center_y = y + h / 2
        centers.append([center_x, center_y])

    # Determine if arrangement is more horizontal (rows) or vertical (columns)
    # Calculate the variance along x and y axes
    centers_np = np.array(centers)
    variance_x = np.var(centers_np[:, 0])
    variance_y = np.var(centers_np[:, 1])

    if variance_y > variance_x:
        arrangement = 'rows'
        # Cluster based on y-coordinate
        cluster_axis = 1  # y-axis
    else:
        arrangement = 'columns'
        # Cluster based on x-coordinate
        cluster_axis = 0  # x-axis

    # Use hierarchical clustering to cluster the coins into rows or columns
    # The threshold can be adjusted based on expected spacing
    threshold = 50  # pixels; adjust as needed
    clusters = fclusterdata(centers, t=threshold, criterion='distance', metric='euclidean', method='single')

    # Organize coins into rows or columns
    layout = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        layout[cluster_id].append({
            'index': idx,
            'rect': rectangles[idx],
            'center': centers[idx],
            'area': rectangles[idx][4]
        })

    # Sort the clusters based on the primary axis to maintain order
    sorted_layout = sorted(layout.values(), key=lambda row: row[0]['center'][cluster_axis])

    return sorted_layout, arrangement


# Function to determine if back image is rotated relative to front image
def determine_rotation(layout_front, layout_back, arrangement_front, arrangement_back):
    if arrangement_front == arrangement_back:
        # Same arrangement; no rotation
        return 0
    else:
        # Different arrangements; likely rotated by 90 or -90 degrees
        # Determine which rotation aligns best by comparing anchor coins
        return None  # Placeholder; rotation will be determined based on anchor matches


# Function to rotate rectangle positions based on rotation angle
def rotate_rectangles(rects, angle, image_width, image_height):
    rotated_rects = []
    for rect in rects:
        x, y, w, h, area = rect
        if angle == 90:
            new_x = image_height - y - h
            new_y = x
            new_w = h
            new_h = w
        elif angle == -90:
            new_x = y
            new_y = image_width - x - w
            new_w = h
            new_h = w
        else:  # 0 degrees
            new_x = x
            new_y = y
            new_w = w
            new_h = h
        rotated_rects.append((new_x, new_y, new_w, new_h, area))
    return rotated_rects


# Function to find anchor coins based on largest area differences
def find_anchor_coins(rects, top_n=3):
    if not rects:
        return []
    areas = np.array([rect[4] for rect in rects])
    median_area = np.median(areas)
    # Compute absolute differences from median
    area_diffs = np.abs(areas - median_area)
    # Get indices of top_n largest differences
    anchor_indices = area_diffs.argsort()[-top_n:][::-1]
    anchors = [rects[i] for i in anchor_indices]
    return anchors


# Function to match anchor coins between front and back images for rotation determination
def match_anchor_coins(front_anchors, back_anchors, scale_factor):
    matched_angles = []
    possible_angles = [0, 90, -90]
    best_angle = 0
    max_matches = -1

    for angle in possible_angles:
        # Rotate back anchors by the angle
        rotated_back_anchors = rotate_rectangles(back_anchors, angle, image_width=1000,
                                                 image_height=1000)  # Placeholder dimensions
        # Compute scaled areas
        scaled_back_areas = [rect[4] * scale_factor for rect in rotated_back_anchors]
        # Compare with front anchors
        matches = 0
        for f_rect, b_rect in zip(front_anchors, rotated_back_anchors):
            area_diff = abs(f_rect[4] - b_rect[4])
            if area_diff < (0.1 * f_rect[4]):  # Allow 10% area difference
                matches += 1
        if matches > max_matches:
            max_matches = matches
            best_angle = angle

    return best_angle


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

    # Dictionary to store detections
    detections = {}
    layouts = {}
    arrangements = {}

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
                print(f"[{filename}] Detected rectangle {detected_objects}: x={x}, y={y}, w={w}, h={h}, area={area}")

            # Optionally, draw removed rectangles in red
            for rect in rectangles:
                if rect not in filtered_rectangles:
                    x, y, w, h, area = rect
                    print(f"[{filename}] Area of removed rectangle: {area}")
                    cv2.rectangle(img_removed, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for removed

            # Save images with rectangles
            save_debug_image(img_kept, 'kept_rectangles', filename, output_dir)
            save_debug_image(img_removed, 'removed_rectangles', filename, output_dir)

            output_contour_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_contour.jpg")
            cv2.imwrite(output_contour_path, img_kept)

            # Store the filtered rectangles
            detections[filename] = filtered_rectangles

            # Assign coins to rows or columns
            layout, arrangement = assign_to_rows_or_columns(filtered_rectangles, filename)
            layouts[filename] = layout
            arrangements[filename] = arrangement
            print(f"[{filename}] Arrangement: {arrangement}, Rows/Columns detected: {len(layout)}")

    print(f"Coin detection and layout assignment completed for all images.")

    # Now, process each pair to match coins and extract them
    extraction_folder = output_dir  # Use the same output directory
    extraction_results = []

    for pair_key, pair in tqdm(pairs.items(), desc="Matching and extracting coins"):
        front_filename = pair['front']
        back_filename = pair['back']
        rects_front = detections.get(front_filename, [])
        rects_back = detections.get(back_filename, [])
        layout_front = layouts.get(front_filename, [])
        layout_back = layouts.get(back_filename, [])
        arrangement_front = arrangements.get(front_filename, 'unknown')
        arrangement_back = arrangements.get(back_filename, 'unknown')

        print(f"Processing pair: Front - {front_filename}, Back - {back_filename}")

        # Compute scale factor
        scale_factor = compute_scale_factor(rects_front, rects_back)
        print(f"Computed scale factor for pair {pair_key}: {scale_factor:.4f}")

        # Find anchor coins in both images
        front_anchors = find_anchor_coins(rects_front, top_n=3)
        back_anchors = find_anchor_coins(rects_back, top_n=3)

        if not front_anchors or not back_anchors:
            print(f"Insufficient anchors for pair {pair_key}. Skipping.")
            continue

        # Determine image rotation using anchor coins
        # To determine rotation, we need actual image dimensions
        front_image_path = os.path.join(image_dir, front_filename)
        back_image_path = os.path.join(image_dir, back_filename)
        img_front = cv2.imread(front_image_path)
        img_back = cv2.imread(back_image_path)
        if img_front is None or img_back is None:
            print(f"Failed to read images for pair {pair_key}. Skipping.")
            continue
        front_height, front_width = img_front.shape[:2]
        back_height, back_width = img_back.shape[:2]

        # Find best rotation angle based on anchor matches
        # Possible angles: 0, 90, -90 degrees
        possible_angles = [0, 90, -90]
        best_angle = 0
        max_matches = -1

        for angle in possible_angles:
            # Rotate back anchors
            rotated_back_anchors = rotate_rectangles(back_anchors, angle, back_width, back_height)
            # Scale back anchor areas
            scaled_back_anchors = [(x, y, w, h, area * scale_factor) for (x, y, w, h, area) in rotated_back_anchors]
            # Match scaled back anchors to front anchors based on area
            matches = 0
            for f_anchor in front_anchors:
                f_area = f_anchor[4]
                # Find if there's a back anchor within 10% area difference
                for b_anchor in scaled_back_anchors:
                    b_area = b_anchor[4]
                    if abs(f_area - b_area) / f_area < 0.1:
                        matches += 1
                        break
            if matches > max_matches:
                max_matches = matches
                best_angle = angle

        print(f"Determined rotation for pair {pair_key}: {best_angle} degrees with {max_matches} anchor matches.")

        # Rotate back image's rectangles based on determined rotation
        if best_angle != 0:
            rotated_rects_back = rotate_rectangles(rects_back, best_angle, back_width, back_height)
            print(f"Rotated back image rectangles by {best_angle} degrees for pair {pair_key}.")
        else:
            rotated_rects_back = rects_back

        # Now, match all coins based on scaled areas and spatial alignment
        # Create lists of scaled front and back rectangles
        scaled_front_rects = rects_front  # Front image remains the same
        scaled_back_rects = rotated_rects_back  # Back image rectangles are rotated

        # Compute a new scale factor if necessary (optional)
        # For simplicity, we use the initial scale factor

        # Match coins based on area similarity
        matched_pairs = []
        used_back = set()
        for i, f_rect in enumerate(scaled_front_rects):
            f_x, f_y, f_w, f_h, f_area = f_rect
            f_center = (f_x + f_w / 2, f_y + f_h / 2)
            # Find back rect with closest scaled area
            best_match_idx = -1
            min_area_diff = float('inf')
            for j, b_rect in enumerate(scaled_back_rects):
                if j in used_back:
                    continue
                b_x, b_y, b_w, b_h, b_area = b_rect
                area_diff = abs(f_area - b_area)
                if area_diff < min_area_diff:
                    min_area_diff = area_diff
                    best_match_idx = j
            # Define a threshold for area difference (10% of front area)
            if min_area_diff / f_area < 0.1 and best_match_idx != -1:
                matched_pairs.append((i, best_match_idx))
                used_back.add(best_match_idx)

        print(f"Number of matched coins in pair {pair_key}: {len(matched_pairs)}")

        # Read rotated back image for extraction
        if best_angle != 0:
            # Rotate the entire back image for accurate extraction if needed
            # This step is optional and depends on whether you want to save rotated images
            # For extraction, we can work with rotated_rects_back directly
            pass

        # Proceed to extract and save matched coins
        for idx, (front_idx, back_idx) in enumerate(matched_pairs, start=1):
            if front_idx >= len(scaled_front_rects) or back_idx >= len(scaled_back_rects):
                print(f"Index out of range for pair {pair_key}: front_idx={front_idx}, back_idx={back_idx}")
                continue

            front_rect = scaled_front_rects[front_idx]
            back_rect = scaled_back_rects[back_idx]

            # Extract front coin
            x_f, y_f, w_f, h_f, _ = front_rect
            coin_front = img_front[y_f:y_f + h_f, x_f:x_f + w_f]
            extracted_front_path = save_extracted_image(coin_front, idx, 'front', front_filename, extraction_folder)

            # Extract back coin
            x_b, y_b, w_b, h_b, _ = back_rect
            coin_back = img_back[y_b:y_b + h_b, x_b:x_b + w_b]
            extracted_back_path = save_extracted_image(coin_back, idx, 'back', back_filename, extraction_folder)

            # Store extraction info
            extraction_results.append({
                'pair_key': pair_key,
                'coin_number': idx,
                'front_image': front_filename,
                'back_image': back_filename,
                'front_extracted': extracted_front_path,
                'back_extracted': extracted_back_path,
                'front_rect': {'x': x_f, 'y': y_f, 'w': w_f, 'h': h_f},
                'back_rect': {'x': x_b, 'y': y_b, 'w': w_b, 'h': h_b},
                'scale_factor': scale_factor,
                'rotation_applied': best_angle
            })

    # Save extraction results to JSON
    extraction_json_path = os.path.join(output_dir, 'extraction_results.json')
    with open(extraction_json_path, 'w') as f:
        json.dump(extraction_results, f, indent=4)
    print(f"Extraction results saved to {extraction_json_path}")

    print(f"All matched coins have been extracted and saved to '{extraction_folder}/extracted/'.")
