import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import platform
import subprocess


def save_debug_image(image, step_name, filename, folder):
    debug_folder = os.path.join(folder, 'debug')
    os.makedirs(debug_folder, exist_ok=True)
    debug_path = os.path.join(debug_folder, f"{os.path.splitext(filename)[0]}_{step_name}.jpg")
    cv2.imwrite(debug_path, image)

    return debug_path


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
        ratio = w/h
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


def main():
    # Create a directory for results with a timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Set the directory for images (relative to the script)
    image_dir = 'images'
    image_formats = ('.jpg', '.jpeg', '.png', '.bmp')  # Common image formats
    detected_objects_total = []
    # Loop through all files in the 'images' directory
    filenames = os.listdir(image_dir)
    image_paths = []
    for filename in tqdm(filenames):
        if filename.lower().endswith(image_formats):
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)

            # Method 1: Contour detection to find coins
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                    11, 2)

            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            # save_debug_image(blurred, 'blurred', filename, output_dir)
            edged = cv2.Canny(blurred, 40, 80)
            debug_image_path = save_debug_image(edged, 'edged', filename, output_dir)
            image_paths.append(debug_image_path)

            kernel = np.ones((11, 11), np.uint8)
            edged_dilated = cv2.dilate(edged, kernel, iterations=1)
            debug_image_path = save_debug_image(edged_dilated, 'edged_dilated', filename, output_dir)
            image_paths.append(debug_image_path)

            # Find contours in the edged image
            contours, _ = cv2.findContours(edged_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[{filename}] Initial contours detected: {len(contours)}")

            img_all_contours = img.copy()
            cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 2)  # Blue contours
            save_debug_image(img_all_contours, 'all_contours', filename, output_dir)

            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40000]
            print(f"[{filename}] Contours after area filtering: {len(filtered_contours)}")

            # Step 5: Filter and extract bounding rectangles from contours
            rectangles = []
            detected_objects = 0
            img_contour = img.copy()

            # img_all_contours = img.copy()
            # cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 2)  # Draw all contours before filtering
            # save_debug_image(img_all_contours, 'all_contours', filename, output_dir)

            for contour in filtered_contours:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                rectangles.append((x, y, w, h, area))
                # print(f"[{filename}] Detected contour: x={x}, y={y}, w={w}, h={h}, area={area}")

                # if area > 40000:  # Filter out small contours based on area
                # # if w*h > 40000:  # Filter out small contours based on area
                #     hull = cv2.convexHull(contour)
                #     x, y, w, h = cv2.boundingRect(hull)
                #     rectangles.append((x, y, w, h, area))  # Store the bounding box and contour area

            filtered_rectangles = []
            filtered_rectangles = filter_nested_rectangles(rectangles, filename)
            filtered_rectangles = filter_non_squares(filtered_rectangles, filename)

            img_kept = img.copy()
            img_removed = img.copy()

            for (x, y, w, h, area) in filtered_rectangles:
                detected_objects += 1
                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for kept
                cv2.rectangle(img_kept, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for kept
                label = f"{w * h:.0f}px"
                label = f"{detected_objects}"  # TODO: Remove
                cv2.putText(img_contour, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
                print(f"[{filename}] Detected rectangle {detected_objects}: x={x}, y={y}, w={w}, h={h}, area={area}")

                # TODO: Je Größenfaktor zwischen Bild _v und _r ausrechnen --> Daraus alle Werte seinem Partnerwert mappen --> Daraus schätzen welcher Wert keinen Partner hat
                # --> Alle so zuordnen damit das Mappen den minimalen Abstand hat considering the factor

            # Optionally, draw removed rectangles in red
            for rect in rectangles:
                if rect not in filtered_rectangles:
                    x, y, w, h, area = rect
                    print(f"[{filename}] Area of removed rectangle: {area}")
                    cv2.rectangle(img_removed, (x, y), (x + w, y + h), (0, 0, 255), 4)  # Red for removed

            # Save images
            save_debug_image(img_kept, 'kept_rectangles', filename, output_dir)
            save_debug_image(img_removed, 'removed_rectangles', filename, output_dir)

            detected_objects_total.append(detected_objects)

            # Save the images with rectangles to the output folder
            output_contour_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_contour.jpg")

            cv2.imwrite(output_contour_path, img_contour)

            image_paths.append(output_contour_path)

    print(f"Detection completed. Results saved to {output_dir}")

    for i, obj in enumerate(detected_objects_total):
        print(f"Coins in {filenames[i]}: {obj}")
    print(f"Nr. of coins in total: {sum(detected_objects_total)//2}")

    open_images(image_paths[-2:])




if __name__ == '__main__':

    import time
    start_time = time.time()
    main()
    seconds = (time.time() - start_time)
    print(f"Time needed: {round(seconds,2)} s")
