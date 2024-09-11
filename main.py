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
    # inner: (x, y, w, h), outer: (x, y, w, h)
    x1, y1, w1, h1, _ = inner
    x2, y2, w2, h2, _ = outer
    return x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2)


# Function to remove rectangles inside larger rectangles
def filter_nested_rectangles(rectangles):
    filtered = []
    for i, rect1 in enumerate(rectangles):
        keep = True
        for j, rect2 in enumerate(rectangles):
            if i != j and is_inside(rect1, rect2):
                keep = False
                break
        if keep:
            filtered.append(rect1)
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
            debug_image_path = save_debug_image(edged, 'edged_dilated', filename, output_dir)
            image_paths.append(debug_image_path)

            # Find contours in the edged image
            contours, _ = cv2.findContours(edged_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 5: Filter and extract bounding rectangles from contours
            rectangles = []
            detected_objects = 0
            img_contour = img.copy()

            # img_all_contours = img.copy()
            # cv2.drawContours(img_all_contours, contours, -1, (255, 0, 0), 2)  # Draw all contours before filtering
            # save_debug_image(img_all_contours, 'all_contours', filename, output_dir)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if area > 40000:  # Filter out small contours based on area
                # if w*h > 40000:  # Filter out small contours based on area
                    hull = cv2.convexHull(contour)
                    x, y, w, h = cv2.boundingRect(hull)
                    rectangles.append((x, y, w, h, area))  # Store the bounding box and contour area

            filtered_rectangles = []
            filtered_rectangles = filter_nested_rectangles(rectangles)

            # Step 6: Draw rectangles and label with actual contour area
            for (x, y, w, h, area) in filtered_rectangles:
                detected_objects += 1
                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Label with the accurate contour area, not w * h
                label = f"{w*h:.0f}px"
                cv2.putText(img_contour, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detected_objects_total.append(detected_objects)

            # Save the images with rectangles to the output folder
            output_contour_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_contour.jpg")

            cv2.imwrite(output_contour_path, img_contour)

            image_paths.append(output_contour_path)

    print(f"Detection completed. Results saved to {output_dir}")

    for i, obj in enumerate(detected_objects_total):
        print(f"Coins in {filenames[i]}: {obj}")
    print(f"Nr. of coins in total: {sum(detected_objects_total)//2}")

    open_images(image_paths[-3:])




if __name__ == '__main__':

    import time
    start_time = time.time()
    main()
    seconds = (time.time() - start_time)
    print(f"Time needed: {round(seconds,2)} s")
