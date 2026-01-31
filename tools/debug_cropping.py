import cv2, sys, csv
import numpy as np
import os
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = f'{current_dir}/inputs/'
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.image_processing import get_reader

parent_dir = os.path.dirname(parent_dir)

crop_outputs = f"{current_dir}/crop_outputs/"
crop_images_hard = f"{current_dir}/crop_images_hard/"
os.makedirs(crop_outputs, exist_ok=True)
os.makedirs(crop_images_hard, exist_ok=True)


def create_debug_image(original_image, image_name, ocr_detections, start_row, start_col, end_col,
                       contours, convex_hull, top_left, top_right, x, y, w, h, output_folder):
    """Create a single comprehensive debug image showing all key information"""

    # Convert to BGR if grayscale
    if len(original_image.shape) == 2:
        debug_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        debug_image = original_image.copy()

    # 1. Draw OCR detection region (the restricted area that was scanned for text)
    cv2.rectangle(debug_image,
                  (start_col, start_row),
                  (end_col, debug_image.shape[0]),
                  (255, 255, 0), 3)  # Cyan - OCR scan region

    # 2. Draw detected text boxes
    for detection in ocr_detections:
        top_left_ocr = tuple(map(int, detection[0][0]))
        bottom_right_ocr = tuple(map(int, detection[0][2]))
        # Adjust coordinates for the start_col and start_row offset
        cv2.rectangle(debug_image,
                     (top_left_ocr[0] + start_col, top_left_ocr[1] + start_row),
                     (bottom_right_ocr[0] + start_col, bottom_right_ocr[1] + start_row),
                     (0, 255, 255), 2)  # Yellow - detected text

    # 3. Draw all contours
    cv2.drawContours(debug_image, contours, -1, (128, 128, 128), 2)  # Gray - all contours

    # 4. Draw the convex hull
    cv2.drawContours(debug_image, [convex_hull], -1, (0, 255, 0), 3)  # Green - convex hull

    # 5. Draw the top edge points (larger circles)
    cv2.circle(debug_image, tuple(top_left), 15, (255, 0, 0), -1)  # Blue - top-left point
    cv2.circle(debug_image, tuple(top_right), 15, (255, 0, 255), -1)  # Magenta - top-right point

    # 6. Draw the final bounding box (thick red rectangle)
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 5)  # Red - final crop box

    # 7. Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3

    # Add legend at the top
    cv2.putText(debug_image, f"Image: {image_name}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(debug_image, f"Crop: ({x}, {y}, {w}, {h})", (10, 90), font, font_scale, (0, 0, 255), font_thickness)

    # Save the debug image
    name_without_ext, ext = os.path.splitext(image_name)
    output_path = os.path.join(output_folder, f"{name_without_ext}_debug{ext}")
    cv2.imwrite(output_path, debug_image)

    return output_path


def safe_process_single_image(image_path):
    try:
        return process_single_image(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()  # This will print the stack trace of the exception

def process_single_image(image_path, output_folder=None):
    """
    Process a single image using the EXACT same cropping logic as process_crop_region
    from src/DB_processing/image_processing.py, with a single comprehensive debug image.
    """
    if output_folder is None:
        output_folder = crop_outputs

    image_name = os.path.basename(image_path)

    # Load grayscale image directly
    original_image = cv2.imread(image_path, 0)
    image = original_image.copy()

    # FIRST PASS: Find edge points without OCR to determine OCR x-range
    # Binary threshold
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Erosion
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=8)

    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort contours by area (largest first)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]

    # Check if second largest contour should be merged
    if len(sorted_contours) > 1:
        second_largest = sorted_contours[1]
        largest_area = cv2.contourArea(largest_contour)
        second_area = cv2.contourArea(second_largest)

        # Merge if second contour is at least 25% of largest
        if second_area >= 0.025 * largest_area:
            # Merge contours by concatenating points
            largest_contour = np.concatenate([largest_contour, second_largest])

    # Convex hull
    convex_hull = cv2.convexHull(largest_contour)

    # Get bounding box from convex hull
    x_coords = convex_hull[:, 0, 0]
    y_coords = convex_hull[:, 0, 1]
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))

    # OCR REGION SETUP: Use convex hull x-range to restrict OCR
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    ocr_padding = 50  # Inset padding from convex hull edges

    # Calculate restricted x-range for OCR (inset from convex hull edges)
    start_col = max(0, x_min + ocr_padding)
    end_col = min(image.shape[1], x_max - ocr_padding)

    # Ensure valid region (if hull is too narrow, use full hull width)
    if start_col >= end_col:
        start_col = x_min
        end_col = x_max

    # Ensure minimum width for OCR region
    min_ocr_width = 10
    if end_col - start_col < min_ocr_width:
        # Expand region around midpoint
        mid_x = (x_min + x_max) // 2
        start_col = max(0, mid_x - min_ocr_width // 2)
        end_col = min(image.shape[1], mid_x + min_ocr_width // 2)

    # OCR in restricted region
    reader_thread = get_reader()
    output = reader_thread.readtext(image[start_row:end_row, start_col:end_col])

    # Find the minimum y-value of detected text (if any)
    text_y_limit = None
    if output:
        # Get the minimum y-coordinate of all detected text boxes
        min_text_y = min(detection[0][0][1] for detection in output)
        text_y_limit = int(min_text_y + start_row)

    # Calculate bounding box from convex hull
    x = x_min
    y = y_min
    w = x_max - x_min

    # Height: stop at text or end of convex hull, whichever comes first
    if text_y_limit is not None:
        h = min(text_y_limit, y_max) - y
    else:
        h = y_max - y

    # Create edge points for visualization (at the top of convex hull)
    top_left = np.array([x_min, y_min])
    top_right = np.array([x_max, y_min])

    # Create single comprehensive debug image (pass start_col and end_col for restricted OCR region)
    create_debug_image(original_image, image_name, output, start_row, start_col, end_col,
                      contours, convex_hull, top_left, top_right, x, y, w, h, output_folder)

    return (image_name, (x, y, w, h))


def get_ultrasound_region(image_folder_path, db_to_process):
    # Construct image paths for only the new data
    image_paths = [os.path.join(image_folder_path, filename) for filename in db_to_process['ImageName']]

    # Collect image data in list
    image_data = []
    
    # Thread pool and TQDM
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(safe_process_single_image, image_path): image_path for image_path in image_paths}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_data.append(result)
                pbar.update()

    return image_data


def load_hard_cases_list(csv_path):
    """Load list of hard case image names from CSV"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return set(df['image'].tolist())


def find_crops_extra():
    

    # Construct image paths for only the new data
    image_paths = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.lower().endswith('.png')]
    
    # Collect image data in list
    image_data = []
    
    # Thread pool and TQDM
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, image_path): image_path for image_path in image_paths}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_data.append(result)
                pbar.update()

    # Write output to CSV with updated headers and row format
    with open(f'{current_dir}/crop_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'X', 'Y', 'Width', 'Height'])  # Updated headers
        for data in image_data:
            image_name, (x, y, w, h) = data  # Unpacking the data
            writer.writerow([image_name, x, y, w, h])  # Writing unpacked data
            
    return image_data

# Load list of hard cases from CSV
csv_path = r"C:\Users\Tristan\Desktop\temp_data_bad_crop_strange_image.csv"
hard_cases_set = load_hard_cases_list(csv_path)
print(f"Loaded {len(hard_cases_set)} hard cases from CSV")

# Get all images from inputs folder
all_image_files = [file for file in os.listdir(image_folder_path) if file.lower().endswith('.png')]

# Separate into hard cases and regular images
hard_case_images = [img for img in all_image_files if img in hard_cases_set]
regular_images = [img for img in all_image_files if img not in hard_cases_set]

print(f"\nProcessing {len(hard_case_images)} hard cases -> crop_images_hard/")
print(f"Processing {len(regular_images)} regular images -> crop_outputs/")

# Process hard cases to crop_images_hard/
hard_case_data = []
if hard_case_images:
    hard_case_paths = [os.path.join(image_folder_path, img) for img in hard_case_images]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, path, crop_images_hard): path
                   for path in hard_case_paths}
        with tqdm(total=len(futures), desc="Hard cases") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    hard_case_data.append(result)
                pbar.update()

# Process regular images to crop_outputs/
regular_data = []
if regular_images:
    regular_paths = [os.path.join(image_folder_path, img) for img in regular_images]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, path, crop_outputs): path
                   for path in regular_paths}
        with tqdm(total=len(futures), desc="Regular images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    regular_data.append(result)
                pbar.update()

# Write hard cases CSV
if hard_case_data:
    with open(f'{current_dir}/crop_data_hard_cases.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'X', 'Y', 'Width', 'Height'])
        for image_name, (x, y, w, h) in hard_case_data:
            writer.writerow([image_name, x, y, w, h])
    print(f"\nSaved hard cases data to: crop_data_hard_cases.csv")

# Write regular images CSV
if regular_data:
    with open(f'{current_dir}/crop_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'X', 'Y', 'Width', 'Height'])
        for image_name, (x, y, w, h) in regular_data:
            writer.writerow([image_name, x, y, w, h])
    print(f"Saved regular images data to: crop_data.csv")

print(f"\nDone! Debug images saved to:")
print(f"  - crop_images_hard/: {len(hard_case_data)} images")
print(f"  - crop_outputs/: {len(regular_data)} images")