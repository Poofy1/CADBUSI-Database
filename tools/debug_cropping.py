import cv2, sys, csv
import numpy as np
import os
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.image_processing import *

parent_dir = os.path.dirname(parent_dir)
current_dir = f'{parent_dir}/tools/'
print(current_dir)

image_output = f"{current_dir}/debug_output/"
os.makedirs(image_output, exist_ok=True)


def save_step_image(image, image_name, step_num, step_desc):
    """Save an image for a specific processing step with numbered, descriptive filename"""
    # Make sure image is in color format for visualization
    if len(image.shape) == 2 or image.shape[2] == 1:  # If grayscale
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Parse image name to create a descriptive filename
    name_without_ext, ext = os.path.splitext(image_name)
    # Create numbered descriptive filename: 01_image001_original.png
    descriptive_filename = f"{step_num:02d}_{name_without_ext}_{step_desc}{ext}"
    
    # Save the image in the single output directory
    output_path = os.path.join(image_output, descriptive_filename)
    cv2.imwrite(output_path, vis_image)
    return output_path


def safe_process_single_image(image_path):
    try:
        return process_single_image(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()  # This will print the stack trace of the exception


def Show_Crop(image_path, image, contours, x, y, w, h):
    # Check if the image was loaded properly
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    # Check if the box is too vertical
    aspect_ratio = w / h
    if aspect_ratio < 0.25:  # Too vertical
        print(f"Image is too vertical: {image_path}")
    elif aspect_ratio > 4:  # Too horizontal
        print(f"Image is too horizontal: {image_path}")
    # Check if the resolution is less than 200x200
    if w < 200 or h < 200:
        print(f"Image resolution is too small: {image_path}")
        
    # Create a copy for visualization
    if len(image.shape) == 2 or image.shape[2] == 1:  # Check if the image is grayscale
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw the bounding box only (no contours)
    start_point = (x, y)
    end_point = (x + w, y + h)
    color = (0, 0, 255)  # Red color in BGR
    thickness = 10
    vis_image = cv2.rectangle(vis_image, start_point, end_point, color, thickness)
    
    # Save the final result
    save_step_image(vis_image, image_path, 9, "final_result")


def find_top_edge_points(largest_contour, vertical_range):
    # Find the highest y-coordinate in the contour
    top_y = min(largest_contour[:, 0, 1])

    # Filter points that are within the vertical range from the top y-coordinate
    top_edge_points = [pt[0] for pt in largest_contour if top_y <= pt[0][1] <= top_y + vertical_range]

    # Find the leftmost and rightmost points from the filtered top edge points
    top_left = min(top_edge_points, key=lambda x: x[0])
    top_right = max(top_edge_points, key=lambda x: x[0])

    return top_left, top_right


def process_single_image(image_path):
    image_name = os.path.basename(image_path)  # Get image name from image path
    
    # Step 1: Load grayscale image directly
    original_image = cv2.imread(image_path, 0)
    image = original_image.copy()  # Make a copy to work with
    save_step_image(image, image_name, 1, "original")
    
    # Calculate the start and end row indices for the bottom quarter of the image
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    margin = 10
    
    # Step 2: Remove caliper box
    reader_thread = get_reader()
    output = reader_thread.readtext(image[start_row:end_row, :])
    
    # Create a copy for visualization before modifying
    image_with_text_boxes = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    for detection in output:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        
        # Draw text boxes on the visualization image
        cv2.rectangle(image_with_text_boxes, 
                     (top_left[0], top_left[1] + start_row), 
                     (bottom_right[0], bottom_right[1] + start_row), 
                     (0, 255, 0), 10)
    
    save_step_image(image_with_text_boxes, image_name, 2, "detected_text")
    
    # Apply text removal to the original grayscale image
    for detection in output:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        top_left = (0, max(0, top_left[1] + start_row - margin))
        bottom_right = (image.shape[1], min(image.shape[0], bottom_right[1] + start_row + margin))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    
    save_step_image(image, image_name, 3, "caliper_removed")
    
    # Step 3: Create binary mask
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    save_step_image(binary_mask, image_name, 4, "binary_mask")
    
    # Step 4: Erode the binary mask
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=5)
    save_step_image(eroded_mask, image_name, 5, "eroded_mask")

    # Step 5: Find contours and get the largest one
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # Check if contours is empty
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Visualize the largest contour
    contour_image = cv2.cvtColor(eroded_mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 0, 255), 10)
    save_step_image(contour_image, image_name, 6, "largest_contour")
    
    # Step 6: Get the convex hull
    convex_hull = cv2.convexHull(largest_contour)
    
    # Visualize the convex hull
    hull_image = cv2.cvtColor(eroded_mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(hull_image, [convex_hull], -1, (0, 255, 0), 10)
    save_step_image(hull_image, image_name, 7, "convex_hull")
    
    # Step 7: Find top edge points and bounding box
    top_left, top_right = find_top_edge_points(convex_hull, vertical_range=20)

    # Visualize the top edge points
    edge_image = hull_image.copy()
    cv2.circle(edge_image, tuple(top_left), 20, (255, 0, 0), -1)  # Blue for top-left
    cv2.circle(edge_image, tuple(top_right), 20, (0, 0, 255), -1)  # Red for top-right
    save_step_image(edge_image, image_name, 8, "top_edge_points")

    # Calculate the bounding box
    x = top_left[0]
    y = top_left[1]
    w = top_right[0] - x
    h = max(convex_hull[:, 0, 1]) - y  # Bottom y-coordinate - top y-coordinate

    # Step 8: Show and save the final result
    Show_Crop(image_name, original_image, convex_hull, x, y, w, h)
    
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


def find_crops_extra():
    image_folder_path = f'{current_dir}/inputs/'

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

#image_masks = get_ultrasound_region(image_folder_path, image_df)
image_masks = find_crops_extra()