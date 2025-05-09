
import cv2, sys, csv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.image_processing import *

current_dir = f'{parent_dir}/debug_tools/'



image_output = f"{current_dir}/debug_output/"
os.makedirs(image_output, exist_ok=True)


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
        
        
    if len(image.shape) == 2 or image.shape[2] == 1:  # Check if the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
    
    # Draw the box
    start_point = (x, y)
    end_point = (x + w, y + h)
    color = (0, 0, 255)  # Blue color in BGR
    thickness = 4
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    # Save the image with bounding box
    output_path = os.path.join(image_output, image_path)
    cv2.imwrite(output_path, image)


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
    
    image = cv2.imread(image_path, 0)  # Load grayscale image directly

    # Calculate the start and end row indices for the bottom quarter of the image
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    margin = 10
    
    # Remove caliper box
    reader_thread = get_reader()
        
    output = reader_thread.readtext(image[start_row:end_row, :])
    for detection in output:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        top_left = (0, max(0, top_left[1] + start_row - margin))
        bottom_right = (image.shape[1], min(image.shape[0], bottom_right[1] + start_row + margin))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        
        
    # Create binary mask of nonzero pixels
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=5)

    # Find contours and get the largest one
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # Check if contours is empty
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    convex_hull = cv2.convexHull(largest_contour)
    
    # Use the function to find the top edge points
    top_left, top_right = find_top_edge_points(convex_hull, vertical_range=20)

    # Now you have the top left and top right points, use them to find x, y, w, h
    x = top_left[0]
    y = top_left[1]
    w = top_right[0] - x
    h = max(convex_hull[:, 0, 1]) - y  # Bottom y-coordinate - top y-coordinate

    Show_Crop(image_name, image, [convex_hull], x, y, w, h)
    
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








