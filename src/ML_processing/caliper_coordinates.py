import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from tqdm import tqdm
from src.ML_processing.lesion_detection import clamp_coordinates
from tools.storage_adapter import *
from src.DB_processing.tools import get_reader, reader, append_audit
from src.DB_processing.database import DatabaseManager
env = os.path.dirname(os.path.abspath(__file__))
    
    
    
def get_caliper_inpainted_pairs(db):
    """
    Find all image pairs of original caliper images and their inpainted/clean versions.
    
    Two cases are handled:
    1. Inpainted pairs: Images with inpainted_from not null paired with their originals
    2. Closest clean pairs: Images with distance <= 5 and has_calipers = True paired with closest_fn
    
    Ignores pairs where either image has photometric_interpretation = 'RGB'
    """
    print("Finding caliper and inpainted image pairs...")
    
    # Load image data from database
    data = db.get_images_dataframe()
    
    # Create a dictionary for quick lookup of all images by image_name
    all_images_dict = {}
    for index, row in data.iterrows():
        all_images_dict[row['image_name']] = {
            'index': index,
            'data': row
        }
    
    # CASE 1: Find inpainted pairs (existing logic)
    inpainted_pairs = []
    inpainted_images = data[data['inpainted_from'].notna() & (data['inpainted_from'] != '')]

    # Create a dictionary for original images (no inpainted_from value or empty string)
    original_images_dict = {}
    for index, row in data.iterrows():
        inpainted_from = row.get('inpainted_from')
        if pd.isna(inpainted_from) or inpainted_from == '':
            original_images_dict[row['image_name']] = {
                'index': index,
                'data': row
            }
    
    for index, inpainted_row in inpainted_images.iterrows():
        original_filename = inpainted_row['inpainted_from']
        
        if original_filename in original_images_dict:
            original_info = original_images_dict[original_filename]
            
            # Skip if either image has photometric_interpretation = 'RGB'
            if (original_info['data'].get('photometric_interpretation') == 'RGB' or 
                inpainted_row.get('photometric_interpretation') == 'RGB'):
                continue
            
            pair = {
                'type': 'inpainted',
                'caliper_image': original_filename,
                'clean_image': inpainted_row['image_name']
            }
            
            inpainted_pairs.append(pair)
        else:
            print(f"Warning: Original image '{original_filename}' not found for inpainted image '{inpainted_row['image_name']}'")
    
    # CASE 2: Find closest clean pairs
    closest_pairs = []
    
    # Filter for images with distance <= 5, has_calipers = True, and closest_fn not null
    closest_candidates = data[
        (data['distance'] <= 5) & (data['distance'] > 0) & 
        (data['has_calipers'] == True) & 
        (data['closest_fn'].notna())
    ]
    
    for index, caliper_row in closest_candidates.iterrows():
        closest_filename = caliper_row['closest_fn']
        
        # Find the corresponding clean image
        if closest_filename in all_images_dict:
            clean_info = all_images_dict[closest_filename]
            
            # Skip if either image has photometric_interpretation = 'RGB'
            if (caliper_row.get('photometric_interpretation') == 'RGB' or 
                clean_info['data'].get('photometric_interpretation') == 'RGB'):
                continue
            
            pair = {
                'type': 'closest_clean',
                'caliper_image': caliper_row['image_name'],
                'clean_image': closest_filename
            }
            
            closest_pairs.append(pair)
        else:
            print(f"Warning: Closest clean image '{closest_filename}' not found for caliper image '{caliper_row['image_name']}'")
    
    # Combine all pairs
    all_pairs = inpainted_pairs + closest_pairs
    
    print(f"Found {len(inpainted_pairs)} inpainted pairs")
    print(f"Found {len(closest_pairs)} closest clean pairs")
    print(f"Total: {len(all_pairs)} image pairs")
    return all_pairs


def create_difference_mask(caliper_path, clean_path, input_folder, threshold=30):
    
    # Get image paths
    caliper_path = os.path.join(input_folder, caliper_path)
    clean_path = os.path.join(input_folder, clean_path)
    caliper_path = os.path.normpath(caliper_path)
    clean_path = os.path.normpath(clean_path)
    
    # Load images and convert to grayscale
    caliper_img = read_image(caliper_path, use_pil=True).convert('L')
    clean_img = read_image(clean_path, use_pil=True).convert('L')
    caliper_array = np.array(caliper_img, dtype=np.float32)
    clean_array = np.array(clean_img, dtype=np.float32)
    
    # Create binary mask (threshold the difference)
    difference_array = np.abs(caliper_array - clean_array)
    mask_array = np.where(difference_array > threshold, 255, 0).astype(np.uint8)
    mask_img = Image.fromarray(mask_array, mode='L')

    return caliper_img, clean_img, mask_img


def detect_calipers_from_mask(difference_mask, min_area=1, max_area=1000, merge_distance=20,
                            crop_x=None, crop_y=None, crop_x2=None, crop_y2=None, edge_margin=20):
    """
    Improved caliper detection with more lenient parameters and multi-pass detection.
    """
    # Convert to numpy if PIL Image
    if hasattr(difference_mask, 'convert'):
        mask_array = np.array(difference_mask)
    else:
        mask_array = difference_mask.copy()

    # Ensure binary mask
    if len(mask_array.shape) == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

    # Get image dimensions first for validation
    img_height, img_width = mask_array.shape

    # Validate and clamp crop parameters to image bounds
    if crop_x is not None:
        crop_x = max(0, min(crop_x, img_width - 1))
    if crop_y is not None:
        crop_y = max(0, min(crop_y, img_height - 1))
    if crop_x2 is not None:
        crop_x2 = max(0, min(crop_x2, img_width))
    if crop_y2 is not None:
        crop_y2 = max(0, min(crop_y2, img_height))

    # Try multiple binary thresholds to catch different intensities
    thresholds = [30, 50, 80, 127]  # Multiple sensitivity levels
    all_caliper_centers = []

    for threshold in thresholds:
        binary_mask = (mask_array > threshold).astype(np.uint8) * 255

        # Preprocess to enhance thin structures
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        enhanced_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Tile dimensions for exclusion zones
        tile_height = img_height // 3
        tile_width = img_width // 3

        # Calculate exclusion zones
        bottom_right_x_start = 2 * tile_width
        bottom_right_y_start = 2 * tile_height
        bottom_left_x_end = tile_width
        bottom_left_y_start = 2 * tile_height

        # Calculate safe zone within crop region
        safe_crop_x = crop_x + edge_margin if crop_x is not None else edge_margin
        safe_crop_y = crop_y + edge_margin if crop_y is not None else edge_margin
        safe_crop_x2 = crop_x2 - edge_margin if crop_x2 is not None else img_width - edge_margin
        safe_crop_y2 = crop_y2 - edge_margin if crop_y2 is not None else img_height - edge_margin

        # Clamp safe crop zone to image bounds
        safe_crop_x = max(0, min(safe_crop_x, img_width - 1))
        safe_crop_y = max(0, min(safe_crop_y, img_height - 1))
        safe_crop_x2 = max(0, min(safe_crop_x2, img_width))
        safe_crop_y2 = max(0, min(safe_crop_y2, img_height))
        
        # Find contours
        contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # More lenient aspect ratio check
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio > 8:  # More lenient
                    continue
                
                # Extract ROI with padding for better analysis
                pad = 2
                roi_x1 = max(0, x - pad)
                roi_y1 = max(0, y - pad)
                roi_x2 = min(img_width, x + w + pad)
                roi_y2 = min(img_height, y + h + pad)
                roi = enhanced_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if is_cross_crosshair_or_x_shape(roi):
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # CRITICAL: Validate coordinates are within image bounds
                        if cx < 0 or cx >= img_width or cy < 0 or cy >= img_height:
                            continue

                        # Apply exclusion filters
                        if (bottom_right_x_start <= cx < img_width and
                            bottom_right_y_start <= cy < img_height):
                            continue

                        if (0 <= cx < bottom_left_x_end and
                            bottom_left_y_start <= cy < img_height):
                            continue

                        if (crop_x is not None and crop_y is not None and
                            crop_x2 is not None and crop_y2 is not None):
                            if not (safe_crop_x <= cx <= safe_crop_x2 and
                                   safe_crop_y <= cy <= safe_crop_y2):
                                continue

                        all_caliper_centers.append((cx, cy))
    
    # Remove duplicates from multiple thresholds
    if all_caliper_centers:
        all_caliper_centers = merge_nearby_centers(all_caliper_centers, merge_distance)

        # Final validation: ensure all coordinates are within bounds after merging
        validated_centers = []
        for cx, cy in all_caliper_centers:
            if 0 <= cx < img_width and 0 <= cy < img_height:
                validated_centers.append((cx, cy))
            else:
                print(f"Warning: Merged coordinate ({cx},{cy}) out of bounds ({img_width},{img_height}). Skipping.")

        all_caliper_centers = validated_centers

    return all_caliper_centers

def merge_nearby_centers(centers, merge_distance):
    """
    Merge centers that are within merge_distance of each other by averaging their positions.
    """
    if len(centers) <= 1:
        return centers
    
    centers = np.array(centers)
    merged_centers = []
    used = set()
    
    for i, center in enumerate(centers):
        if i in used:
            continue
            
        # Find all centers within merge_distance of this center
        cluster = [i]
        for j, other_center in enumerate(centers):
            if j != i and j not in used:
                distance = np.linalg.norm(center - other_center)
                if distance <= merge_distance:
                    cluster.append(j)
        
        # Mark all centers in this cluster as used
        used.update(cluster)
        
        # Calculate average position of the cluster
        cluster_centers = centers[cluster]
        avg_center = np.mean(cluster_centers, axis=0)
        merged_centers.append((int(avg_center[0]), int(avg_center[1])))
    
    return merged_centers


def is_cross_crosshair_or_x_shape(roi):
    """
    Improved version with better thin line detection and multiple validation approaches.
    """
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return False
    
    # Ensure binary
    roi = (roi > 127).astype(np.uint8) * 255
    
    # METHOD 1: Enhanced morphological approach
    # Create specific kernels for horizontal and vertical line detection
    h_size = max(3, roi.shape[1] // 4)  # Adaptive horizontal kernel
    v_size = max(3, roi.shape[0] // 4)  # Adaptive vertical kernel
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    
    # Extract horizontal and vertical lines separately
    horizontal_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vertical_kernel)
    
    # Check if we have both horizontal and vertical components
    has_horizontal = np.sum(horizontal_lines > 0) > 2
    has_vertical = np.sum(vertical_lines > 0) > 2
    
    if has_horizontal and has_vertical:
        return True
    
    # METHOD 2: Enhanced Hough line detection with multiple parameter sets
    # Preprocess for better edge detection
    kernel = np.ones((3,3), np.uint8)
    enhanced_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    enhanced_roi = cv2.morphologyEx(enhanced_roi, cv2.MORPH_OPEN, kernel)
    
    # More aggressive edge detection for thin lines
    edges = cv2.Canny(enhanced_roi, 20, 60, apertureSize=3)
    
    # Try multiple Hough parameter combinations
    hough_params = [
        (1, np.pi/180, max(1, min(roi.shape)//8)),   # Very sensitive
        (1, np.pi/180, max(2, min(roi.shape)//6)),   # Medium sensitive  
        (1, np.pi/180, max(2, min(roi.shape)//4)),   # Less sensitive
        (2, np.pi/90, max(1, min(roi.shape)//8)),    # Different resolution
    ]
    
    all_lines = []
    for rho, theta, threshold in hough_params:
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        if lines is not None:
            all_lines.extend(lines)
    
    if len(all_lines) >= 2:
        # Enhanced angle analysis
        angles = [line[0][1] for line in all_lines]
        angles = np.array(angles)
        
        # Remove very similar angles (within 0.05 radians)
        unique_angles = []
        for angle in angles:
            if not any(abs(angle - ua) < 0.05 for ua in unique_angles):
                unique_angles.append(angle)
        
        angles = np.array(unique_angles)
        
        if len(angles) >= 2:
            # More lenient angle checks
            horizontal = (np.sum(np.abs(angles - 0) < 0.3) + 
                         np.sum(np.abs(angles - np.pi) < 0.3))
            vertical = np.sum(np.abs(angles - np.pi/2) < 0.3)
            
            diagonal1 = np.sum(np.abs(angles - np.pi/4) < 0.4)
            diagonal2 = np.sum(np.abs(angles - 3*np.pi/4) < 0.4)
            
            # Check for any perpendicular pairs
            perpendicular_pairs = 0
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    angle_diff = abs(angles[i] - angles[j])
                    if (abs(angle_diff - np.pi/2) < 0.4 or 
                        abs(angle_diff - 3*np.pi/2) < 0.4):
                        perpendicular_pairs += 1
            
            is_cross_or_crosshair = (horizontal > 0 and vertical > 0)
            is_x_pattern = (diagonal1 > 0 and diagonal2 > 0)
            has_perpendicular_lines = perpendicular_pairs > 0
            
            if is_cross_or_crosshair or is_x_pattern or has_perpendicular_lines:
                return True
    
    # METHOD 3: Template matching approach for very small ROIs
    if roi.shape[0] <= 10 or roi.shape[1] <= 10:
        # Simple cross templates for tiny calipers
        templates = [
            np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8) * 255,  # Plus
            np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8) * 255,  # X
        ]
        
        for template in templates:
            if template.shape[0] <= roi.shape[0] and template.shape[1] <= roi.shape[1]:
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                if np.max(result) > 0.3:  # Lower threshold for small shapes
                    return True
    
    return False

def save_debug_image(caliper_img, mask_img, caliper_centers, output_path, max_distance_cm=None):
    # Convert PIL to numpy array for drawing
    img_array = np.array(caliper_img.convert('RGB'))

    # Draw circle and coordinates at each caliper center
    for cx, cy in caliper_centers:
        cv2.circle(img_array, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle
        cv2.putText(img_array, f"({cx},{cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display max distance in cm if available
    if max_distance_cm is not None:
        text = f"Max Distance: {max_distance_cm:.2f} cm"
        cv2.putText(img_array, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Convert mask to RGB for side-by-side comparison
    mask_array = np.array(mask_img.convert('RGB'))

    # Concatenate horizontally
    combined = np.hstack([img_array, mask_array])

    # Convert back to PIL and save
    debug_img = Image.fromarray(combined)
    debug_img.save(output_path)


def process_single_image_pair(pair, image_dir, image_data_row, save_debug=False, debug_dir=None):
    """
    Process a single image pair to detect calipers and get their center coordinates.
    """
    caliper_path = pair['caliper_image']
    clean_path = pair['clean_image']

    # Extract caliper centers from mask
    caliper_img, clean_img, mask_img = create_difference_mask(caliper_path, clean_path, image_dir)

    # Find Calipers
    # Mask out everything outside the crop region
    crop_x, crop_y, crop_w, crop_h = None, None, None, None
    crop_x2, crop_y2 = None, None  # Initialize to None to avoid undefined variable error

    if all(col in image_data_row for col in ['crop_x', 'crop_y', 'crop_h', 'crop_w']):
        crop_x = int(image_data_row['crop_x']) if pd.notna(image_data_row['crop_x']) else 0
        crop_y = int(image_data_row['crop_y']) if pd.notna(image_data_row['crop_y']) else 0
        crop_w = int(image_data_row['crop_w']) if pd.notna(image_data_row['crop_w']) else mask_img.width
        crop_h = int(image_data_row['crop_h']) if pd.notna(image_data_row['crop_h']) else mask_img.height

        # Convert PIL Image to numpy array for masking
        mask_array = np.array(mask_img)

        # Create a mask that blacks out everything outside the crop region
        masked_array = np.zeros_like(mask_array)

        # Ensure crop coordinates are within image bounds
        img_h, img_w = mask_array.shape
        crop_x, crop_y, crop_x2, crop_y2 = clamp_coordinates(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h,
                                                            img_w, img_h
                                                        )

        # Copy only the crop region to the masked array
        if crop_x2 > crop_x and crop_y2 > crop_y:
            masked_array[crop_y:crop_y2, crop_x:crop_x2] = mask_array[crop_y:crop_y2, crop_x:crop_x2]

        # Convert back to PIL Image
        mask_img = Image.fromarray(masked_array, mode='L')

    caliper_centers = detect_calipers_from_mask(
        mask_img,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        edge_margin=30
    )

    # Filter out instances with 10 or more caliper centers (likely false positives)
    if len(caliper_centers) >= 10:
        caliper_centers = []

    # Format coordinates as "x,y;x,y;x,y;" string
    if caliper_centers:
        caliper_coordinates_str = ';'.join([f"{cx},{cy}" for cx, cy in caliper_centers]) + ';'
    else:
        caliper_coordinates_str = ''

    # Calculate max distance between caliper centers in cm
    max_distance_cm = None
    if len(caliper_centers) >= 2:
        # Get physical_delta_x from the image metadata (distance per pixel)
        physical_delta_x = image_data_row.get('physical_delta_x')

        if physical_delta_x is not None and not pd.isna(physical_delta_x):
            physical_delta_x = float(physical_delta_x)

            # Calculate all pairwise distances and find the maximum
            max_distance_pixels = 0
            for i in range(len(caliper_centers)):
                for j in range(i + 1, len(caliper_centers)):
                    cx1, cy1 = caliper_centers[i]
                    cx2, cy2 = caliper_centers[j]
                    distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                    max_distance_pixels = max(max_distance_pixels, distance)

            # Convert to cm (physical_delta_x is already in cm per pixel)
            max_distance_cm = max_distance_pixels * physical_delta_x

    # Save debug image if requested
    if save_debug and debug_dir and caliper_centers:
        os.makedirs(debug_dir, exist_ok=True)
        # Use just the base filename to avoid path issues
        base_filename = os.path.basename(caliper_path)
        debug_path = os.path.join(debug_dir, f"debug_{base_filename}")
        save_debug_image(caliper_img, mask_img, caliper_centers, debug_path, max_distance_cm)

    return {
        'caliper_coordinates': caliper_coordinates_str
    }


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image_pairs_multithreading(pairs, image_dir, image_data_df, num_threads=6, save_debug=False, debug_dir=None):
    """
    Process multiple image pairs using multithreading with tqdm progress bar
    """

    # Pre-compute lookup dictionary ONCE
    image_name_to_row = {}
    for idx, row in image_data_df.iterrows():
        image_name_to_row[row['image_name']] = row

    def worker(pair):
        # O(1) dictionary lookup
        caliper_name = pair['caliper_image']
        image_data_row = image_name_to_row[caliper_name]

        return process_single_image_pair(pair, image_dir, image_data_row, save_debug, debug_dir)

    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks with index tracking
        future_to_index = {}
        for i, pair in enumerate(pairs):
            future = executor.submit(worker, pair)
            future_to_index[future] = i

        # Collect results with tqdm progress bar
        with tqdm(total=len(pairs), desc="Processing image pairs") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    pair = pairs[idx]
                    print(f"Error processing {pair['caliper_image']}: {e}")
                    results[idx] = None

                pbar.update(1)

    # Convert back to list in correct order
    return [results[i] for i in range(len(pairs))]
    
def Locate_Calipers(image_dir, save_debug=False, debug_dir='debug_caliper_coords'):
    print("Starting caliper location using mask-based detection...")

    with DatabaseManager() as db:
        # Ensure caliper_coordinates columns exist in the database
        db.add_column_if_not_exists('Images', 'caliper_coordinates', 'TEXT')

        # Load image data from database
        image_data = db.get_images_dataframe()

        # Get image pairs for mask-based detection
        pairs = get_caliper_inpainted_pairs(db)

        if not pairs:
            print("No image pairs found for mask-based detection.")
            return None

        # Create a mapping from image_name to row index for faster lookup
        image_name_to_idx = {row['image_name']: idx for idx, row in image_data.iterrows()}

        # Prepare valid pairs with their corresponding rows
        valid_pairs = []
        pair_to_clean_idx = {}

        for pair in pairs:
            clean_image_name = pair['clean_image']

            if clean_image_name in image_name_to_idx:
                clean_idx = image_name_to_idx[clean_image_name]
                valid_pairs.append(pair)
                pair_to_clean_idx[len(valid_pairs) - 1] = clean_idx
            else:
                print(f"Warning: Clean image '{clean_image_name}' not found in database")

        if not valid_pairs:
            print("No valid image pairs found.")
            return

        print(f"Processing {len(valid_pairs)} image pairs...")
        if save_debug:
            print(f"Debug images will be saved to: {debug_dir}")

        # Process pairs
        results = process_image_pairs_multithreading(valid_pairs, image_dir, image_data, save_debug=save_debug, debug_dir=debug_dir)

        # Update the database with results
        cursor = db.conn.cursor()
        processed_count = 0

        for i, result in enumerate(results):
            if result is not None:
                caliper_image_name = valid_pairs[i]['caliper_image']

                cursor.execute("""
                    UPDATE Images
                    SET caliper_coordinates = ?
                    WHERE image_name = ?
                """, (result['caliper_coordinates'],
                      caliper_image_name))

                processed_count += 1
            else:
                print(f"Warning: Failed to process pair {i}")

        db.conn.commit()
        print(f"Processed {processed_count} image pairs with caliper coordinate detection")