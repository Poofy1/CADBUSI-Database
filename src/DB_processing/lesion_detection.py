import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import pytesseract
from itertools import combinations
import re
import os
from PIL import Image
from storage_adapter import *



def get_caliper_inpainted_pairs(csv_file_path):
    """
    Find all image pairs of original caliper images and their inpainted/clean versions.
    
    Two cases are handled:
    1. Inpainted pairs: Images with inpainted_from not null paired with their originals
    2. Closest clean pairs: Images with distance <= 5 and has_calipers = True paired with closest_fn
    
    Ignores pairs where either image has PhotometricInterpretation = 'RGB'
    
    Returns:
        Dictionary containing:
        - 'all_pairs': Combined list of all pairs
    """
    print("Finding caliper and inpainted image pairs...")
    
    # Load the CSV file
    data = read_csv(csv_file_path)
    
    # Create a dictionary for quick lookup of all images by ImageName
    all_images_dict = {}
    for index, row in data.iterrows():
        all_images_dict[row['ImageName']] = {
            'index': index,
            'data': row
        }
    
    # CASE 1: Find inpainted pairs (existing logic)
    inpainted_pairs = []
    inpainted_images = data[data['inpainted_from'].notna()]
    
    # Create a dictionary for original images (no inpainted_from value)
    original_images_dict = {}
    for index, row in data.iterrows():
        if pd.isna(row.get('inpainted_from')):
            original_images_dict[row['ImageName']] = {
                'index': index,
                'data': row
            }
    
    for index, inpainted_row in inpainted_images.iterrows():
        original_filename = inpainted_row['inpainted_from']
        
        if original_filename in original_images_dict:
            original_info = original_images_dict[original_filename]
            
            # Skip if either image has PhotometricInterpretation = 'RGB'
            if (original_info['data'].get('PhotometricInterpretation') == 'RGB' or 
                inpainted_row.get('PhotometricInterpretation') == 'RGB'):
                continue
            
            pair = {
                'type': 'inpainted',
                'caliper_image': original_filename,
                'clean_image': inpainted_row['ImageName']
            }
            
            inpainted_pairs.append(pair)
        else:
            print(f"Warning: Original image '{original_filename}' not found for inpainted image '{inpainted_row['ImageName']}'")
    
    # CASE 2: Find closest clean pairs
    closest_pairs = []
    
    # Filter for images with distance <= 5, has_calipers = True, and closest_fn not null
    closest_candidates = data[
        (data['distance'] <= 5) & 
        (data['has_calipers'] == True) & 
        (data['closest_fn'].notna())
    ]
    
    for index, caliper_row in closest_candidates.iterrows():
        closest_filename = caliper_row['closest_fn']
        
        # Find the corresponding clean image
        if closest_filename in all_images_dict:
            clean_info = all_images_dict[closest_filename]
            
            # Skip if either image has PhotometricInterpretation = 'RGB'
            if (caliper_row.get('PhotometricInterpretation') == 'RGB' or 
                clean_info['data'].get('PhotometricInterpretation') == 'RGB'):
                continue
            
            pair = {
                'type': 'closest_clean',
                'caliper_image': caliper_row['ImageName'],
                'clean_image': closest_filename
            }
            
            closest_pairs.append(pair)
        else:
            print(f"Warning: Closest clean image '{closest_filename}' not found for caliper image '{caliper_row['ImageName']}'")
    
    # Combine all pairs
    all_pairs = inpainted_pairs + closest_pairs
    
    print(f"Found {len(inpainted_pairs)} inpainted pairs")
    print(f"Found {len(closest_pairs)} closest clean pairs")
    print(f"Total: {len(all_pairs)} image pairs")
    
    return all_pairs



def create_difference_masks(pairs, input_folder, threshold=30):
    """
    Load image pairs in grayscale and create binary difference masks.
    
    Args:
        pairs: List of image pairs from get_caliper_inpainted_pairs()
        input_folder: Folder containing the images
        threshold: Threshold value (0-255) for creating binary mask. 
                  Differences above this value will be white (255), below will be black (0)
    
    Returns:
        List of dictionaries containing:
        - 'caliper_image': PIL grayscale image of caliper image
        - 'clean_image': PIL grayscale image of clean image
        - 'difference_mask': PIL binary mask showing differences
    """
    print(f"Creating difference masks for {len(pairs)} image pairs...")
    
    results = []
    successful_count = 0
    
    for i, pair in enumerate(pairs):
        try:
            # Get image paths
            caliper_path = os.path.join(input_folder, pair['caliper_image'])
            clean_path = os.path.join(input_folder, pair['clean_image'])
            caliper_path = os.path.normpath(caliper_path)
            clean_path = os.path.normpath(clean_path)
            
            # Check if files exist
            if not os.path.exists(caliper_path):
                print(f"Warning: Caliper image not found: {caliper_path}")
                continue
                
            if not os.path.exists(clean_path):
                print(f"Warning: Clean image not found: {clean_path}")
                continue
            
            # Load images and convert to grayscale
            caliper_img = read_image(caliper_path, use_pil=True).convert('L')
            clean_img = read_image(clean_path, use_pil=True).convert('L')
            caliper_array = np.array(caliper_img, dtype=np.float32)
            clean_array = np.array(clean_img, dtype=np.float32)
            
            # Create binary mask (threshold the difference)
            difference_array = np.abs(caliper_array - clean_array)
            mask_array = np.where(difference_array > threshold, 255, 0).astype(np.uint8)
            mask_img = Image.fromarray(mask_array, mode='L')
            
            # Store results
            result = {
                'caliper_image': caliper_img,
                'clean_image': clean_img,
                'difference_mask': mask_img,
            }
            
            results.append(result)
            successful_count += 1
            
        except Exception as e:
            print(f"Error processing pair {i+1} ({pair['caliper_image']} -> {pair['clean_image']}): {str(e)}")
    
    print(f"Successfully processed {successful_count}/{len(pairs)} image pairs")
    return results








def extract_expected_lengths(img_gray):
    h, w = img_gray.shape
    check_row = img_gray[-3, :]
    nonblack_x = np.where(check_row > 20)[0]
    x1, x2 = (nonblack_x[0], nonblack_x[-1]) if len(nonblack_x) > 0 else (0, w - 1)
    row_sums = np.sum(img_gray[:, x1:x2] > 100, axis=1)
    white_rows = np.where(row_sums > 0.9 * (x2 - x1))[0]
    white_rows = white_rows[white_rows > h-73]
    y1, y2 = (white_rows[0], h-2) if white_rows[0] < 650 else (white_rows[3], h-2)
    legend_crop = img_gray[y1:y2, x1:x2]
    config = "--psm 11 -c tessedit_char_whitelist=0123456789.Lcm "
    text = pytesseract.image_to_string(legend_crop, config=config).replace("\n", " ").replace(":", "")
    matches = re.findall(r"\d+\.?\d*\s*cm", text)
    lengths = [float(m.replace("cm", "").strip()) for m in matches]
    return lengths, (x1, y1, x2, y2)

def do_segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c): return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def pair_calipers_best_matching(centers, expected_lengths_cm, px_spacing_mm):
    expected_px = [l * 10 / px_spacing_mm for l in expected_lengths_cm]
    candidate_pairs = list(combinations(range(len(centers)), 2))
    distances = [(i, j, np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))) for i, j in candidate_pairs]
    pairs, used = [], set()
    for target_px in expected_px:
        best_match = None
        best_error = float("inf")
        for i, j, d in distances:
            if i in used or j in used: continue
            err = abs(d - target_px)
            if err < best_error and err / target_px < 0.2:
                best_error = err
                best_match = (i, j)
        if best_match:
            pairs.append(best_match)
            used.update(best_match)
    intersects = False
    if len(pairs) == 2:
        i1, j1 = pairs[0]
        i2, j2 = pairs[1]
        intersects = do_segments_intersect(centers[i1], centers[j1], centers[i2], centers[j2])
    return pairs, intersects

def compute_oriented_bbox_and_cyan_bbox(centers):
    pts = np.array(centers, dtype=np.float32)
    max_dist, anchor = 0, (pts[0], pts[1])
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist, anchor = d, (pts[i], pts[j])
    p1, p2 = anchor
    angle = np.arctan2(*(p2 - p1)[::-1])
    R = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated = (pts - p1) @ R.T
    x0, y0 = rotated.min(axis=0)
    x1, y1 = rotated.max(axis=0)
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    box_coords = (box @ R) + p1
    xmin, ymin = box_coords.min(axis=0)
    xmax, ymax = box_coords.max(axis=0)
    return box_coords, [int(xmin), int(ymin), int(xmax), int(ymax)]



def detect_calipers_from_mask(difference_mask, min_area=20, max_area=500, morphology_cleanup=True):
    """
    Extract caliper locations from a difference mask.
    
    Args:
        difference_mask: PIL Image or numpy array of the binary difference mask
        min_area: Minimum area of connected components to consider as calipers
        max_area: Maximum area of connected components to consider as calipers
        morphology_cleanup: Apply morphological operations to clean up mask
    
    Returns:
        List of (x, y) coordinates of detected caliper centers
    """
    # Convert to numpy if PIL Image
    if hasattr(difference_mask, 'convert'):
        mask_array = np.array(difference_mask)
    else:
        mask_array = difference_mask
    
    # Optional morphological cleanup
    if morphology_cleanup:
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel_open)
        
        # Fill small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel_close)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
    
    caliper_centers = []
    
    for i in range(1, num_labels):  # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_area <= area <= max_area:
            # Additional filtering based on shape (optional)
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                # Calculate circularity (calipers are typically somewhat circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Accept if reasonably circular (adjust threshold as needed)
                    if 0.1 <= circularity <= 1.2:
                        cx, cy = centroids[i]
                        caliper_centers.append((int(cx), int(cy)))
    
    return caliper_centers

def process_single_image_pair(pair_data, mask_result, image_data_row):
    """
    Process a single image pair to detect calipers and compute bounding boxes.
    
    Args:
        pair_data: Dictionary with pair information
        mask_result: Result from create_difference_masks for this pair
        image_data_row: Row from the CSV for this image
    
    Returns:
        Dictionary with detection results
    """
    # Extract caliper centers from mask
    caliper_centers = detect_calipers_from_mask(mask_result['difference_mask'])
    
    # Get parameters for pairing
    px_spacing = float(image_data_row.get("delta_x", 0.1)) * 10 if "delta_x" in image_data_row else 1.0
    
    # Load the full image to extract expected lengths
    # Note: You might need to adapt this path construction based on your setup
    img_path = os.path.join("your_image_dir", image_data_row["filename"])  # Update this path
    try:
        img = read_image(img_path, use_pil=True).convert('L')
        img_array = np.array(img)
        expected_lengths, _ = extract_expected_lengths(img_array)
    except:
        expected_lengths = [2.0, 1.5]  # Default values if OCR fails
    
    result = {
        'caliper_centers': caliper_centers,
        'pairs': [],
        'caliper_pairs': "",
        'caliper_box': "",
        'status': "undetected"
    }
    
    if len(caliper_centers) == 2:
        # Simple case: 2 calipers = 1 pair
        result['pairs'] = [(0, 1)]
        result['caliper_pairs'] = str([caliper_centers])
        result['status'] = "1_pair"
        
    elif len(caliper_centers) == 4:
        # Complex case: 4 calipers = 2 pairs
        pairs, intersects = pair_calipers_best_matching(caliper_centers, expected_lengths, px_spacing)
        
        if pairs:
            result['pairs'] = pairs
            result['caliper_pairs'] = str([(caliper_centers[i], caliper_centers[j]) for i, j in pairs])
            
            if intersects:
                result['status'] = "2_pairs_intersect"
                # Compute bounding box
                pts = [caliper_centers[i] for pair in pairs for i in pair]
                _, bbox = compute_oriented_bbox_and_cyan_bbox(pts)
                result['caliper_box'] = str(bbox)
            else:
                result['status'] = "2_pairs_nonintersect"
    
    return result

def Locate_Lesions(csv_file_path, image_dir):
    
    print("Starting lesion location using mask-based caliper detection...")
    
    # Load image data
    image_data = read_csv(csv_file_path)
    
    # Add new columns if they don't exist
    for col in ["caliper_pairs", "caliper_box"]:
        if col not in image_data.columns:
            image_data[col] = ""
    
    # Get image pairs for mask-based detection
    pairs = get_caliper_inpainted_pairs(csv_file_path)
    
    if not pairs:
        print("No image pairs found for mask-based detection.")
        return None
    
    # Create difference masks
    mask_results = create_difference_masks(pairs, image_dir)
    
    # Create lookup dictionary using CLEAN image filename as key
    mask_lookup = {}
    for i, pair in enumerate(pairs):
        if i < len(mask_results):
            mask_lookup[pair['clean_image']] = {
                'pair_data': pair,
                'mask_result': mask_results[i]
            }
    
    # Process each row
    processed_count = 0
    for idx, row in image_data.iterrows():
        filename = row["filename"]
        
        # Check if this is a clean image that has mask data
        if filename in mask_lookup:
            # Use mask-based detection
            mask_info = mask_lookup[filename]
            detection_result = process_single_image_pair(
                mask_info['pair_data'],
                mask_info['mask_result'],
                row
            )
            
            # Update the CSV row (this will be the clean image row)
            image_data.at[idx, "caliper_pairs"] = detection_result['caliper_pairs']
            image_data.at[idx, "caliper_box"] = detection_result['caliper_box']
            
            processed_count += 1
    
    # Save updated CSV
    save_data(image_data, csv_file_path)
    
    print(f"Processed {processed_count} images with mask-based detection")