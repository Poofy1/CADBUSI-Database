import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import json
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


def find_smallest_perpendicular_pairs(caliper_centers):
    """
    Find perpendicular pairs of calipers.

    Rule: A lesion can ONLY have 2 lines if they are perpendicular to each other.
    Otherwise, each pair of calipers is a separate lesion.

    Returns:
        List of lesion groups, each containing:
        - 'indices': caliper indices belonging to this lesion (2 or 4 calipers)
        - 'connections': list of (i, j) pairs to draw (1 or 2 lines)
        - 'perpendicular': True only if this lesion has exactly 2 perpendicular lines
    """
    n = len(caliper_centers)
    if n < 2:
        return []

    points = np.array(caliper_centers)

    # Calculate all pairwise info
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            vec = points[j] - points[i]
            dist = np.linalg.norm(vec)
            all_pairs.append({
                'indices': (i, j),
                'distance': dist,
                'vector': vec
            })

    lesion_groups = []
    used_points = set()

    # STEP 1: Find all perpendicular 4-point combinations
    # (Skip if we have fewer than 4 points - can't form perpendicular pairs)
    perp_combos = []
    if n >= 4:
        for idx1, pair1 in enumerate(all_pairs):
            for idx2 in range(idx1 + 1, len(all_pairs)):
                pair2 = all_pairs[idx2]

                # Must use exactly 4 different points
                points_set = set(pair1['indices'] + pair2['indices'])
                if len(points_set) != 4:
                    continue

                # Check perpendicularity using dot product
                v1_norm = pair1['vector'] / (np.linalg.norm(pair1['vector']) + 1e-10)
                v2_norm = pair2['vector'] / (np.linalg.norm(pair2['vector']) + 1e-10)
                dot_product = abs(np.dot(v1_norm, v2_norm))

                # Within 45 degrees: cos(45°) ≈ 0.707
                if dot_product < 0.707:
                    # IMPORTANT: Check spatial proximity - all 4 points should be close together
                    # Calculate the bounding box of the 4 points
                    point_indices = list(points_set)
                    point_coords = points[point_indices]

                    # Get min/max x and y coordinates
                    min_x, min_y = point_coords.min(axis=0)
                    max_x, max_y = point_coords.max(axis=0)
                    bbox_width = max_x - min_x
                    bbox_height = max_y - min_y
                    bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)

                    # The bounding box diagonal should be reasonable compared to line lengths
                    # If the 4 points are too spread out, they're likely from different lesions
                    max_line_length = max(pair1['distance'], pair2['distance'])

                    # Allow bbox diagonal to be at most 1.8x the longest line
                    # This ensures the 4 points form a very compact cluster
                    if bbox_diagonal <= 1.8 * max_line_length:
                        # CRITICAL: Check if the lines actually intersect or nearly intersect
                        # Get the line segment endpoints
                        i1, j1 = pair1['indices']
                        i2, j2 = pair2['indices']
                        p1, p2 = points[i1], points[j1]  # Line 1
                        p3, p4 = points[i2], points[j2]  # Line 2

                        # Calculate intersection point of infinite lines using parametric form
                        # Line 1: p1 + t*(p2-p1), Line 2: p3 + s*(p4-p3)
                        d1 = p2 - p1  # Direction vector of line 1
                        d2 = p4 - p3  # Direction vector of line 2

                        # Solve: p1 + t*d1 = p3 + s*d2
                        # Cross product to find intersection
                        cross = d1[0] * d2[1] - d1[1] * d2[0]

                        if abs(cross) > 1e-6:  # Lines are not parallel
                            # Calculate parameter t for line 1
                            diff = p3 - p1
                            t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
                            s = (diff[0] * d1[1] - diff[1] * d1[0]) / cross

                            # Check if intersection is within both segments (strict)
                            # For perpendicular caliper measurements of the same lesion,
                            # the lines MUST actually cross each other
                            # t=0 is p1, t=1 is p2; s=0 is p3, s=1 is p4
                            # Allow only tiny tolerance for numerical errors
                            if -0.05 <= t <= 1.05 and -0.05 <= s <= 1.05:
                                # Lines intersect or nearly intersect - this is a valid perpendicular pair
                                perp_combos.append({
                                    'pair1': pair1,
                                    'pair2': pair2,
                                    'points': points_set,
                                    'perpendicularity': dot_product,  # Lower is better
                                    'avg_distance': (pair1['distance'] + pair2['distance']) / 2,
                                    'bbox_diagonal': bbox_diagonal,
                                    'intersection_t': t,
                                    'intersection_s': s
                                })

    # Sort by closest pairs (smallest average distance first)
    perp_combos.sort(key=lambda x: x['avg_distance'])

    # STEP 2: Assign perpendicular pairs as lesions (these get 2 lines each)
    for combo in perp_combos:
        # Skip if any of these points are already assigned
        if combo['points'] & used_points:
            continue

        lesion_groups.append({
            'indices': list(combo['points']),
            'connections': [combo['pair1']['indices'], combo['pair2']['indices']],
            'perpendicular': True
        })
        used_points.update(combo['points'])

    # STEP 3: All remaining pairs become separate single-line lesions
    # Sort by smallest distance first
    remaining_pairs = [p for p in all_pairs if not (set(p['indices']) & used_points)]
    remaining_pairs.sort(key=lambda x: x['distance'])

    for pair in remaining_pairs:
        # Skip if either point is already used
        if pair['indices'][0] in used_points or pair['indices'][1] in used_points:
            continue

        # Each remaining pair is a separate lesion with 1 line
        lesion_groups.append({
            'indices': list(pair['indices']),
            'connections': [pair['indices']],
            'perpendicular': False
        })
        used_points.update(pair['indices'])

    return lesion_groups


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

def save_debug_image(caliper_img, mask_img, caliper_centers, output_path, cluster_info=None):
    """
    Save debug image with caliper centers and cluster connections visualized.

    Args:
        caliper_img: Original image with calipers
        mask_img: Difference mask
        caliper_centers: List of all caliper center coordinates
        output_path: Where to save the debug image
        cluster_info: List of dicts with cluster analysis results, each containing:
            - 'indices': List of caliper indices in this cluster
            - 'measurements': List of measurements in pixels
            - 'measurements_cm': List of measurements in cm
            - 'connections': List of (i1, i2) tuples for drawing lines
            - 'perpendicular': Boolean indicating perpendicular measurements
    """
    # Convert PIL to numpy array for drawing
    img_array = np.array(caliper_img.convert('RGB'))

    # Define colors for different clusters (BGR format for OpenCV)
    cluster_colors = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (128, 0, 128),    # Purple
        (0, 128, 255),    # Light Orange
    ]

    if cluster_info:
        # Draw clusters with connections
        for cluster_idx, cluster_data in enumerate(cluster_info):
            color = cluster_colors[cluster_idx % len(cluster_colors)]
            indices = cluster_data['indices']
            connections = cluster_data['connections']
            measurements_cm = cluster_data.get('measurements_cm', [])
            perpendicular = cluster_data.get('perpendicular', False)

            # Draw connection lines first (so they appear behind circles)
            for conn_idx, (i1, i2) in enumerate(connections):
                # Connections now use global indices directly
                p1 = caliper_centers[i1]
                p2 = caliper_centers[i2]

                # Draw line
                cv2.line(img_array, p1, p2, color, 2)

                # Draw measurement text at midpoint
                if conn_idx < len(measurements_cm):
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    text = f"{measurements_cm[conn_idx]:.2f} cm"
                    cv2.putText(img_array, text, (mid_x + 5, mid_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Draw circles at caliper centers
            for global_idx in indices:
                cx, cy = caliper_centers[global_idx]
                cv2.circle(img_array, (cx, cy), radius=6, color=color, thickness=-1)
                cv2.circle(img_array, (cx, cy), radius=8, color=(255, 255, 255), thickness=1)  # White border

            # Draw cluster label
            if indices:
                # Label near the first caliper of the cluster
                first_idx = indices[0]
                cx, cy = caliper_centers[first_idx]
                label = f"L{cluster_idx + 1}"
                if perpendicular:
                    label += " (⊥)"
                cv2.putText(img_array, label, (cx - 30, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    else:
        # Legacy mode: just draw circles without clustering
        for cx, cy in caliper_centers:
            cv2.circle(img_array, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_array, f"({cx},{cy})", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw summary info at top
    if cluster_info:
        y_offset = 25
        cv2.putText(img_array, f"Found {len(cluster_info)} lesion(s)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        for cluster_idx, cluster_data in enumerate(cluster_info):
            y_offset += 25
            measurements_cm = cluster_data.get('measurements_cm', [])
            n_calipers = len(cluster_data['indices'])

            if measurements_cm:
                max_measurement = max(measurements_cm)
                text = f"L{cluster_idx + 1}: {max_measurement:.2f} cm ({n_calipers} cal)"
            else:
                text = f"L{cluster_idx + 1}: {n_calipers} cal"

            color = cluster_colors[cluster_idx % len(cluster_colors)]
            cv2.putText(img_array, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Convert mask to RGB for side-by-side comparison
    mask_array = np.array(mask_img.convert('RGB'))

    # Concatenate horizontally
    combined = np.hstack([img_array, mask_array])

    # Convert back to PIL and save
    debug_img = Image.fromarray(combined)
    debug_img.save(output_path)


def process_single_image_pair(pair, image_dir, image_data_row, save_debug=False, debug_dir=None):
    """
    Process a single image pair to detect calipers, cluster them by lesion, and calculate measurements.

    Returns:
        Dictionary with:
        - 'caliper_coordinates': String format "x,y;x,y;..." for all calipers
        - 'lesion_measurements': JSON string with per-lesion measurements
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

    # NEW: Find perpendicular pairs and group by lesion
    cluster_info_list = []
    lesion_measurements_json = None

    if len(caliper_centers) >= 2:
        # Get physical_delta_x for converting pixels to cm
        physical_delta_x = image_data_row.get('physical_delta_x')
        if physical_delta_x is not None and not pd.isna(physical_delta_x):
            physical_delta_x = float(physical_delta_x)
        else:
            physical_delta_x = None

        # Find smallest perpendicular pairs
        lesion_groups = find_smallest_perpendicular_pairs(caliper_centers)

        # Process each lesion group
        lesion_data = []
        points = np.array(caliper_centers)

        for group in lesion_groups:
            indices = group['indices']
            connections = group['connections']
            perpendicular = group['perpendicular']

            # Calculate distances for each connection
            measurements_px = []
            for i, j in connections:
                dist = np.linalg.norm(points[i] - points[j])
                measurements_px.append(dist)

            # Convert to cm if possible
            measurements_cm = []
            if physical_delta_x:
                measurements_cm = [m * physical_delta_x for m in measurements_px]

            # Store cluster info for debug visualization (keep all measurements for lines)
            cluster_info = {
                'indices': indices,
                'measurements': measurements_px,
                'measurements_cm': measurements_cm,
                'connections': connections,
                'perpendicular': perpendicular
            }
            cluster_info_list.append(cluster_info)

            # Store lesion data for database - use only MAX measurement
            max_measurement_px = max(measurements_px) if measurements_px else 0
            lesion_entry = {
                'caliper_count': len(indices),
                'measurement_px': float(max_measurement_px),  # Single max value
                'perpendicular': perpendicular
            }
            if physical_delta_x and measurements_cm:
                max_measurement_cm = max(measurements_cm)
                lesion_entry['measurement_cm'] = float(max_measurement_cm)  # Single max value

            lesion_data.append(lesion_entry)

        # Convert to JSON string for database storage
        if lesion_data:
            lesion_measurements_json = json.dumps(lesion_data)

    # Save debug image if requested
    if save_debug and debug_dir and caliper_centers:
        os.makedirs(debug_dir, exist_ok=True)
        # Use just the base filename to avoid path issues
        base_filename = os.path.basename(caliper_path)
        debug_path = os.path.join(debug_dir, f"debug_{base_filename}")

        # Pass cluster info to debug visualization
        save_debug_image(caliper_img, mask_img, caliper_centers, debug_path,
                        cluster_info=cluster_info_list if cluster_info_list else None)

    return {
        'caliper_coordinates': caliper_coordinates_str,
        'lesion_measurements': lesion_measurements_json
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
    print("Starting caliper location using mask-based detection with multi-lesion clustering...")

    with DatabaseManager() as db:
        # Ensure columns exist in the database
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
        multi_lesion_count = 0
        all_lesion_records = []

        for i, result in enumerate(results):
            if result is not None:
                caliper_image_name = valid_pairs[i]['caliper_image']

                # Get accession_number and patient_id from image data
                image_row = image_data[image_data['image_name'] == caliper_image_name].iloc[0]
                accession_number = image_row['accession_number']
                patient_id = image_row['patient_id']

                # Update Images table with caliper coordinates
                cursor.execute("""
                    UPDATE Images
                    SET caliper_coordinates = ?
                    WHERE image_name = ?
                """, (result['caliper_coordinates'], caliper_image_name))

                # Delete existing lesion records for this image
                cursor.execute("""
                    DELETE FROM Lesions
                    WHERE image_name = ?
                """, (caliper_image_name,))

                # Insert lesion records into Lesions table
                if result.get('lesion_measurements'):
                    lesion_data = json.loads(result['lesion_measurements'])

                    if len(lesion_data) > 1:
                        multi_lesion_count += 1

                    for lesion in lesion_data:
                        lesion_record = {
                            'accession_number': accession_number,
                            'patient_id': patient_id,
                            'image_name': caliper_image_name,
                            'lesion_measurement_cm': lesion.get('measurement_cm')
                        }
                        all_lesion_records.append(lesion_record)

                processed_count += 1
            else:
                print(f"Warning: Failed to process pair {i}")

        # Batch insert all lesion records
        if all_lesion_records:
            db.insert_lesions_batch(all_lesion_records)

        db.conn.commit()
        print(f"Processed {processed_count} image pairs with caliper coordinate detection")
        print(f"Found {multi_lesion_count} images with multiple lesions")
        print(f"Inserted {len(all_lesion_records)} lesion records into Lesions table")