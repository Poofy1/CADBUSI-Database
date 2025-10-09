import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.ML_processing.samus.model_dict import get_model
import cv2
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from storage_adapter import *
from src.DB_processing.tools import get_reader, reader, append_audit
from src.DB_processing.database import DatabaseManager
env = os.path.dirname(os.path.abspath(__file__))



class Config_BUSI:
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "val"
    visual = True
    modelname = "SAM"


def clamp_coordinates(x1, y1, x2, y2, max_width, max_height, min_x=0, min_y=0):
    """Clamp coordinates to valid image bounds."""
    return (
        max(min_x, min(x1, max_width)),
        max(min_y, min(y1, max_height)),
        max(min_x, min(x2, max_width)),
        max(min_y, min(y2, max_height))
    )
    
def get_caliper_inpainted_pairs(db):
    """
    Find all image pairs of original caliper images and their inpainted/clean versions.
    
    Two cases are handled:
    1. Inpainted pairs: Images with inpainted_from not null paired with their originals
    2. Closest clean pairs: Images with distance <= 5 and has_calipers = True paired with closest_fn
    
    Ignores pairs where either image has photometric_interpretation = 'RGB'
    
    Returns:
        List of all pairs (combined)
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
    inpainted_images = data[data['inpainted_from'].notna()]
    
    # Create a dictionary for original images (no inpainted_from value)
    original_images_dict = {}
    for index, row in data.iterrows():
        if pd.isna(row.get('inpainted_from')):
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
    
    # Try multiple binary thresholds to catch different intensities
    thresholds = [30, 50, 80, 127]  # Multiple sensitivity levels
    all_caliper_centers = []
    
    for threshold in thresholds:
        binary_mask = (mask_array > threshold).astype(np.uint8) * 255
        
        # Preprocess to enhance thin structures
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        enhanced_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Get image dimensions for tile calculation
        img_height, img_width = enhanced_mask.shape
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


def create_caliper_bounding_boxes(caliper_centers, max_calipers_per_box=4, padding=20, 
                                crop_x=None, crop_y=None, crop_x2=None, crop_y2=None):
    """
    Create bounding boxes for caliper centers, grouping them spatially.
    For groups with exactly 2 calipers, creates a rectangle with height = 3/4 * width.
    
    Args:
        caliper_centers: List of (x, y) tuples for caliper center coordinates
        max_calipers_per_box: Maximum number of calipers per bounding box
        padding: Padding around the bounding box
        crop_x, crop_y, crop_x2, crop_y2: Crop region bounds (optional)
    
    Returns:
        List of dictionaries containing box coordinates and associated caliper indices
    """
    if len(caliper_centers) <= 1:
        return []
    
    caliper_centers = np.array(caliper_centers)
    
    def clamp_bbox_to_crop(xmin, ymin, xmax, ymax):
        """Clamp bounding box coordinates to crop region if specified"""
        if all(coord is not None for coord in [crop_x, crop_y, crop_x2, crop_y2]):
            return clamp_coordinates(xmin, ymin, xmax, ymax, crop_x2, crop_y2, crop_x, crop_y)
        return xmin, ymin, xmax, ymax
    
    def create_two_caliper_bbox(coord1, coord2):
        """Create a rectangle bounding box for exactly 2 calipers (height = 3/4 * width)"""
        center_x = (coord1[0] + coord2[0]) / 2
        center_y = (coord1[1] + coord2[1]) / 2
        
        distance = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
        width = distance + 2 * padding
        height = width * 3 / 4
        
        half_width = width / 2
        half_height = height / 2
        
        xmin = int(center_x - half_width)
        ymin = int(center_y - half_height)
        xmax = int(center_x + half_width)
        ymax = int(center_y + half_height)
        
        return clamp_bbox_to_crop(xmin, ymin, xmax, ymax)
    
    def create_regular_bbox(coords):
        """Create a regular bounding box from coordinates"""
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        xmin = int(np.min(x_coords) - padding)
        ymin = int(np.min(y_coords) - padding)
        xmax = int(np.max(x_coords) + padding)
        ymax = int(np.max(y_coords) + padding)
        
        return clamp_bbox_to_crop(xmin, ymin, xmax, ymax)
    
    def create_bbox_for_group(group_indices):
        """Create bounding box for a group of caliper indices"""
        group_coords = caliper_centers[group_indices]
        
        if len(group_indices) == 2:
            bbox = create_two_caliper_bbox(group_coords[0], group_coords[1])
        else:
            bbox = create_regular_bbox(group_coords)
        
        return {
            'bbox': list(bbox),
            'caliper_indices': group_indices,
            'caliper_coords': group_coords.tolist()
        }
    
    # Handle case where all calipers fit in one box
    if len(caliper_centers) <= max_calipers_per_box:
        return [create_bbox_for_group(list(range(len(caliper_centers))))]
    
    # Group calipers spatially using simple distance-based clustering
    boxes = []
    remaining_indices = list(range(len(caliper_centers)))
    
    while remaining_indices:
        current_group = [remaining_indices.pop(0)]
        
        # Add nearby calipers to the current group
        while len(current_group) < max_calipers_per_box and remaining_indices:
            group_coords = caliper_centers[current_group]
            group_center = np.mean(group_coords, axis=0)
            
            # Find closest remaining caliper
            closest_distance = float('inf')
            closest_idx = None
            
            for idx in remaining_indices:
                distance = np.linalg.norm(caliper_centers[idx] - group_center)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_idx = idx
            
            if closest_idx is not None:
                current_group.append(closest_idx)
                remaining_indices.remove(closest_idx)
        
        boxes.append(create_bbox_for_group(current_group))
    
    return boxes


def prepare_box_prompts(boxes, image_size, model_input_size):
    """
    Prepare box prompts for the model by scaling coordinates to model input size.
    
    Args:
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        image_size: Tuple of (width, height) of the cropped image
        model_input_size: Target size for model input (e.g., 256)
        crop_coords: Optional crop coordinates (crop_x, crop_y, crop_w, crop_h)
    
    Returns:
        torch.Tensor: Box tensor scaled to model input size [N, 4]
    """
    if not boxes:
        return None
    
    cropped_width, cropped_height = image_size
    
    # Convert boxes to the right format and scale
    scaled_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds
        x1, y1, x2, y2 = clamp_coordinates(x1, y1, x2, y2, cropped_width, cropped_height)
                
        # Scale to model input size
        scaled_x1 = int(x1 * model_input_size / cropped_width)
        scaled_y1 = int(y1 * model_input_size / cropped_height)
        scaled_x2 = int(x2 * model_input_size / cropped_width)
        scaled_y2 = int(y2 * model_input_size / cropped_height)
        
        # Ensure coordinates are within model input bounds
        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = clamp_coordinates(scaled_x1, scaled_y1, scaled_x2, scaled_y2, 
                                                                        model_input_size - 1, model_input_size - 1
                                                                        )
        
        scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
    
    return torch.tensor(scaled_boxes, dtype=torch.float32)

def calculate_mask_coverage_in_boxes(mask_binary, bounding_boxes):
    """
    Calculate the percentage of mask pixels that fall within the given bounding boxes.
    
    Args:
        mask_binary: 2D numpy array with binary mask (0s and 1s)
        bounding_boxes: List of bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        float: Percentage of mask pixels that are inside any of the boxes (0-100)
    """
    if len(bounding_boxes) == 0:
        return 0.0
    
    # Count total mask pixels
    total_mask_pixels = np.sum(mask_binary > 0)
    
    if total_mask_pixels == 0:
        return 0.0
    
    # Create a combined mask for all bounding boxes
    mask_height, mask_width = mask_binary.shape
    combined_box_mask = np.zeros_like(mask_binary, dtype=bool)
    
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within bounds
        x1, y1, x2, y2 = clamp_coordinates(int(x1), int(y1), int(x2), int(y2), mask_width, mask_height)
                
        # Add this box to the combined mask
        if x2 > x1 and y2 > y1:
            combined_box_mask[y1:y2, x1:x2] = True
    
    # Count mask pixels that fall within any box
    mask_pixels_in_boxes = np.sum((mask_binary > 0) & combined_box_mask)
    
    # Calculate percentage
    coverage_percentage = (mask_pixels_in_boxes / total_mask_pixels) * 100
    
    return coverage_percentage

def clean_mask(mask_binary, min_hole_area=100, min_object_area=200, 
               kernel_size=3, iterations=1):
    """
    Clean binary mask by filling small holes and removing small islands.
    
    Args:
        mask_binary: Binary mask (0s and 1s)
        min_hole_area: Minimum hole area to keep (smaller holes will be filled)
        min_object_area: Minimum object area to keep (smaller objects will be removed)
        kernel_size: Size of morphological kernel
        iterations: Number of iterations for morphological operations
    
    Returns:
        Cleaned binary mask
    """
    
    
    # Convert to binary (0s and 1s) for scipy
    if mask_binary.dtype != np.uint8:
        mask_binary = mask_binary.astype(np.uint8)
    
    binary_mask = mask_binary > 0
    
    # Fill holes using scipy (much better than morphological closing)
    mask_filled = ndimage.binary_fill_holes(binary_mask)
    
    # Convert back to uint8 (0s and 255s)
    mask_filled = mask_filled.astype(np.uint8) * 255
    
    # Remove small islands using connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    
    # Create output mask
    cleaned_mask = np.zeros_like(mask_filled)
    
    # Keep only components larger than min_object_area (skip label 0 which is background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_object_area:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask


def process_single_image_pair(pair, image_dir, image_data_row, model, transform, device, encoder_input_size, save_debug_images=True):
    """
    Process a single image pair to detect calipers and compute bounding boxes.
    Updated to use bounding box prompts instead of point prompts.
    """
    caliper_path = pair['caliper_image']
    clean_path = pair['clean_image']
    
    # Extract caliper centers from mask
    caliper_img, clean_img, mask_img = create_difference_mask(caliper_path, clean_path, image_dir)
    
    # Find Calipers
    # Mask out everything outside the crop region
    crop_x, crop_y, crop_w, crop_h = None, None, None, None
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
    
    result = {
        'has_caliper_mask': False,
        'caliper_boxes': [],
    }
    
    # Only process if we have more than one caliper detected
    if len(caliper_centers) > 1:
        # Create bounding boxes for caliper groups
        caliper_boxes = create_caliper_bounding_boxes(
                            caliper_centers, 
                            max_calipers_per_box=4, 
                            padding=20,
                            crop_x=crop_x,
                            crop_y=crop_y, 
                            crop_x2=crop_x2 if crop_x is not None else None,
                            crop_y2=crop_y2 if crop_x is not None else None
                        )
        
        if caliper_boxes:
            # Create string format for bbox coordinates only
            if len(caliper_boxes) == 1:
                bbox = caliper_boxes[0]['bbox']
                result['caliper_boxes'] = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            else:
                # Multiple boxes - join with semicolon
                bbox_strings = []
                for box in caliper_boxes:
                    bbox = box['bbox']
                    bbox_strings.append(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                result['caliper_boxes'] = "; ".join(bbox_strings)
            
            # Crop the clean image and get dimensions
            clean_img_cropped = caliper_img.crop((crop_x, crop_y, crop_x2, crop_y2))
            cropped_width, cropped_height = clean_img_cropped.size
            
            # IMPORTANT: Adjust all caliper boxes to be relative to the cropped region
            adjusted_caliper_boxes = []
            for box_info in caliper_boxes:
                bbox = box_info['bbox']
                # Subtract crop offsets to make coordinates relative to cropped image
                adjusted_bbox = [
                    bbox[0] - crop_x,  # x1
                    bbox[1] - crop_y,  # y1  
                    bbox[2] - crop_x,  # x2
                    bbox[3] - crop_y   # y2
                ]
                
                # Ensure coordinates are within cropped image bounds
                adjusted_bbox = list(clamp_coordinates(adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[2], adjusted_bbox[3],
                                                        cropped_width, cropped_height
                                                    ))
                
                adjusted_caliper_boxes.append(adjusted_bbox)
            
            # Prepare box prompts for the model (Only call once)
            box_tensor = prepare_box_prompts(
                adjusted_caliper_boxes,
                (cropped_width, cropped_height),
                encoder_input_size
            )
            
            # Transform cropped image for model input
            image_tensor = transform(clean_img_cropped).unsqueeze(0).to(device)
            
            if box_tensor is not None:
                box_tensor = box_tensor.to(device)
                
                with torch.no_grad():

                    outputs = model(image_tensor, bbox=box_tensor.unsqueeze(0))

                    # Get the segmentation mask (same as before)
                    if isinstance(outputs, tuple):
                        mask = outputs[0]
                    elif isinstance(outputs, dict):
                        possible_keys = ['masks', 'pred_masks', 'output', 'prediction', 'segmentation', 'mask', 'pred']
                        mask = None
                        for key in possible_keys:
                            if key in outputs:
                                mask = outputs[key]
                                break
                        
                        if mask is None:
                            first_key = list(outputs.keys())[0]
                            mask = outputs[first_key]
                    else:
                        mask = outputs
                        
                    # Convert mask to numpy for visualization
                    mask_np = mask.squeeze().cpu().numpy()
                    mask_np = torch.sigmoid(torch.tensor(mask_np)).numpy()
                    
                    # Threshold the mask
                    mask_binary = (mask_np > 0.5).astype(np.uint8)
                    
                    # Resize mask to match cropped image dimensions
                    mask_resized = cv2.resize(mask_binary, (cropped_width, cropped_height), 
                                            interpolation=cv2.INTER_NEAREST)
                    
                    # Clean the mask: fill holes and remove small islands
                    mask_cleaned = clean_mask(
                        mask_resized, 
                        min_hole_area=50,      # Fill holes smaller than 50 pixels
                        min_object_area=100,   # Remove objects smaller than 100 pixels
                        kernel_size=3,         # 3x3 morphological kernel
                        iterations=1           # Number of iterations
                    )

                    # Use the cleaned mask for further processing
                    mask_resized = mask_cleaned

                    # Calculate mask coverage within bounding boxes
                    coverage_percentage = calculate_mask_coverage_in_boxes(mask_resized, adjusted_caliper_boxes)
                    
                    # Check if most of the mask is outside the boxes
                    if coverage_percentage < 50:
                        #print(f"Skipping image {clean_path}: Only {coverage_percentage:.1f}% of mask is inside boxes (threshold: {50}%)")
                        return result
                    
                    # Save the binary lesion mask (ALWAYS save if we get here)
                    parent_dir = os.path.dirname(os.path.normpath(image_dir))
                    lesion_mask_dir = os.path.join(parent_dir, "lesion_masks")
   
                    # Use the clean image name as the base for filename
                    base_name = clean_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    mask_filename = f"{base_name}.png"
                    mask_save_path = os.path.join(lesion_mask_dir, mask_filename)
                    
                    # Convert binary mask to 0-255 range and save
                    mask_to_save = mask_resized.astype(np.uint8)
                    save_data(mask_to_save, mask_save_path)
                    result['has_caliper_mask'] = True

                    # Save debug images if requested
                    if save_debug_images:
                        # Create debug image using CROPPED clean image as base
                        debug_img = clean_img_cropped.copy()
                        
                        # Convert PIL Image to numpy array and ensure RGB format
                        if isinstance(debug_img, Image.Image):
                            if debug_img.mode != 'RGB':
                                debug_img = debug_img.convert('RGB')
                            debug_img = np.array(debug_img)
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

                        # Draw ADJUSTED bounding boxes (relative to cropped image)
                        for i, adjusted_bbox in enumerate(adjusted_caliper_boxes):
                            x1, y1, x2, y2 = adjusted_bbox
                            
                            # Draw bounding box
                            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                        (0, 255, 0), 2)  # Green box
                            
                            # Draw caliper centers (also adjust coordinates)
                            original_box_info = caliper_boxes[i]
                            for coord in original_box_info['caliper_coords']:
                                cx, cy = coord
                                # Adjust caliper center coordinates to cropped image
                                adjusted_cx = cx - crop_x
                                adjusted_cy = cy - crop_y
                                cv2.circle(debug_img, (int(adjusted_cx), int(adjusted_cy)), 4, (255, 0, 0), -1)  # Red center dots

                        mask_overlay = np.zeros_like(debug_img)
                        mask_overlay[mask_resized > 0] = [0, 255, 255]  # Yellow in BGR
                        
                        # Blend the mask with the image (30% mask opacity)
                        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)\
                        
                        # Convert back to RGB for saving
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                        
                        # Save the debug image
                        parent_dir = os.path.dirname(os.path.normpath(image_dir))
                        save_dir = os.path.join(parent_dir, "test_images")
                        
                        # Use the clean image name as the base for filename
                        base_name = clean_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                        debug_filename = f"{base_name}.png"
                        debug_save_path = os.path.join(save_dir, debug_filename)
                        
                        # Save using your save_data function
                        save_data(debug_img, debug_save_path)
                        
                        print(f"Debug image saved: {debug_save_path}")
    
    return result


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image_pairs_multithreading(pairs, image_dir, image_data_df, model, transform, 
                                     device, encoder_input_size, save_debug_images=False, 
                                     num_threads=6):
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
        
        return process_single_image_pair(
            pair, image_dir, image_data_row, 
            model, transform, device, encoder_input_size, save_debug_images
        )
    
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(worker, pair): pair for pair in pairs}
        
        # Collect results with tqdm progress bar
        with tqdm(total=len(pairs), desc="Processing image pairs") as pbar:
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {pair['caliper_image']}: {e}")
                    results.append(None)
                
                pbar.update(1)
    
    return results
    
def Locate_Lesions(image_dir, save_debug_images=False):
    print("Starting lesion location using mask-based caliper detection...")
    
    with DatabaseManager() as db:
        # Load image data from database
        image_data = db.get_images_dataframe()
        
        # Add new columns if they don't exist
        for col in ["caliper_pairs", "caliper_boxes"]:
            if col not in image_data.columns:
                image_data[col] = ""
                
        # Initialize has_caliper_mask to False for ALL rows
        if "has_caliper_mask" not in image_data.columns:
            image_data["has_caliper_mask"] = False
        else:
            image_data["has_caliper_mask"] = False
        
        # Set up parameters
        encoder_input_size = 256
        low_image_size = 128
        modelname = 'SAMUS'
        checkpoint_path = os.path.join(env, 'models', 'SAMUS.pth')
        
        # Set up config
        opt = Config_BUSI
        opt.modelname = modelname
        device = torch.device(opt.device)
        
        # Load model
        class Args:
            def __init__(self):
                self.modelname = modelname
                self.encoder_input_size = encoder_input_size
                self.low_image_size = low_image_size
                self.vit_name = 'vit_b'
                self.sam_ckpt = checkpoint_path
                self.batch_size = 1
                self.n_gpu = 1
                self.base_lr = 0.0001
                self.warmup = False
                self.warmup_period = 250
                self.keep_log = False
        
        args = Args()
        
        model = get_model(modelname, args=args, opt=opt)
        model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # Transform for preprocessing images
        transform = transforms.Compose([
            transforms.Resize((encoder_input_size, encoder_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
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
        
        results = process_image_pairs_multithreading(
            pairs=valid_pairs,
            image_dir=image_dir,
            image_data_df=image_data,
            model=model,
            transform=transform,
            device=device,
            encoder_input_size=encoder_input_size,
            save_debug_images=save_debug_images,
        )
        
        # Update the database with results
        cursor = db.conn.cursor()
        processed_count = 0
        
        for i, result in enumerate(results):
            if result is not None:
                clean_idx = pair_to_clean_idx[i]
                clean_image_name = valid_pairs[i]['clean_image']
                
                cursor.execute("""
                    UPDATE Images
                    SET caliper_boxes = ?,
                        has_caliper_mask = ?
                    WHERE image_name = ?
                """, (result['caliper_boxes'], 
                      1 if result['has_caliper_mask'] else 0,
                      clean_image_name))
                
                processed_count += 1
            else:
                print(f"Warning: Failed to process pair {i}")
        
        db.conn.commit()
        print(f"Processed {processed_count} image pairs with mask-based detection")