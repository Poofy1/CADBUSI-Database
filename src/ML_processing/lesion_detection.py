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
from src.ML_processing.download_models import *
from src.DB_processing.tools import get_reader, reader, append_audit
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add YOLO import
from ultralytics import YOLO

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


def get_target_data(csv_file_path, paired_only=False, limit=None):
    """
    Find all target images (has_calipers = False) and their corresponding caliper images if they exist.
    
    Logic:
    1. Get all images with has_calipers = False as target images
    2. For each target image, look for a corresponding caliper image that:
       - has_calipers = True
       - distance <= 5 and distance > 0  
       - closest_fn matches the target image name
    
    Ignores pairs where either image has PhotometricInterpretation = 'RGB'
    
    Args:
        csv_file_path: Path to the CSV file
        paired_only: If True, only return target images that have corresponding caliper images
        limit: If specified, only process this many target images (for debugging)
    
    Returns:
        List of dictionaries with keys:
        - 'clean_image': target image name (has_calipers = False)
        - 'caliper_image': corresponding caliper image name (or None if no match)
        - 'type': 'target_pair'
    """
    limit_msg = f" (processing max {limit} for debugging)" if limit else ""
    print(f"Finding target images and their corresponding caliper images{limit_msg}...")
    
    # Load the CSV file
    data = read_csv(csv_file_path)
    
    # Create a dictionary for quick lookup of all images by ImageName
    all_images_dict = {}
    for index, row in data.iterrows():
        all_images_dict[row['ImageName']] = {
            'index': index,
            'data': row
        }
    
    # Get all target images (has_calipers = False)
    target_images = data[data['has_calipers'] == False]
    
    # Get all potential caliper images that could match targets
    caliper_candidates = data[
        (data['distance'] <= 5) & (data['distance'] > 0) & 
        (data['has_calipers'] == True) & 
        (data['closest_fn'].notna())
    ]
    
    # Create a dictionary for quick lookup of caliper images by their closest_fn
    caliper_lookup = {}
    for index, caliper_row in caliper_candidates.iterrows():
        closest_fn = caliper_row['closest_fn']
        if closest_fn not in caliper_lookup:
            caliper_lookup[closest_fn] = []
        caliper_lookup[closest_fn].append({
            'index': index,
            'data': caliper_row
        })
    
    pairs = []
    matched_targets = 0
    processed_count = 0
    
    for index, target_row in target_images.iterrows():
        # Check limit for debugging
        if limit and processed_count >= limit:
            print(f"Reached debug limit of {limit} target images, stopping...")
            break
            
        # Skip if target image has PhotometricInterpretation = 'RGB'
        if target_row.get('PhotometricInterpretation') == 'RGB':
            continue
            
        target_image_name = target_row['ImageName']
        caliper_image = None
        
        # Look for a corresponding caliper image
        if target_image_name in caliper_lookup:
            # If multiple caliper images point to this target, take the first valid one
            for caliper_info in caliper_lookup[target_image_name]:
                caliper_data = caliper_info['data']
                
                # Skip if caliper image has PhotometricInterpretation = 'RGB'
                if caliper_data.get('PhotometricInterpretation') == 'RGB':
                    continue
                
                caliper_image = caliper_data['ImageName']
                matched_targets += 1
                break
        
        # If paired_only is True, skip entries without caliper pairs
        if paired_only and caliper_image is None:
            processed_count += 1  # Still count this as processed
            continue
        
        # Create the pair dictionary
        pair = {
            'type': 'target_pair',
            'clean_image': target_image_name,  # The target image (has_calipers = False)
            'caliper_image': caliper_image     # The corresponding caliper image (or None)
        }
        
        pairs.append(pair)
        processed_count += 1
    
    limit_suffix = f" (debug limit: {limit})" if limit else ""
    print(f"Found {len(pairs)} target images" + (" with caliper pairs" if paired_only else "") + limit_suffix)
    if not paired_only:
        print(f"Found {matched_targets} target images with corresponding caliper images")
        print(f"Found {len(pairs) - matched_targets} target images without caliper pairs")
    
    return pairs

def clamp_coordinates(x1, y1, x2, y2, max_width, max_height, min_x=0, min_y=0):
    """Clamp coordinates to valid image bounds."""
    return (
        max(min_x, min(x1, max_width)),
        max(min_y, min(y1, max_height)),
        max(min_x, min(x2, max_width)),
        max(min_y, min(y2, max_height))
    )
    
    

def clean_mask(mask_binary, min_object_area=200):
    """
    Clean binary mask by filling small holes and removing small islands.
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


def load_yolo_model():
    """Load the YOLO model for caliper detection"""
    # First try to download the model if it doesn't exist
    try:
        model_path = download_yolo_model()
    except Exception as e:
        print(f"Failed to download YOLO model: {e}")
        # Fallback to checking the env path
        model_path = os.path.join(env, 'models', 'yolo_lesion_detect.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
    
    yolo_model = YOLO(model_path)
    return yolo_model

def detect_calipers_yolo(image, yolo_model, confidence_threshold=0.5):
    """
    Use YOLO model to detect calipers in the image
    
    Args:
        image: PIL Image or numpy array
        yolo_model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of bounding boxes in format [x1, y1, x2, y2]
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Run YOLO inference
    results = yolo_model(image_np, conf=confidence_threshold, verbose=False)
    
    caliper_boxes = []
    
    # Extract bounding boxes from results
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box
                    caliper_boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    return caliper_boxes

def get_image_crop_coordinates(image_width, image_height, crop_margin=50):
    """
    Calculate crop coordinates for the image with some margin
    For now, return the full image coordinates, but you can modify this
    based on your specific cropping requirements
    """
    crop_x = max(0, crop_margin)
    crop_y = max(0, crop_margin)
    crop_x2 = min(image_width, image_width - crop_margin)
    crop_y2 = min(image_height, image_height - crop_margin)
    
    return crop_x, crop_y, crop_x2, crop_y2

def load_image(caliper_path, clean_path, image_dir):
    
    # Determine which image to load and process
    is_caliper = False
    if caliper_path is not None:
        # Case 1: Both caliper and clean images exist - load caliper for YOLO detection
        target_path = caliper_path
        is_caliper = True
    else:
        # Case 2: Only clean image exists - load clean image for YOLO detection
        target_path = clean_path
    
    # Load the target image
    target_path = os.path.normpath(target_path)
    target_image_path = os.path.join(image_dir, target_path)
    target_img = read_image(target_image_path, use_pil=True)
        
    if target_img.mode != 'RGB':
        target_img = target_img.convert('RGB')
        
    return target_img, is_caliper
            
def process_single_image_pair(pair, image_dir, image_data_row, model, yolo_model, transform, device, encoder_input_size, save_debug_images=True, use_samus_model=True):
    """
    Process a single image pair to detect calipers using YOLO and optionally compute bounding boxes with SAMUS.
    
    Args:
        use_samus_model (bool): If True, runs SAMUS segmentation model. If False, only returns YOLO bounding boxes.
    """
    caliper_path = pair['caliper_image']
    clean_path = pair['clean_image']
    
    result = {
        'has_caliper_mask': False,
        'caliper_boxes': [],
    }
    
    # Initialize variables for debug saving
    target_image = None
    adjusted_caliper_boxes = []
    mask_resized = None
    is_caliper = False
    
    try:
        target_image, is_caliper = load_image(caliper_path, clean_path, image_dir)

        # Extract crop parameters from the data row
        crop_x = image_data_row.get('crop_x', None)
        crop_y = image_data_row.get('crop_y', None)
        crop_w = image_data_row.get('crop_w', None)
        crop_h = image_data_row.get('crop_h', None)
        
        # Handle missing or null crop parameters
        if pd.isna(crop_x) or pd.isna(crop_y) or pd.isna(crop_w) or pd.isna(crop_h):
            # Use full image if crop parameters are missing
            crop_x, crop_y = 0, 0
            crop_w, crop_h = target_image.size[0], target_image.size[1]
        else:
            # Convert to integers
            crop_x, crop_y, crop_w, crop_h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)
        
        # Calculate crop coordinates
        crop_x2 = crop_x + crop_w
        crop_y2 = crop_y + crop_h
        
        # Ensure crop coordinates are within image bounds
        img_width, img_height = target_image.size
        crop_x = max(0, min(crop_x, img_width))
        crop_y = max(0, min(crop_y, img_height))
        crop_x2 = max(crop_x, min(crop_x2, img_width))
        crop_y2 = max(crop_y, min(crop_y2, img_height))
        
        # Crop and return the target image
        target_image = target_image.crop((crop_x, crop_y, crop_x2, crop_y2))

        # Detect calipers using YOLO on the CROPPED caliper image
        caliper_boxes = detect_calipers_yolo(target_image, yolo_model, confidence_threshold=0.3)
        
        if len(caliper_boxes) > 0:
            # IMPORTANT: Adjust caliper boxes to full image coordinates
            # (YOLO detected on cropped image, so we need to add crop offset)
            full_image_caliper_boxes = []
            for bbox in caliper_boxes:
                x1, y1, x2, y2 = bbox
                # Add crop offset to get coordinates relative to full image
                full_bbox = [
                    x1 + crop_x,
                    y1 + crop_y,
                    x2 + crop_x,
                    y2 + crop_y
                ]
                full_image_caliper_boxes.append(full_bbox)
            
            # Store caliper boxes in result format (using full image coordinates)
            if len(full_image_caliper_boxes) == 1:
                bbox = full_image_caliper_boxes[0]
                result['caliper_boxes'] = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            else:
                # Multiple boxes - join with semicolon
                bbox_strings = []
                for bbox in full_image_caliper_boxes:
                    bbox_strings.append(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                result['caliper_boxes'] = "; ".join(bbox_strings)
            
            # Adjust caliper boxes to be relative to the cropped region for SAMUS
            cropped_width, cropped_height = target_image.size
            for bbox in full_image_caliper_boxes:
                # Subtract crop offsets to make coordinates relative to cropped image
                adjusted_bbox = [
                    bbox[0] - crop_x,  # x1
                    bbox[1] - crop_y,  # y1  
                    bbox[2] - crop_x,  # x2
                    bbox[3] - crop_y   # y2
                ]
                
                # Ensure coordinates are within cropped image bounds
                adjusted_bbox = list(clamp_coordinates(
                    adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[2], adjusted_bbox[3],
                    cropped_width, cropped_height
                ))
                
                # Only add boxes that have valid dimensions
                if adjusted_bbox[2] > adjusted_bbox[0] and adjusted_bbox[3] > adjusted_bbox[1]:
                    adjusted_caliper_boxes.append(adjusted_bbox)
            
            # Only run SAMUS model if use_samus_model is True
            if use_samus_model and adjusted_caliper_boxes:
                # Prepare box prompts for the model
                box_tensor = prepare_box_prompts(
                    adjusted_caliper_boxes,
                    (cropped_width, cropped_height),
                    encoder_input_size
                )
                
                # Transform cropped image for model input
                image_tensor = transform(target_image).unsqueeze(0).to(device)
                
                if box_tensor is not None:
                    box_tensor = box_tensor.to(device)
                    
                    with torch.no_grad():
                        outputs = model(image_tensor, bbox=box_tensor.unsqueeze(0))

                        # Get the segmentation mask
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
                        mask_cleaned = clean_mask(mask_resized)
                        mask_resized = mask_cleaned

                        # Calculate mask coverage within bounding boxes (using cropped mask and adjusted boxes)
                        coverage_percentage = calculate_mask_coverage_in_boxes(mask_resized, adjusted_caliper_boxes)

                        # Check if most of the mask is outside the boxes
                        if coverage_percentage < 50:
                            print(f"Skipping image {clean_path}: Only {coverage_percentage:.1f}% of mask is inside boxes")
                        else:
                            # Coverage check passed - now create full-sized mask for saving
                            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            
                            # Place the cropped mask in the correct position within the full mask
                            full_mask[crop_y:crop_y2, crop_x:crop_x2] = mask_resized
                            
                            # Save the binary lesion mask (now full-sized)
                            parent_dir = os.path.dirname(os.path.normpath(image_dir))
                            lesion_mask_dir = os.path.join(parent_dir, "lesion_masks")

                            # Use the clean image name as the base for filename
                            base_name = clean_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                            mask_filename = f"{base_name}.png"
                            mask_save_path = os.path.join(lesion_mask_dir, mask_filename)
                            
                            # Convert binary mask to 0-255 range and save (using full-sized mask)
                            mask_to_save = full_mask.astype(np.uint8)
                            save_data(mask_to_save, mask_save_path)
                            result['has_caliper_mask'] = True
        
    except Exception as e:
        print(f"Error processing pair {clean_path} -> {clean_path}: {str(e)}")
    
    # Save debug images for ALL images when requested
    if save_debug_images and target_image is not None:
        try:
            # Create debug image using CROPPED image as base
            debug_img = target_image.copy()
            
            # Convert PIL Image to numpy array and ensure RGB format
            if isinstance(debug_img, Image.Image):
                if debug_img.mode != 'RGB':
                    debug_img = debug_img.convert('RGB')
                debug_img = np.array(debug_img)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes if they exist
            if adjusted_caliper_boxes:
                for adjusted_bbox in adjusted_caliper_boxes:
                    x1, y1, x2, y2 = adjusted_bbox
                    # Draw bounding box
                    cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)  # Green box

            # Add mask overlay if mask exists
            if mask_resized is not None:
                mask_overlay = np.zeros_like(debug_img)
                mask_overlay[mask_resized > 0] = [0, 255, 255]  # Yellow in BGR
                # Blend the mask with the image (30% mask opacity)
                debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
            
            
            if mask_resized is None and not is_caliper:
                print("missing mask on clean")
                
            # Convert back to RGB for saving
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            
            # Save the debug image with organized folder structure
            parent_dir = os.path.dirname(os.path.normpath(image_dir))
            save_dir = os.path.join(parent_dir, "test_images")
            
            # Determine subfolder based on whether calipers were detected
            if is_caliper:
                subfolder = "calipers"
            else:
                subfolder = "clean"
            
            # Create the full save directory path
            full_save_dir = os.path.join(save_dir, subfolder)
            os.makedirs(full_save_dir, exist_ok=True)
            
            # Use the clean image name as the base for filename
            base_name = clean_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            debug_filename = f"{base_name}.png"
            debug_save_path = os.path.join(full_save_dir, debug_filename)
            
            # Save using your save_data function
            save_data(debug_img, debug_save_path)
            
            print(f"Debug image saved: {debug_save_path})")
            
        except Exception as debug_e:
            print(f"Error saving debug image for {clean_path}: {str(debug_e)}")
    
    return result

def process_image_pairs_multithreading(pairs, image_dir, image_data_df, model, yolo_model, transform, 
                                     device, encoder_input_size, save_debug_images=False, 
                                     num_threads=6):
    """
    Process multiple image pairs using multithreading with tqdm progress bar
    """
    
    # Pre-compute lookup dictionary ONCE
    image_name_to_row = {}
    for idx, row in image_data_df.iterrows():
        image_name_to_row[row['ImageName']] = row
    
    def worker(pair_with_index):
        pair_index, pair = pair_with_index
        # O(1) dictionary lookup
        image_name = pair['clean_image']
        image_data_row = image_name_to_row[image_name]
        
        result = process_single_image_pair(
            pair, image_dir, image_data_row, 
            model, yolo_model, transform, device, encoder_input_size, save_debug_images
        )
        return pair_index, result  # Return both index and result
    
    results = [None] * len(pairs)  # Pre-allocate results array
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks with their indices
        futures = {
            executor.submit(worker, (i, pair)): i 
            for i, pair in enumerate(pairs)
        }
        
        # Collect results with tqdm progress bar
        with tqdm(total=len(pairs), desc="Processing image pairs") as pbar:
            for future in as_completed(futures):
                try:
                    pair_index, result = future.result()
                    results[pair_index] = result  # ✅ Store in correct position
                except Exception as e:
                    pair_index = futures[future]
                    print(f"Error processing {pairs[pair_index]['clean_image']}: {e}")
                    results[pair_index] = None
                
                pbar.update(1)
    
    return results

def Locate_Lesions(csv_file_path, image_dir, save_debug_images=False):
    # Download model
    download_samus_model()
    
    print("Starting lesion location using YOLO-based caliper detection...")
    
    # Load YOLO model
    try:
        yolo_model = load_yolo_model()
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None
    
    # Load image data
    image_data = read_csv(csv_file_path)
    
    # Add new columns if they don't exist
    for col in ["caliper_pairs", "caliper_boxes"]:
        image_data[col] = ""
            
    # Initialize has_caliper_mask to False for ALL rows
    if "has_caliper_mask" not in image_data.columns:
        image_data["has_caliper_mask"] = False
    else:
        # If column exists, ensure all values are False initially
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
    
    # Load SAMUS model
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
    pairs = get_target_data(csv_file_path)
    
    if not pairs:
        print("No image pairs found for mask-based detection.")
        return None
    
    # Create a mapping from ImageName to row index for faster lookup
    image_name_to_idx = {row['ImageName']: idx for idx, row in image_data.iterrows()}
    
    # Prepare valid pairs with their corresponding rows
    valid_pairs = []
    pair_to_clean_idx = {}  # Map to track which dataframe row each result corresponds to
    
    for pair in pairs:
        clean_image_name = pair['clean_image']
        
        if clean_image_name in image_name_to_idx:
            clean_idx = image_name_to_idx[clean_image_name]
            valid_pairs.append(pair)
            pair_to_clean_idx[len(valid_pairs) - 1] = clean_idx  # Map result index to dataframe index
        else:
            print(f"Warning: Clean image '{clean_image_name}' not found in CSV data")
    
    if not valid_pairs:
        print("No valid image pairs found.")
        return image_data
    
    print(f"Processing {len(valid_pairs)} image pairs...")
    
    results = process_image_pairs_multithreading(
        pairs=valid_pairs,
        image_dir=image_dir,
        image_data_df=image_data,
        model=model,
        yolo_model=yolo_model,
        transform=transform,
        device=device,
        encoder_input_size=encoder_input_size,
        save_debug_images=save_debug_images,
    )
    
    # Update the dataframe with results
    processed_count = 0
    for i, result in enumerate(results):
        if result is not None:
            clean_idx = pair_to_clean_idx[i]
            
            # Store both pieces of information
            image_data.at[clean_idx, "caliper_boxes"] = result['caliper_boxes']
            image_data.at[clean_idx, "has_caliper_mask"] = result['has_caliper_mask']
            
            processed_count += 1
        else:
            print(f"Warning: Failed to process pair {i}")
    
    # Save updated CSV
    save_data(image_data, csv_file_path)
    
    print(f"Processed {processed_count} image pairs with YOLO-based caliper detection")
    return image_data