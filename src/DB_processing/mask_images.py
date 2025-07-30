import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from storage_adapter import read_csv, read_image, save_data
import ast

def parse_caliper_boxes(caliper_boxes_str):
    """
    Parse caliper_boxes string and return list of bounding boxes.
    
    Args:
        caliper_boxes_str: String like "[x1, y1, x2, y2]" or "[x1, y1, x2, y2]; [x1, y1, x2, y2]"
    
    Returns:
        List of [x1, y1, x2, y2] coordinates
    """
    if pd.isna(caliper_boxes_str) or caliper_boxes_str == "":
        return []
    
    boxes = []
    # Split by semicolon for multiple boxes
    box_strings = caliper_boxes_str.split(';')
    
    for box_str in box_strings:
        box_str = box_str.strip()
        if box_str:
            try:
                # Parse the string like "[x1, y1, x2, y2]"
                box = ast.literal_eval(box_str)
                if len(box) == 4:
                    boxes.append([int(coord) for coord in box])
            except:
                print(f"Warning: Could not parse caliper box: {box_str}")
    
    return boxes

def dilate_mask(mask, expansion_factor=1.4, iterations=1):
    mask_area = np.sum(mask > 127)
    if mask_area > 0:
        estimated_radius = np.sqrt(mask_area / np.pi)
        # For 40% area expansion: (1 + K/R)² = 1.4
        # So K/R = sqrt(1.4) - 1 ≈ 0.183
        kernel_size = max(3, int(estimated_radius * (np.sqrt(expansion_factor) - 1)))
    else:
        kernel_size = 3
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated_mask

def apply_mask_with_white_background(image, mask):
    """
    Apply mask to grayscale image and set background to pure white.
    
    Args:
        image: PIL Image or numpy array (original image, will be converted to grayscale)
        mask: numpy array (binary mask where 255 = lesion, 0 = background)
    
    Returns:
        numpy array: Grayscale image with lesion visible and white background
    """
    # Convert PIL to numpy if needed and ensure grayscale
    if isinstance(image, Image.Image):
        # Convert to grayscale first
        if image.mode != 'L':
            image = image.convert('L')
        image_np = np.array(image)
    else:
        # If numpy array, convert RGB to grayscale if needed
        if len(image.shape) == 3:
            image_np = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_np = image.copy()
    
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Create binary mask (True where lesion exists)
    lesion_mask = mask > 127  # Use 127 as threshold to handle any potential noise
    
    # Create white background (255 for white in grayscale)
    white_background = np.ones_like(image_np) * 255
    
    # Apply mask: keep original pixels where lesion exists, white elsewhere
    result_image = np.where(lesion_mask, image_np, white_background)
    
    return result_image.astype(np.uint8)

def crop_to_caliper_box(image, caliper_boxes, expand_factor=1.2):
    """
    Crop image to the first caliper box coordinates, expanded by a factor.
    
    Args:
        image: numpy array (grayscale image)
        caliper_boxes: List of [x1, y1, x2, y2] coordinates
        expand_factor: Factor to expand the crop (1.2 = 20% larger)
    
    Returns:
        tuple: (cropped_image, expanded_boxes) where expanded_boxes contains the new coordinates
    """
    if not caliper_boxes:
        return image, []
    
    # Use the first caliper box
    x1, y1, x2, y2 = caliper_boxes[0]
    
    # Get image dimensions
    img_height, img_width = image.shape
    
    # Calculate current box dimensions and center
    current_width = x2 - x1
    current_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate expanded dimensions
    new_width = current_width * expand_factor
    new_height = current_height * expand_factor
    
    # Calculate new coordinates centered on the original box
    new_x1 = center_x - new_width / 2
    new_x2 = center_x + new_width / 2
    new_y1 = center_y - new_height / 2
    new_y2 = center_y + new_height / 2
    
    # Clamp coordinates to image bounds
    new_x1 = max(0, min(new_x1, img_width))
    new_y1 = max(0, min(new_y1, img_height))
    new_x2 = max(new_x1, min(new_x2, img_width))
    new_y2 = max(new_y1, min(new_y2, img_height))
    
    # Convert to integers
    new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
    
    # Create expanded box list
    expanded_box = [new_x1, new_y1, new_x2, new_y2]
    
    # Crop the image
    cropped_image = image[new_y1:new_y2, new_x1:new_x2]
    
    return cropped_image, [expanded_box]

def format_caliper_boxes(boxes):
    """
    Format caliper boxes back to string format.
    
    Args:
        boxes: List of [x1, y1, x2, y2] coordinates
    
    Returns:
        String in format "[x1, y1, x2, y2]" or "[x1, y1, x2, y2]; [x1, y1, x2, y2]"
    """
    if not boxes:
        return ""
    
    if len(boxes) == 1:
        bbox = boxes[0]
        return f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
    else:
        # Multiple boxes - join with semicolon
        bbox_strings = []
        for bbox in boxes:
            bbox_strings.append(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        return "; ".join(bbox_strings)

def Mask_Lesions(image_data, input_dir, output_dir):
    image_dir = f"{input_dir}/images/"
    mask_dir = f"{input_dir}/lesion_masks/"
    output_dir = f"{output_dir}/lesions/"

    # Filter for rows where has_caliper_mask = True
    masked_images = image_data[image_data['has_caliper_mask'] == True]
    
    if len(masked_images) == 0:
        print("No images found with has_caliper_mask = True")
        return image_data
    
    print(f"Found {len(masked_images)} images with masks")

    # Initialize the new column if it doesn't exist
    if 'caliper_boxes_expand' not in image_data.columns:
        image_data['caliper_boxes_expand'] = ""

    processed_count = 0
    failed_count = 0
    
    # Process each image with a mask
    for idx, row in tqdm(masked_images.iterrows(), total=len(masked_images), desc="Processing masked images"):
        try:
            image_name = row['ImageName']
            
            # Construct paths and fix double slashes
            image_path = os.path.join(image_dir, image_name).replace('//', '/')
            
            # Construct mask path (replace extension with .png)
            base_name = image_name.replace('.png', '')
            mask_filename = f"{base_name}.png"
            mask_path = os.path.join(mask_dir, mask_filename).replace('//', '/')
            
            # Load the original image (converted to grayscale)
            original_image = read_image(image_path, use_pil=True).convert('L')
            
            # Load the mask using read_image
            mask_pil = read_image(mask_path, use_pil=True)
            if mask_pil is None:
                print(f"Warning: Could not load mask: {mask_path}")
                failed_count += 1
                continue
            
            # Convert mask to grayscale numpy array
            if mask_pil.mode != 'L':
                mask_pil = mask_pil.convert('L')
            mask = np.array(mask_pil)
            
            # Resize mask to match image dimensions if needed
            img_width, img_height = original_image.size
            if mask.shape[:2] != (img_height, img_width):
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # Dilate the mask to enlarge it slightly
            mask = dilate_mask(mask, kernel_size=5, iterations=1)
            
            # Apply mask with white background
            result_image = apply_mask_with_white_background(original_image, mask)
            
            # Parse caliper boxes and crop the result with 20% expansion
            caliper_boxes_str = row.get('caliper_boxes', '')
            caliper_boxes = parse_caliper_boxes(caliper_boxes_str)
            
            expanded_boxes = []
            if caliper_boxes:
                result_image, expanded_boxes = crop_to_caliper_box(result_image, caliper_boxes, expand_factor=1.4)
            
            # Update the dataframe with expanded box coordinates
            expanded_boxes_str = format_caliper_boxes(expanded_boxes)
            image_data.loc[idx, 'caliper_boxes_expand'] = expanded_boxes_str
            
            # Save images
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename).replace('//', '/')
            
            # Convert numpy array back to PIL for saving
            result_pil = Image.fromarray(result_image, mode='L')  # 'L' for grayscale
            save_data(result_pil, output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {row['ImageName']}: {str(e)}")
            failed_count += 1
    
    print(f"Successfully processed: {processed_count} lesions | Failed: {failed_count}")
    
    return image_data