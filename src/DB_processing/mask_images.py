import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from storage_adapter import read_csv, read_image, save_data
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def dilate_mask(mask, expansion_factor=1.6, iterations=1):
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

def crop_to_caliper_box(image, caliper_box, expand_factor=1.2):
    """
    Crop image to a single caliper box coordinates, expanded by a factor.
    
    Args:
        image: numpy array (grayscale image)
        expand_factor: Factor to expand the crop (1.2 = 20% larger)
    
    Returns:
        tuple: (cropped_image, expanded_box) where expanded_box contains the new coordinates
    """
    x1, y1, x2, y2 = caliper_box
    
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
    
    # Crop the image
    cropped_image = image[new_y1:new_y2, new_x1:new_x2]
    
    return cropped_image

def process_single_mask(args):
    """
    Process a single image mask. This function will be called by each thread.
    
    Args:
        args: Tuple containing (idx, row, image_dir, mask_dir, output_dir)
    
    Returns:
        Dict containing results for this image processing
    """
    idx, row, image_dir, mask_dir, output_dir = args
    
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
            return {
                'idx': idx,
                'success': False,
                'error': f"Could not load mask: {mask_path}",
                'lesion_images': "",
                'image_w': "",
                'image_h': "",
                'lesions_created': 0
            }
        
        # Convert mask to grayscale numpy array
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        mask = np.array(mask_pil)
        
        # Resize mask to match image dimensions if needed
        img_width, img_height = original_image.size
        if mask.shape[:2] != (img_height, img_width):
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Dilate the mask to enlarge it slightly
        mask = dilate_mask(mask)
        
        # Apply mask with white background
        result_image = apply_mask_with_white_background(original_image, mask)
        
        # Parse caliper boxes
        caliper_boxes_str = row.get('caliper_boxes', '')
        caliper_boxes = parse_caliper_boxes(caliper_boxes_str)
        
        # Track created lesion image names and dimensions
        created_lesion_files = []
        image_dimensions = []
        lesions_created = 0
        
        # If no caliper boxes, save the full masked image
        if not caliper_boxes:
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename).replace('//', '/')
            result_pil = Image.fromarray(result_image, mode='L')
            save_data(result_pil, output_path)
            lesions_created += 1
            created_lesion_files.append(output_filename)
            
            # Get dimensions of the full image
            img_h, img_w = result_image.shape
            image_dimensions.append((img_w, img_h))
            
        else:
            # Process each caliper box separately
            for box_idx, caliper_box in enumerate(caliper_boxes):
                # Crop the result with 40% expansion for this specific box
                cropped_image, expanded_box = crop_to_caliper_box(result_image, caliper_box, expand_factor=1.4)
                
                # Get dimensions of the cropped image
                crop_h, crop_w = cropped_image.shape
                image_dimensions.append((crop_w, crop_h))
                
                # Generate output filename with index if multiple boxes
                output_filename = f"{base_name}_{box_idx}.png"
                created_lesion_files.append(output_filename)
                
                output_path = os.path.join(output_dir, output_filename).replace('//', '/')
                
                # Convert numpy array back to PIL for saving
                result_pil = Image.fromarray(cropped_image, mode='L')  # 'L' for grayscale
                save_data(result_pil, output_path)
                lesions_created += 1
        
        
        # Extract width and height strings
        if len(image_dimensions) == 1:
            # Single image - store width and height as separate values
            image_w_str = str(image_dimensions[0][0])
            image_h_str = str(image_dimensions[0][1])
        else:
            # Multiple images - store as semicolon-separated lists
            widths = [str(dim[0]) for dim in image_dimensions]
            heights = [str(dim[1]) for dim in image_dimensions]
            image_w_str = "; ".join(widths)
            image_h_str = "; ".join(heights)
        
        
        return {
            'idx': idx,
            'success': True,
            'error': None,
            'lesion_images': ", ".join(created_lesion_files),
            'image_w': image_w_str,
            'image_h': image_h_str,
            'lesions_created': lesions_created
        }
        
    except Exception as e:
        return {
            'idx': idx,
            'success': False,
            'error': f"Error processing {row['ImageName']}: {str(e)}",
            'lesion_images': "",
            'image_w': "",
            'image_h': "",
            'lesions_created': 0
        }

def Mask_Lesions(image_data, input_dir, output_dir, max_workers=None):
    """
    Multithreaded version of Mask_Lesions that creates separate rows for each lesion image.
    
    Args:
        image_data: DataFrame containing image data
        input_dir: Input directory path
        output_dir: Output directory path  
        max_workers: Maximum number of worker threads (None = use all CPU cores)
    
    Returns:
        DataFrame with separate rows for each lesion image
    """
    image_dir = f"{input_dir}/images/"
    mask_dir = f"{input_dir}/lesion_masks/"
    lesion_output_dir = f"{output_dir}/images/"

    # Create output directory
    os.makedirs(lesion_output_dir, exist_ok=True)

    # Filter for rows where has_caliper_mask = True
    masked_images = image_data[image_data['has_caliper_mask'] == True]
    non_masked_images = image_data[image_data['has_caliper_mask'] != True]
    
    if len(masked_images) == 0:
        print("No images found with has_caliper_mask = True")
        return image_data
    
    print(f"Found {len(masked_images)} images with masks")

    # Prepare arguments for each worker
    worker_args = []
    for idx, row in masked_images.iterrows():
        worker_args.append((idx, row, image_dir, mask_dir, lesion_output_dir))

    # Initialize counters
    processed_count = 0
    failed_count = 0
    total_lesions_created = 0
    
    # Store results for creating new rows
    processing_results = {}
    
    # Use ThreadPoolExecutor for concurrent processing
    if max_workers is None:
        max_workers = min(os.cpu_count(), len(worker_args))
    
    print(f"Using {max_workers} worker threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_mask, args): args[0] for args in worker_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(futures), desc="Processing masked images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result['idx']
                processing_results[idx] = result
                
                if result['success']:
                    processed_count += 1
                    total_lesions_created += result['lesions_created']
                else:
                    print(result['error'])
                    failed_count += 1
                
                pbar.update(1)
    
    # Create new rows for each lesion image
    new_lesion_rows = []
    
    for idx, row in masked_images.iterrows():
        if idx in processing_results and processing_results[idx]['success']:
            result = processing_results[idx]
            
            # Parse lesion images (comma-separated)
            lesion_images_str = result['lesion_images']
            if lesion_images_str:
                lesion_images = [img.strip() for img in lesion_images_str.split(',') if img.strip()]
                
                # Parse dimensions
                image_w_str = result['image_w']
                image_h_str = result['image_h']
                
                if '; ' in image_w_str:
                    # Multiple dimensions
                    widths = [w.strip() for w in image_w_str.split(';')]
                    heights = [h.strip() for h in image_h_str.split(';')]
                else:
                    # Single dimension or empty
                    widths = [image_w_str] * len(lesion_images)
                    heights = [image_h_str] * len(lesion_images)
                
                # Create a new row for each lesion image
                for i, lesion_img in enumerate(lesion_images):
                    new_row = row.copy()
                    new_row['ImageName'] = lesion_img
                    
                    # Set individual dimensions
                    if i < len(widths):
                        new_row['image_w'] = widths[i]
                        new_row['image_h'] = heights[i]
                        
                        # Also update crop_w and crop_h to match the actual lesion dimensions
                        try:
                            new_row['crop_w'] = int(widths[i]) if widths[i] else 0
                            new_row['crop_h'] = int(heights[i]) if heights[i] else 0
                        except (ValueError, TypeError):
                            new_row['crop_w'] = 0
                            new_row['crop_h'] = 0
                    else:
                        new_row['image_w'] = ""
                        new_row['image_h'] = ""
                        new_row['crop_w'] = 0
                        new_row['crop_h'] = 0
                    
                    new_lesion_rows.append(new_row)
        # If processing failed, we skip this original row (it won't be in the final result)
    
    # Return only the new lesion rows, not the non-masked images
    if new_lesion_rows:
        lesion_df = pd.DataFrame(new_lesion_rows)
        # Ensure columns match
        for col in image_data.columns:
            if col not in lesion_df.columns:
                lesion_df[col] = ""
        
        # Reorder columns to match original
        lesion_df = lesion_df[image_data.columns]
        
        result_df = lesion_df  # Return only the lesion rows
    else:
        # No lesion images created, return empty DataFrame with same structure
        result_df = pd.DataFrame(columns=image_data.columns)

    print(f"Successfully processed: {processed_count} images | Failed: {failed_count} | Total lesions created: {total_lesions_created}")
    print(f"Original rows with masks: {len(masked_images)} | New lesion rows created: {len(new_lesion_rows)}")

    return result_df