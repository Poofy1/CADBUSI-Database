import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from tools.storage_adapter import read_image, save_data
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.database import DatabaseManager

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

def dilate_mask(mask, expansion_factor=1.7, max_iterations=70):
    """
    Dilate mask iteratively until it reaches the target area expansion.
    Uses adaptive kernel sizes and iteration counts for faster convergence.
    
    Args:
        mask: numpy array (binary mask)
        expansion_factor: float (1.7 = 70% larger)
        max_iterations: int (safety limit to prevent infinite loops)
    
    Returns:
        numpy array: Dilated mask with target area
    """
    original_area = np.sum(mask > 127)
    if original_area == 0:
        return mask.copy()
    
    target_area = original_area * expansion_factor
    current_mask = mask.copy()
    
    for iteration in range(max_iterations):
        current_area = np.sum(current_mask > 127)
        
        # Check if we've reached the target
        if current_area >= target_area:
            return current_mask
        
        # Calculate how much more area we need
        area_ratio_needed = target_area / current_area
        
        # Choose kernel size and iterations based on how much expansion is still needed
        if area_ratio_needed > 2.0:
            kernel_size = 7
            iterations = 3  # Big steps when very far from target
        elif area_ratio_needed > 1.5:
            kernel_size = 5
            iterations = 2  # Medium steps
        elif area_ratio_needed > 1.2:
            kernel_size = 3
            iterations = 2  # Smaller steps but still 2 iterations
        else:
            kernel_size = 3
            iterations = 1  # Fine control when close to target
        
        # Create kernel and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        current_mask = cv2.dilate(current_mask, kernel, iterations=iterations)
    
    # If we haven't reached target after max iterations, return what we have
    #print(f"Warning: Reached max iterations ({max_iterations}) without achieving target area expansion")
    return current_mask

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
    
    return cropped_image, (new_x1, new_y1, new_x2, new_y2)

def save_debug_images(original_image, original_mask, dilated_mask, caliper_boxes, base_name, debug_dir, expand_factor=1.4):
    """
    Save debug visualization images.
    
    Args:
        original_image: PIL Image (original grayscale image)
        original_mask: numpy array (original mask)
        dilated_mask: numpy array (dilated/expanded mask)
        caliper_boxes: list of caliper box coordinates
        base_name: string (base filename without extension)
        debug_dir: string (debug output directory)
        expand_factor: float (expansion factor used for cropping)
    """
    # Convert PIL to numpy for processing
    img_np = np.array(original_image)
    img_height, img_width = img_np.shape
    
    # Create colored version for overlays (convert grayscale to RGB)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Ensure masks are proper binary (0 or 255) and same size as image
    original_mask_binary = (original_mask > 127).astype(np.uint8) * 255
    dilated_mask_binary = (dilated_mask > 127).astype(np.uint8) * 255
    
    # Make sure masks match image dimensions
    if original_mask_binary.shape != (img_height, img_width):
        original_mask_binary = cv2.resize(original_mask_binary, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    if dilated_mask_binary.shape != (img_height, img_width):
        dilated_mask_binary = cv2.resize(dilated_mask_binary, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    
    alpha = 0.4  # Slightly higher opacity for better visibility
    
    # 1. Original lesion mask overlay (red)
    mask_overlay = img_rgb.copy().astype(np.float32)
    red_mask = original_mask_binary.astype(np.float32) / 255.0  # Normalize to 0-1
    
    # Apply red overlay where mask exists
    mask_overlay[:, :, 0] = mask_overlay[:, :, 0] * (1 - alpha * red_mask) + 255 * alpha * red_mask
    mask_overlay = np.clip(mask_overlay, 0, 255).astype(np.uint8)
    
    # 2. Expanded lesion mask overlay (blue)
    expanded_mask_overlay = img_rgb.copy().astype(np.float32)
    blue_mask = dilated_mask_binary.astype(np.float32) / 255.0  # Normalize to 0-1
    
    # Apply blue overlay where expanded mask exists
    expanded_mask_overlay[:, :, 2] = expanded_mask_overlay[:, :, 2] * (1 - alpha * blue_mask) + 255 * alpha * blue_mask
    expanded_mask_overlay = np.clip(expanded_mask_overlay, 0, 255).astype(np.uint8)
    
    # 3. Combined comparison showing both masks
    combined_overlay = img_rgb.copy().astype(np.float32)
    
    # Apply red for original mask
    combined_overlay[:, :, 0] = combined_overlay[:, :, 0] * (1 - alpha * red_mask) + 255 * alpha * red_mask
    
    # Apply blue for expanded mask  
    combined_overlay[:, :, 2] = combined_overlay[:, :, 2] * (1 - alpha * blue_mask) + 255 * alpha * blue_mask
    
    # Areas with both masks will appear purple (red + blue)
    combined_overlay = np.clip(combined_overlay, 0, 255).astype(np.uint8)
    
    combined_overlay_path = os.path.join(debug_dir, f"{base_name}_mask_comparison.png")
    combined_overlay_pil = Image.fromarray(combined_overlay, mode='RGB')
    save_data(combined_overlay_pil, combined_overlay_path)

def process_single_mask(args):
    """
    Process a single image mask. This function will be called by each thread.
    
    Args:
        args: Tuple containing (idx, row, image_dir, mask_dir, output_dir, masks_output_dir, debug_enabled, debug_dir)
    
    Returns:
        Dict containing results for this image processing
    """
    idx, row, image_dir, mask_dir, output_dir, masks_output_dir, debug_enabled, debug_dir = args
    
    try:
        image_name = row['image_name']
        
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
                'crop_coords': "",
                'lesions_created': 0
            }
        
        # Convert mask to grayscale numpy array
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        original_mask = np.array(mask_pil)
        
        # Resize mask to match image dimensions if needed
        img_width, img_height = original_image.size
        if original_mask.shape[:2] != (img_height, img_width):
            original_mask = cv2.resize(original_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Parse caliper boxes
        caliper_boxes_str = row.get('caliper_boxes', '')
        caliper_boxes = parse_caliper_boxes(caliper_boxes_str)
        
        # Dilate the mask to enlarge it slightly
        dilated_mask = dilate_mask(original_mask)
        
        # Save debug images if enabled
        if debug_enabled:
            save_debug_images(original_image, original_mask, dilated_mask, caliper_boxes, base_name, debug_dir, expand_factor=1.4)
        
        # Apply mask with white background
        result_image = apply_mask_with_white_background(original_image, dilated_mask)
        
        # Track created lesion image names and dimensions
        created_lesion_files = []
        image_dimensions = []
        crop_coordinates = []  # Track crop coordinates (x1, y1, x2, y2)
        lesions_created = 0


        # Process each caliper box separately
        for box_idx, caliper_box in enumerate(caliper_boxes):
            # Crop the result with 40% expansion for this specific box
            cropped_image, expanded_box = crop_to_caliper_box(result_image, caliper_box, expand_factor=1.4)

            # Get dimensions of the cropped image
            crop_h, crop_w = cropped_image.shape
            image_dimensions.append((crop_w, crop_h))

            # Store crop coordinates (top-left corner of the crop in the original image)
            new_x1, new_y1, new_x2, new_y2 = expanded_box
            crop_coordinates.append((new_x1, new_y1))
            
            # Generate output filename with index if multiple boxes
            output_filename = f"{base_name}_{box_idx}.png"
            created_lesion_files.append(output_filename)
            output_path = os.path.join(output_dir, output_filename).replace('//', '/')

            # Convert numpy array back to PIL for saving
            result_pil = Image.fromarray(cropped_image, mode='L')  # 'L' for grayscale
            save_data(result_pil, output_path)
            
            lesions_created += 1
            
        # Process each caliper box separately to get raw masks
        for box_idx, caliper_box in enumerate(caliper_boxes):
            # Crop the result with 40% expansion for this specific box
            cropped_mask, expanded_box = crop_to_caliper_box(dilated_mask, caliper_box, expand_factor=1.4)
            
            # Uncrop: place the cropped mask back onto a black background
            # This zeros out everything outside the crop region
            uncropped_mask = np.zeros_like(dilated_mask)  # Create black background with original dimensions
            new_x1, new_y1, new_x2, new_y2 = expanded_box
            uncropped_mask[new_y1:new_y2, new_x1:new_x2] = cropped_mask
            
            # Apply the database crop region to further restrict the mask
            crop_x = int(row.get('crop_x', 0))
            crop_y = int(row.get('crop_y', 0))
            crop_w = int(row.get('crop_w', img_width))
            crop_h = int(row.get('crop_h', img_height))

            # Create final mask with only the database crop region
            final_mask = uncropped_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Generate output filename with index if multiple boxes
            output_filename = f"{base_name}_{box_idx}.png"
            masks_output_path = os.path.join(masks_output_dir, output_filename).replace('//', '/')

            # Convert numpy array back to PIL for saving
            result_pil = Image.fromarray(final_mask, mode='L')  # 'L' for grayscale
            save_data(result_pil, masks_output_path)
        

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

        # Extract crop coordinates strings
        if len(crop_coordinates) == 1:
            # Single lesion - store as "x,y"
            crop_coords_str = f"{crop_coordinates[0][0]},{crop_coordinates[0][1]}"
        else:
            # Multiple lesions - store as semicolon-separated "x,y" pairs
            coords = [f"{coord[0]},{coord[1]}" for coord in crop_coordinates]
            crop_coords_str = "; ".join(coords)


        return {
            'idx': idx,
            'success': True,
            'error': None,
            'lesion_images': ", ".join(created_lesion_files),
            'image_w': image_w_str,
            'image_h': image_h_str,
            'crop_coords': crop_coords_str,
            'lesions_created': lesions_created
        }
        
    except Exception as e:
        print(e)
        return {
            'idx': idx,
            'success': False,
            'error': f"Error processing {row['image_name']}: {str(e)}",
            'lesion_images': "",
            'image_w': "",
            'image_h': "",
            'crop_coords': "",
            'lesions_created': 0
        }

def Mask_Lesions(database_path, output_dir, filtered_image_df=None, max_workers=None, debug=False):
    """
    Multithreaded version of Mask_Lesions that creates lesion records.

    Args:
        database_path: Path to the database directory
        output_dir: Output directory for processed lesions
        filtered_image_df: DataFrame of pre-filtered images to process (optional)
        max_workers: Maximum number of worker threads (None = use all CPU cores)
        debug: Boolean flag to enable debug image saving

    Returns:
        pd.DataFrame: DataFrame with lesion information matching database schema
    """
    with DatabaseManager() as db:
        image_data = db.get_images_dataframe()
        
        # Filter for rows where has_caliper_mask = True
        if 'has_caliper_mask' in image_data.columns:
            masked_images = image_data[image_data['has_caliper_mask'] == True]
        else:
            masked_images = image_data
        
        # Apply the filter from upstream processing
        if filtered_image_df is not None:
            valid_image_names = set(filtered_image_df['image_name'].unique())
            masked_images = masked_images[masked_images['image_name'].isin(valid_image_names)]
            print(f"Filtered to {len(masked_images)} images based on upstream filters")

        if len(masked_images) == 0:
            print("No images found with masks")
            return pd.DataFrame(columns=['image_source', 'image_name', 'accession_number', 'patient_id', 'crop_w', 'crop_h'])

        print(f"Found {len(masked_images)} images to process")

        image_dir = f"{database_path}/images/"
        mask_dir = f"{database_path}/lesion_masks/"
        lesion_output_dir = f"{output_dir}/lesions/"
        masks_output_dir = f"{output_dir}/masks/"

        # Create debug directory if debug is enabled
        debug_dir = None
        if debug:
            debug_dir = f"{database_path}/debug/"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled. Debug images will be saved to: {debug_dir}")

        # Prepare arguments for each worker
        worker_args = []
        for idx, row in masked_images.iterrows():
            worker_args.append((idx, row, image_dir, mask_dir, lesion_output_dir, masks_output_dir, debug, debug_dir))

        # Initialize counters
        processed_count = 0
        failed_count = 0
        total_lesions_created = 0
        failed_images = []

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
                        failed_images.append(result['error'])
                        failed_count += 1

                    pbar.update(1)

        # Print errors at the end
        if failed_images:
            print("\nFailed masks and errors:")
            for error in failed_images:
                print(error)

        print(f"Successfully processed: {processed_count} images | Failed: {failed_count} | Total lesions created: {total_lesions_created}")

        if debug:
            print(f"Debug images saved to: {debug_dir}")

        # Convert processing_results to DataFrame
        lesion_records = []
        
        for idx, result in processing_results.items():
            if result['success'] and result['lesions_created'] > 0:
                # Get the original image row
                original_row = masked_images.loc[idx]
                source_image_name = original_row['image_name']
                patient_id = original_row['patient_id']
                accession_number = original_row['accession_number']
                laterality = original_row.get('laterality', None)
                yolo_confidence_str = original_row.get('yolo_confidence', '')
                samus_confidence_str = original_row.get('samus_confidence', '')

                # Get parent image crop offsets to adjust lesion coordinates
                parent_crop_x = int(original_row.get('crop_x', 0))
                parent_crop_y = int(original_row.get('crop_y', 0))

                # Parse lesion images (comma-separated)
                lesion_images_str = result['lesion_images']
                lesion_images = [img.strip() for img in lesion_images_str.split(',') if img.strip()]

                # Parse dimensions
                image_w_str = result['image_w']
                image_h_str = result['image_h']

                if '; ' in image_w_str:
                    # Multiple dimensions
                    widths = [int(w.strip()) for w in image_w_str.split(';')]
                    heights = [int(h.strip()) for h in image_h_str.split(';')]
                else:
                    # Single dimension
                    widths = [int(image_w_str)] if image_w_str else [0]
                    heights = [int(image_h_str)] if image_h_str else [0]

                # Parse crop coordinates
                crop_coords_str = result['crop_coords']
                crop_coords_list = []
                if crop_coords_str:
                    if '; ' in crop_coords_str:
                        # Multiple coordinates
                        for coord_pair in crop_coords_str.split(';'):
                            x, y = coord_pair.strip().split(',')
                            crop_coords_list.append((int(x), int(y)))
                    else:
                        # Single coordinate
                        x, y = crop_coords_str.split(',')
                        crop_coords_list.append((int(x), int(y)))

                # Parse YOLO confidence scores (semicolon-separated)
                if yolo_confidence_str and '; ' in yolo_confidence_str:
                    yolo_confidences = [conf.strip() for conf in yolo_confidence_str.split(';')]
                elif yolo_confidence_str:
                    yolo_confidences = [yolo_confidence_str.strip()]
                else:
                    yolo_confidences = []

                # Parse SAMUS confidence scores (semicolon-separated)
                if samus_confidence_str and '; ' in samus_confidence_str:
                    samus_confidences = [conf.strip() for conf in samus_confidence_str.split(';')]
                elif samus_confidence_str:
                    samus_confidences = [samus_confidence_str.strip()]
                else:
                    samus_confidences = []

                # Create a record for each lesion image
                for i, lesion_name in enumerate(lesion_images):
                    crop_w = widths[i] if i < len(widths) else 0
                    crop_h = heights[i] if i < len(heights) else 0
                    # Each lesion gets its corresponding confidence values
                    yolo_conf = yolo_confidences[i] if i < len(yolo_confidences) else ''
                    samus_conf = samus_confidences[i] if i < len(samus_confidences) else ''
                    # Get crop coordinates for this lesion (relative to original DICOM)
                    lesion_crop_x, lesion_crop_y = crop_coords_list[i] if i < len(crop_coords_list) else (0, 0)

                    # Adjust lesion coordinates to be relative to the cropped parent image
                    adjusted_crop_x = lesion_crop_x - parent_crop_x
                    adjusted_crop_y = lesion_crop_y - parent_crop_y

                    lesion_records.append({
                        'image_source': source_image_name,
                        'lesion_name': lesion_name,
                        'accession_number': accession_number,
                        'patient_id': patient_id,
                        'laterality': laterality,
                        'crop_w': crop_w,
                        'crop_h': crop_h,
                        'crop_x': adjusted_crop_x,
                        'crop_y': adjusted_crop_y,
                        'yolo_confidence': yolo_conf,
                        'samus_confidence': samus_conf
                    })
        
        lesion_df = pd.DataFrame(lesion_records)
        
        return lesion_df