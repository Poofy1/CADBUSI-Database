import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from PIL import Image
from src.ML_processing.samus.model_dict import load_samus_model
import cv2
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from storage_adapter import *
from torch.cuda.amp import autocast
from src.DB_processing.database import DatabaseManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
env = os.path.dirname(os.path.abspath(__file__))

def get_target_data(db_manager, limit=None):
    """Get target images from database."""
    # Query images, filtering out RGB images
    where_clause = "photometric_interpretation != 'RGB' AND label = 1"
    df = db_manager.get_images_dataframe(where_clause=where_clause)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"Reached debug limit of {limit} images, stopping...")
    
    # Get image names as list
    images = df['image_name'].tolist()
    print(f"Found {len(images)} images to process")
    
    return images

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
    if mask_binary.dtype == np.uint8:
        binary_mask = mask_binary > 0
    else:
        binary_mask = mask_binary.astype(bool)
    
    binary_mask = mask_binary > 0
    
    # Fill holes using scipy (much better than morphological closing)
    mask_filled = ndimage.binary_fill_holes(binary_mask).astype(np.uint8) * 255
    
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

def prepare_box_prompts(boxes, image_size, model_input_size):
    """
    Prepare box prompts for the model by scaling coordinates to model input size.
    
    Args:
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        image_size: Tuple of (width, height) of the cropped image
        model_input_size: Target size for model input (e.g., 256)
    
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
        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = clamp_coordinates(
            scaled_x1, scaled_y1, scaled_x2, scaled_y2, 
            model_input_size - 1, model_input_size - 1
        )
        
        scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
    
    return torch.tensor(scaled_boxes, dtype=torch.float32)

def detect_calipers_yolo(image, yolo_model, confidence_threshold):
    """
    Use YOLO model to detect calipers in the image
    
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
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                caliper_boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    return caliper_boxes

def load_image(image_name, image_data_row, image_dir):
    """Load and crop image based on database row information."""
    # Load the target image
    target_path = os.path.normpath(image_name)
    target_image_path = os.path.join(image_dir, target_path)
    target_image = read_image(target_image_path, use_pil=True)
        
    if target_image.mode != 'RGB':
        target_image = target_image.convert('RGB')
        
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
        
    return target_image, (crop_x, crop_y, crop_x2, crop_y2), (img_width, img_height)


def save_debug_image(image_name, image_dir, target_image, cropped_caliper_boxes, mask_resized=None):
    """
    Save a debug visualization image with bounding boxes and optional mask overlay.
    
    Args:
        image_name: Name of the image file
        image_dir: Directory containing the original images
        target_image: PIL Image or numpy array (cropped image)
        cropped_caliper_boxes: List of bounding boxes in cropped image coordinates
        mask_resized: Optional binary mask to overlay (same size as target_image)
    """
    try:
        # Convert PIL Image to numpy array and ensure RGB format
        if isinstance(debug_img, Image.Image):
            if debug_img.mode != 'RGB':
                debug_img = debug_img.convert('RGB')
            debug_img = np.array(debug_img)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes if they exist
        if cropped_caliper_boxes:
            for adjusted_bbox in cropped_caliper_boxes:
                x1, y1, x2, y2 = adjusted_bbox
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)  # Green box

        # Add mask overlay if mask exists
        if mask_resized is not None:
            mask_overlay = np.zeros_like(debug_img)
            mask_overlay[mask_resized > 0] = [0, 255, 255]  # Yellow in BGR
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        else: 
            print(f"No mask available for {image_name}")
            
        # Convert back to RGB for saving
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        
        # Save the debug image with organized folder structure
        parent_dir = os.path.dirname(os.path.normpath(image_dir))
        save_dir = os.path.join(parent_dir, "test_images")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Use the clean image name as the base for filename
        base_name = image_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        debug_filename = f"{base_name}.png"
        debug_save_path = os.path.join(save_dir, debug_filename)
        
        # Save using your save_data function
        save_data(debug_img, debug_save_path)
        
        print(f"Debug image saved: {debug_save_path}")
        
    except Exception as debug_e:
        print(f"Error saving debug image for {image_name}: {str(debug_e)}")


def process_single_image(image_name, image_dir, image_data_row, model, yolo_model, 
                              transform, device, encoder_input_size, save_debug_images=True, 
                              use_samus_model=True):
    """
    Process a single image pair to detect calipers using YOLO and optionally compute bounding boxes with SAMUS.
    
    Args:
        use_samus_model (bool): If True, runs SAMUS segmentation model. If False, only returns YOLO bounding boxes.
    """
    result = {
        'has_caliper_mask': False,
        'caliper_boxes': [],
    }
    
    # Initialize variables for debug saving
    target_image = None
    mask_resized = None
    cropped_caliper_boxes = []
    
    try:
        target_image, crops, dim = load_image(image_name, image_data_row, image_dir)
        crop_x, crop_y, crop_x2, crop_y2 = crops
        img_width, img_height = dim

        # Detect boxes using YOLO on the CROPPED image
        cropped_caliper_boxes = detect_calipers_yolo(target_image, yolo_model, confidence_threshold=0.3)

        if len(cropped_caliper_boxes) == 0:
            return result

        # Convert to full image coordinates ONLY for storage in result
        full_image_caliper_boxes = []
        for bbox in cropped_caliper_boxes:
            x1, y1, x2, y2 = bbox
            full_bbox = [x1 + crop_x, y1 + crop_y, x2 + crop_x, y2 + crop_y]
            full_image_caliper_boxes.append(full_bbox)

        # Store caliper boxes in result format (using full image coordinates)
        result['caliper_boxes'] = "; ".join(f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for b in full_image_caliper_boxes)
        
        # Only run SAMUS model if use_samus_model is True
        if use_samus_model and cropped_caliper_boxes:
            # Prepare box prompts for the model
            cropped_width, cropped_height = target_image.size
            box_tensor = prepare_box_prompts(
                cropped_caliper_boxes,
                (cropped_width, cropped_height),
                encoder_input_size
            )
            
            if box_tensor is not None:
                # Transform cropped image for model input
                image_tensor = transform(target_image).unsqueeze(0).to(device)
                box_tensor = box_tensor.to(device)
                
                with torch.no_grad():
                    with autocast():
                        outputs = model(image_tensor, bbox=box_tensor.unsqueeze(0))
                    mask = outputs['masks']
                        
                    # Convert mask to numpy for visualization
                    mask_np = mask.squeeze().cpu().numpy()
                    mask_np = torch.sigmoid(torch.tensor(mask_np)).numpy()
                    mask_binary = (mask_np > 0.5).astype(np.uint8)
                    
                    # Resize mask to match cropped image dimensions
                    mask_resized = cv2.resize(mask_binary, (cropped_width, cropped_height), 
                                            interpolation=cv2.INTER_NEAREST)

                    # Clean the mask: fill holes and remove small islands
                    mask_resized = clean_mask(mask_resized)

                    # Place the cropped mask in the correct position within the full mask
                    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    full_mask[crop_y:crop_y2, crop_x:crop_x2] = mask_resized
                    
                    # Save the binary lesion mask (now full-sized)
                    parent_dir = os.path.dirname(os.path.normpath(image_dir))
                    lesion_mask_dir = os.path.join(parent_dir, "lesion_masks")
                    mask_save_path = os.path.join(lesion_mask_dir, image_name)
                    
                    # Convert binary mask to 0-255 range and save (using full-sized mask)
                    mask_to_save = full_mask.astype(np.uint8)
                    save_data(mask_to_save, mask_save_path)
                    result['has_caliper_mask'] = True
        
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
    
    # Save debug images for ALL images when requested
    if save_debug_images and target_image is not None:
        save_debug_image(image_name, image_dir, target_image, cropped_caliper_boxes, mask_resized)
    
    return result


def process_image_multithreading(image_names, image_dir, image_data_df, model, yolo_model, 
                                      transform, device, encoder_input_size, save_debug_images=False, 
                                      num_threads=6):
    """
    Process multiple images using multithreading with tqdm progress bar
    """
    # Pre-compute lookup dictionary ONCE
    image_name_to_row = {}
    for idx, row in image_data_df.iterrows():
        # Use 'image_name' (the database column name)
        image_name_to_row[row['image_name']] = row
    
    def worker(image_with_index):
        image_index, image_name = image_with_index
        # O(1) dictionary lookup
        image_data_row = image_name_to_row[image_name]
        
        result = process_single_image(
            image_name, image_dir, image_data_row, 
            model, yolo_model, transform, device, encoder_input_size, save_debug_images
        )
        return image_index, result
    
    results = [None] * len(image_names)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks with their indices
        futures = {
            executor.submit(worker, (i, image_name)): i 
            for i, image_name in enumerate(image_names)
        }
        
        # Collect results with tqdm progress bar
        with tqdm(total=len(image_names), desc="Cropping lesions") as pbar:
            for future in as_completed(futures):
                try:
                    image_index, result = future.result()
                    results[image_index] = result
                except Exception as e:
                    image_index = futures[future]
                    print(f"Error processing {image_names[image_index]}: {e}")
                    results[image_index] = None
                
                pbar.update(1)
    
    return results



def Locate_Lesions(image_dir, save_debug_images=False):
    """
    Locate lesions using YOLO-based caliper detection and SAMUS segmentation.
    
    Args:
        database_path: Path to the database directory
        image_dir: Directory containing images
        save_debug_images: Whether to save debug visualization images
    
    Returns:
        None (updates database directly)
    """
    print("Starting lesion location using YOLO-based caliper detection...")
    
    # Load YOLO model
    try:
        model_path = os.path.join(env, 'models', 'yolo_lesion_detect.pt')
        yolo_model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None
    
    # Load SAMUS model
    encoder_input_size = 256
    model, transform, device = load_samus_model(
        env,
        device='cuda',
        encoder_input_size=encoder_input_size,
        low_image_size=128
    )
    
    # Connect to database
    with DatabaseManager() as db:
        # Load image data from database
        image_data = db.get_images_dataframe()
        
        # Add columns to dataframe if they don't exist (for in-memory processing)
        for col in ["caliper_boxes", "has_caliper_mask"]:
            if col not in image_data.columns:
                image_data[col] = "" if col == "caliper_boxes" else False
        
        # Initialize has_caliper_mask to False for ALL rows
        image_data["has_caliper_mask"] = False
        
        # Get image names to process
        image_names = get_target_data(db)

        if not image_names:
            print("No images found for mask-based detection.")
            return None

        # Create a mapping from image_name to row index for faster lookup
        image_name_to_idx = {row['image_name']: idx for idx, row in image_data.iterrows()}

        # Prepare valid image names with their corresponding rows
        valid_images = []
        image_to_clean_idx = {}

        for image_name in image_names:
            if image_name in image_name_to_idx:
                clean_idx = image_name_to_idx[image_name]
                valid_images.append(image_name)
                image_to_clean_idx[len(valid_images) - 1] = clean_idx
            else:
                print(f"Warning: Image '{image_name}' not found in database")

        if not valid_images:
            print("No valid images found.")
            return

        results = process_image_multithreading(
            image_names=valid_images,
            image_dir=image_dir,
            image_data_df=image_data,
            model=model,
            yolo_model=yolo_model,
            transform=transform,
            device=device,
            encoder_input_size=encoder_input_size,
            save_debug_images=save_debug_images,
        )

        # Update the database with results
        processed_count = 0
        batch_updates = []

        for i, result in enumerate(results):
            if result is not None:
                clean_idx = image_to_clean_idx[i]
                image_name = valid_images[i]
                
                # Prepare update data for batch insert
                update_data = {
                    'image_name': image_name,
                    'caliper_boxes': result['caliper_boxes'],
                    'has_caliper_mask': result['has_caliper_mask']
                }
                batch_updates.append(update_data)
                processed_count += 1
            else:
                print(f"Warning: Failed to process image {i}")

        if batch_updates:
            db.insert_images_batch(batch_updates, update_only=True)
            print(f"Processed {processed_count} images with YOLO-based caliper detection")