import cv2, sys
import numpy as np
import os
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.image_processing import get_reader
from src.DB_processing.database import DatabaseManager
from tools.storage_adapter import *
from ML_processing.simplify_region import simplify_us_region, polygon_to_storage

# ML Model configuration
IMG_SIZE = 256
MODEL_PATH = os.path.join(current_dir, 'us_region_2026_02_06.pth')
YOLO_MODEL_PATH = os.path.join('C:/Users/Tristan/Desktop', 'orientation_yolo_training', 'orientation_v15', 'weights', 'best.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
DEBUG_IMAGES = False
MAX_PER_MODEL = None #For debug

debug_crop_outputs = f"{current_dir}/debug_crop_outputs/"

EXCLUSION_BOXES = {
    'LOGIQE9':    {'right': 140, 'left_ratio': 0.04},
    'LOGIQE10':   {'right': 130, 'left_ratio': 0.04},
    'EPIQ 5G':    {'right': 135, 'left': 102, 'post_boxes': [(0, 0.6, 153, 1.0)]},
    'EPIQ 7G':    {'right': 135, 'left': 102},
    'EPIQ Elite': {'right': 135, 'left': 102},
}


def load_segmentation_model(checkpoint_path, device):
    """Load the MobileNetV2 U-Net segmentation model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name=ckpt.get("encoder_name", "mobilenet_v2"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class CropDataset(Dataset):
    """Dataset for batch processing images through the segmentation model."""

    def __init__(self, image_paths, image_to_model):
        self.image_paths = image_paths
        self.image_to_model = image_to_model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        model_name = self.image_to_model.get(image_name, None)

        # Load image using storage adapter (works with both local and GCP)
        image = read_image(image_path, use_pil=True)
        if image is None:
            return None

        # Convert PIL image to grayscale numpy array
        gray = np.array(image.convert('L'))

        # Ensure image is 2D (grayscale)
        if len(gray.shape) == 3:
            gray = gray[:, :, 0]  # Take first channel if multiple channels exist

        orig_h, orig_w = gray.shape

        # Convert grayscale to 3-channel for the model
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        # Normalize and convert to tensor (HWC -> CHW)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        return {
            'tensor': tensor,
            'image_path': image_path,
            'image_name': image_name,
            'model_name': model_name,
            'orig_size': (orig_h, orig_w),
            'gray': gray,
        }


def collate_fn(batch):
    """Custom collate that filters out None values."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'tensor': torch.stack([b['tensor'] for b in batch]),
        'image_path': [b['image_path'] for b in batch],
        'image_name': [b['image_name'] for b in batch],
        'model_name': [b['model_name'] for b in batch],
        'orig_size': [b['orig_size'] for b in batch],
        'gray': [b['gray'] for b in batch],
    }


def apply_exclusion_to_mask(mask, model_name):
    """Apply exclusion boxes to the mask by zeroing out those regions."""
    if model_name not in EXCLUSION_BOXES:
        return mask

    result = mask.copy()
    h, w = result.shape
    boxes = EXCLUSION_BOXES[model_name]

    if 'left' in boxes:
        result[:, :boxes['left']] = 0
    if 'left_ratio' in boxes:
        result[:, :int(w * boxes['left_ratio'])] = 0
    if 'right' in boxes:
        result[:, w - boxes['right']:] = 0

    return result


def fill_orientation_markers(image, yolo_model, model_name, conf_threshold=0.5):
    """
    Detect orientation markers using YOLO and fill them with gray value 128.
    Only runs on LOGIQ9/10 images.

    Args:
        image: Grayscale image (numpy array)
        yolo_model: Loaded YOLO model
        model_name: Scanner model name
        conf_threshold: Confidence threshold for detections

    Returns:
        Tuple of (modified image, list of detected boxes [(x1, y1, x2, y2), ...], confidence score)
    """
    # Only run YOLO on LOGIQ9/10 images
    if model_name not in ['LOGIQE9', 'LOGIQE10']:
        return image, [], None

    result = image.copy()
    detected_boxes = []
    max_confidence = None

    # Get image dimensions
    img_h = image.shape[0]

    # Only check bottom half of image
    bottom_half_start = int(img_h / 2)
    bottom_region = image[bottom_half_start:, :]

    # Convert grayscale to 3-channel for YOLO
    if len(bottom_region.shape) == 2:
        image_for_yolo = cv2.cvtColor(bottom_region, cv2.COLOR_GRAY2BGR)
    else:
        image_for_yolo = bottom_region

    # Run YOLO detection on bottom half only
    results = yolo_model(image_for_yolo, conf=conf_threshold, verbose=False)

    # Process detections - only keep the most confident one
    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
        confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores

        # Find the detection with highest confidence
        max_conf_idx = confidences.argmax()
        box = boxes[max_conf_idx]
        max_confidence = confidences[max_conf_idx]

        # Store box for debug visualization (not filling for now)
        # Adjust coordinates back to full image space
        x1, y1, x2, y2 = map(int, box)
        y1 += bottom_half_start
        y2 += bottom_half_start

        # result[y1:y2, x1:x2] = 128  # Commented out for debug - just outlining
        detected_boxes.append((x1, y1, x2, y2))

    return result, detected_boxes, max_confidence


def mask_to_bbox(mask):
    """Convert binary mask to bounding box (x, y, w, h)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))


def exclusion_boxes_to_polygons(img_h, img_w, model_name):
    """Convert exclusion boxes to list of polygon rectangles.

    Returns:
        List of polygons, each as Nx2 array of (x, y) vertices
    """
    if model_name not in EXCLUSION_BOXES:
        return []

    polygons = []
    boxes = EXCLUSION_BOXES[model_name]

    # Right exclusion box
    if 'right' in boxes:
        x1 = img_w - boxes['right']
        poly = np.array([[x1, 0], [img_w, 0], [img_w, img_h], [x1, img_h]], dtype=np.float32)
        polygons.append(poly)

    # Left exclusion box
    if 'left' in boxes:
        x2 = boxes['left']
        poly = np.array([[0, 0], [x2, 0], [x2, img_h], [0, img_h]], dtype=np.float32)
        polygons.append(poly)

    # Left ratio exclusion box
    if 'left_ratio' in boxes:
        x2 = int(img_w * boxes['left_ratio'])
        poly = np.array([[0, 0], [x2, 0], [x2, img_h], [0, img_h]], dtype=np.float32)
        polygons.append(poly)

    # Post boxes (EPIQ 5G hardcoded regions)
    if 'post_boxes' in boxes:
        for (x1, y1_ratio, x2, y2_ratio) in boxes['post_boxes']:
            y1 = int(img_h * y1_ratio)
            y2 = int(img_h * y2_ratio)
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            polygons.append(poly)

    return polygons


def collect_debris_polygons(img_h, img_w, model_name, yolo_boxes=None, ocr_detections=None, ocr_start_row=0, ocr_left=0, left_ocr_detections=None):
    """Collect all debris polygons from various sources.

    Returns:
        List of polygons (each as Nx2 array)
    """
    debris_polygons = []

    # Add exclusion box polygons
    debris_polygons.extend(exclusion_boxes_to_polygons(img_h, img_w, model_name))

    # Add YOLO detected orientation markers
    if yolo_boxes:
        for (x1, y1, x2, y2) in yolo_boxes:
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            debris_polygons.append(poly)

    # Add OCR text boxes from bottom region
    if ocr_detections:
        for detection in ocr_detections:
            bbox = detection[0]
            # Convert to absolute coordinates
            x1 = int(bbox[0][0]) + ocr_left
            y1 = int(bbox[0][1]) + ocr_start_row
            x2 = int(bbox[2][0]) + ocr_left
            y2 = int(bbox[2][1]) + ocr_start_row
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            debris_polygons.append(poly)

    # Add left side OCR text boxes (LOGIQE9/10)
    if left_ocr_detections:
        for detection in left_ocr_detections:
            bbox = detection[0]
            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])
            x2 = int(bbox[2][0])
            y2 = int(bbox[2][1])
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            debris_polygons.append(poly)

    return debris_polygons


def create_debug_image_ml(original_image, image_name, x, y, w, h, output_folder,
                          model_name=None, polygon=None, shape_type=None, iou=None, debris_polygons=None):
    """Create a debug image showing the simplified polygon, debris polygons, and final crop box."""

    # Convert to BGR if grayscale
    if len(original_image.shape) == 2:
        debug_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        debug_image = original_image.copy()

    img_h, img_w = debug_image.shape[:2]

    # Draw debris polygons in cyan (all debris regions)
    if debris_polygons is not None and len(debris_polygons) > 0:
        for debris_poly in debris_polygons:
            if len(debris_poly) > 0:
                poly_int = debris_poly.astype(np.int32)
                cv2.polylines(debug_image, [poly_int], isClosed=True, color=(255, 255, 0), thickness=2)  # Cyan - debris

    # Draw polygon in green (simplified US region)
    if polygon is not None and len(polygon) > 0:
        polygon_int = polygon.astype(np.int32)
        cv2.polylines(debug_image, [polygon_int], isClosed=True, color=(0, 255, 0), thickness=2)  # Green polygon

    # Draw the final bounding box
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red - final crop box

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3

    cv2.putText(debug_image, f"Image: {image_name}", (10, 40), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(debug_image, f"Model: {model_name or 'Unknown'}", (10, 90), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(debug_image, f"Crop: ({x}, {y}, {w}, {h})", (10, 140), font, font_scale, (0, 0, 255), font_thickness)

    # Show polygon shape type and IoU
    if shape_type is not None:
        cv2.putText(debug_image, f"Shape: {shape_type}", (10, 190), font, font_scale, (0, 255, 0), font_thickness)
    if iou is not None:
        cv2.putText(debug_image, f"IoU: {iou:.3f}", (10, 240), font, font_scale, (0, 255, 0), font_thickness)

    # Show debris polygon count
    if debris_polygons is not None:
        debris_count = len(debris_polygons)
        cv2.putText(debug_image, f"Debris: {debris_count} polygons", (10, 290), font, font_scale, (255, 255, 0), font_thickness)

    # Save the debug image
    name_without_ext, ext = os.path.splitext(image_name)
    output_path = os.path.join(output_folder, f"{name_without_ext}_debug{ext}")
    cv2.imwrite(output_path, debug_image)

    return output_path


def apply_post_boxes(image, model_name):
    """Apply post_boxes to black out regions after crop is decided."""
    if model_name and model_name in EXCLUSION_BOXES:
        boxes = EXCLUSION_BOXES[model_name]
        if 'post_boxes' in boxes:
            img_h = image.shape[0]
            for (box_x1, box_y1_ratio, box_x2, box_y2_ratio) in boxes['post_boxes']:
                box_y1 = int(img_h * box_y1_ratio)
                box_y2 = int(img_h * box_y2_ratio)
                image[box_y1:box_y2, box_x1:box_x2] = 0
    return image


def get_valid_image_names_from_db(allowed_models):
    """Get image names and their model names from database that match the allowed manufacturer model names.
    Returns a dict mapping image_name -> model_name"""
    with DatabaseManager() as db:
        placeholders = ','.join('?' * len(allowed_models))
        where_clause = f"manufacturer_model_name IN ({placeholders})"

        image_df = db.get_images_dataframe(where_clause=where_clause, params=tuple(allowed_models))
        image_to_model = dict(zip(image_df['image_name'], image_df['manufacturer_model_name']))

    return image_to_model


def process_batch_with_model(model, batch, output_folders, ocr_reader, yolo_model=None):
    """Process a batch of images through the model and generate debug images."""
    results = []

    with torch.no_grad():
        tensors = batch['tensor'].to(DEVICE)
        outputs = model(tensors)
        masks = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

    for i in range(len(batch['image_name'])):
        image_name = batch['image_name'][i]
        model_name = batch['model_name'][i]
        orig_h, orig_w = batch['orig_size'][i]
        gray = batch['gray'][i]

        # Resize mask back to original size
        mask_resized = cv2.resize(masks[i], (orig_w, orig_h))

        # Binarize the mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # Apply exclusion boxes to the mask
        binary_mask = apply_exclusion_to_mask(binary_mask, model_name)

        # Simplify mask to polygon
        shape_type, polygon, iou = simplify_us_region(binary_mask, scanner=model_name)

        # Get bounding box from mask
        bbox = mask_to_bbox(binary_mask)
        if bbox is None:
            continue

        x, y, w, h = bbox

        # OCR: scan bottom 1/4 of image with restricted x range
        ocr_start_row = int(3 * orig_h / 4)

        # Determine OCR x boundaries
        if model_name and model_name in EXCLUSION_BOXES:
            boxes = EXCLUSION_BOXES[model_name]
            # Use exclusion box boundaries
            ocr_left = boxes.get('left', 0)
            if 'left_ratio' in boxes:
                ocr_left = max(ocr_left, int(orig_w * boxes['left_ratio']))
            ocr_right = orig_w - boxes.get('right', 0)

        # Ensure valid region
        if ocr_left >= ocr_right:
            ocr_left = 0
            ocr_right = orig_w

        ocr_region = gray[ocr_start_row:orig_h, ocr_left:ocr_right]
        ocr_detections = ocr_reader.readtext(ocr_region)

        # Find the minimum y-value of detected text (if any)
        if ocr_detections:
            # For LOGIQE9/10: check if text is in rightmost 1/3, add 30px padding
            right_third_start = ocr_left + (ocr_right - ocr_left) * 2 // 3

            min_text_y = float('inf')
            for detection in ocr_detections:
                text_top_y = detection[0][0][1]
                text_x = detection[0][0][0] + ocr_left  # Convert to full image coords

                # Check if text is in rightmost 1/3 for LOGIQE9/10
                if model_name in ['LOGIQE9', 'LOGIQE10'] and text_x >= right_third_start:
                    # Add 30px vertical padding above this text
                    text_top_y -= 30

                min_text_y = min(min_text_y, text_top_y)

            if min_text_y != float('inf'):
                text_y_limit = int(min_text_y + ocr_start_row)
                # Adjust height to stop at text if it's within our crop region
                if text_y_limit > y and text_y_limit < y + h:
                    h = text_y_limit - y

        # Apply post_boxes to the original image
        gray_processed = apply_post_boxes(gray.copy(), model_name)

        # For LOGIQE9/10: OCR left side and fill detected text with gray
        left_ocr_detections = None
        if model_name in ['LOGIQE9', 'LOGIQE10']:
            # Scan left 1/5th of image (full height)
            left_width = int(orig_w / 5)
            left_region = gray_processed[:, :left_width]

            # Run OCR on left region
            left_ocr_detections = ocr_reader.readtext(left_region)

            # Fill detected text boxes with gray 128
            for detection in left_ocr_detections:
                # Get bounding box coordinates
                bbox = detection[0]
                x1 = int(bbox[0][0])
                y1 = int(bbox[0][1])
                x2 = int(bbox[2][0])
                y2 = int(bbox[2][1])

                # Fill text region with gray
                gray_processed[y1:y2, x1:x2] = 128

        # Fill orientation markers with gray 128 (after crop is determined)
        yolo_boxes = []
        if yolo_model is not None:
            gray_processed, yolo_boxes, _ = fill_orientation_markers(gray_processed, yolo_model, model_name)

        # Collect all debris polygons
        debris_polygons = collect_debris_polygons(
            orig_h, orig_w, model_name,
            yolo_boxes=yolo_boxes,
            ocr_detections=ocr_detections,
            ocr_start_row=ocr_start_row,
            ocr_left=ocr_left,
            left_ocr_detections=left_ocr_detections
        )

        # Convert polygons to storage format
        us_polygon_str = polygon_to_storage(polygon, precision=1)
        debris_polygons_str = [polygon_to_storage(p, precision=1) for p in debris_polygons]

        # Create debug image
        if DEBUG_IMAGES:
            output_folder = output_folders.get(model_name, debug_crop_outputs)
            create_debug_image_ml(gray_processed, image_name, x, y, w, h, output_folder, model_name, polygon, shape_type, iou, debris_polygons)

        results.append((image_name, (x, y, w, h), model_name, us_polygon_str, debris_polygons_str))

    return results


def generate_crop_regions(CONFIG):
    # Define allowed manufacturer model names
    ALLOWED_MODELS = ['LOGIQE9', 'LOGIQE10', 'EPIQ 5G', 'EPIQ 7G', 'EPIQ Elite']

    # Get image folder path from CONFIG
    image_folder_path = f"{CONFIG['DATABASE_DIR']}/images/"

    # Get valid image names and their models from the database
    image_to_model = get_valid_image_names_from_db(ALLOWED_MODELS)
    print(f"Found {len(image_to_model)} images with allowed model names: {ALLOWED_MODELS}")

    # Get all images from inputs folder that match allowed models
    all_image_files = [file for file in os.listdir(image_folder_path)
                       if file.lower().endswith('.png') and file in image_to_model]
    print(f"Filtered to {len(all_image_files)} images matching allowed models")

    # Group images by model (max per model from CONFIG, or None/0 for all)
    images_by_model = {}
    for img in all_image_files:
        model = image_to_model[img]
        if model not in images_by_model:
            images_by_model[model] = []
        # If MAX_PER_MODEL is None or 0, process all images
        if MAX_PER_MODEL is None or MAX_PER_MODEL == 0 or len(images_by_model[model]) < MAX_PER_MODEL:
            images_by_model[model].append(img)

    # Print counts per model
    print("\nImages per model:")
    for model, images in images_by_model.items():
        print(f"  - {model}: {len(images)} images")

    # Create output folders for each model
    output_folders = {}
    for model in images_by_model.keys():
        model_folder = f"{current_dir}/debug_crop_outputs_{model.replace(' ', '_')}/"
        os.makedirs(model_folder, exist_ok=True)
        output_folders[model] = model_folder

    # Load the segmentation model
    print(f"\nLoading segmentation model from {MODEL_PATH}...")
    print(f"Using device: {DEVICE}")
    seg_model = load_segmentation_model(MODEL_PATH, DEVICE)
    print("Model loaded successfully!")

    # Initialize OCR reader
    print("Initializing OCR reader...")
    ocr_reader = get_reader()
    print("OCR reader initialized!")

    # Load YOLO orientation detection model
    yolo_model = None
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"\nLoading YOLO orientation model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully!")
    else:
        print(f"\nYOLO model not found at {YOLO_MODEL_PATH}, skipping orientation marker detection")

    # Flatten all images into a single list for batch processing
    all_images = []
    for model, images in images_by_model.items():
        for img in images:
            all_images.append(os.path.join(image_folder_path, img))

    # Create dataset and dataloader
    dataset = CropDataset(all_images, image_to_model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4,
                           collate_fn=collate_fn, pin_memory=True)

    # Process all images
    print(f"\nProcessing {len(all_images)} images...")
    all_results = []

    for batch in tqdm(dataloader, desc="Processing"):
        if batch is None:
            continue
        results = process_batch_with_model(seg_model, batch, output_folders, ocr_reader, yolo_model)
        all_results.extend(results)

    # Write results to database
    print(f"\nWriting {len(all_results)} crop regions to database...")
    with DatabaseManager() as db:
        # Ensure new columns exist
        db.add_column_if_not_exists('Images', 'us_polygon', 'TEXT')
        db.add_column_if_not_exists('Images', 'debris_polygons', 'TEXT')

        # Prepare batch updates
        image_updates = []
        for image_name, bbox, model_name, us_polygon_str, debris_polygons_str in all_results:
            x, y, w, h = bbox
            # Join debris polygons with | separator
            debris_str = '|'.join(debris_polygons_str)

            image_updates.append({
                'image_name': image_name,
                'crop_x': x,
                'crop_y': y,
                'crop_w': w,
                'crop_h': h,
                'us_polygon': us_polygon_str,
                'debris_polygons': debris_str
            })

        # Batch update database
        updated_count = db.insert_images_batch(image_updates, upsert=True)
        print(f"Updated {updated_count} images in database with crop regions")

    print(f"\nDone! Processed {len(all_results)} images")



if __name__ == "__main__":
    # For standalone testing - use a minimal CONFIG
    from config import CONFIG
    generate_crop_regions(CONFIG)