"""
OCR-mask detection using YOLOv11.

The model was trained on bottom-half-cropped ultrasound images.
At inference we crop the bottom half, run YOLO, then map the
detected box coordinates back to full-image space.
"""

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from tools.storage_adapter import read_image, list_files, StorageClient

env = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(env, "models", "ocr_mask_yolo_2026_4_13.pt")
CONF = 0.25


def get_first_image_in_each_folder(video_folder_path):
    first_images = []
    video_folder_path = os.path.normpath(video_folder_path)

    storage = StorageClient.get_instance()

    prefix = video_folder_path.replace('\\', '/').rstrip('/') + '/'

    iterator = storage._bucket.list_blobs(prefix=prefix, delimiter='/')
    blobs = list(iterator)
    prefixes = iterator.prefixes

    for folder_prefix in prefixes:
        folder_name = folder_prefix.rstrip('/').split('/')[-1]
        first_image_path = f"{folder_name}/{folder_name}_0.png"
        first_images.append(first_image_path)

    return first_images


def find_masks(images_dir, model_name, db_to_process, max_width, max_height,
               video_format=False):
    """
    Find OCR text-region bounding boxes in images using YOLO.

    Args:
        images_dir: Directory containing images
        model_name: Unused (kept for interface compatibility)
        db_to_process: DataFrame with image data (image_name or images_path column)
        max_width: Unused (kept for interface compatibility)
        max_height: Unused (kept for interface compatibility)
        video_format: Whether processing video frames

    Returns:
        Tuple of ([], description_masks) where description_masks is a list
        of (filename, bbox) tuples. bbox is [x0, y0, x1, y1] in full-image
        pixel coords, or [] if no detection.
    """
    model = YOLO(MODEL_PATH)

    # Resolve which files to process
    if video_format:
        all_first_images = get_first_image_in_each_folder(images_dir)
        images_to_process = set(db_to_process['images_path'].tolist())
        file_list = [img for img in all_first_images
                     if img.split('/')[0] in images_to_process]
    else:
        all_files = list_files(images_dir)
        file_dict = {os.path.basename(img): img for img in all_files}
        file_list = [img_name for img_name in db_to_process['image_name'].values
                     if img_name in file_dict]

    description_masks = []

    for filename in tqdm(file_list, desc='Finding OCR Masks'):
        try:
            img = read_image(os.path.join(images_dir, filename), use_pil=True)
            if img is None:
                description_masks.append((filename, []))
                continue

            if img.mode != 'RGB':
                img = img.convert('RGB')

            w, h = img.size
            mid_y = h // 2

            # Crop bottom half (model was trained on this)
            cropped = img.crop((0, mid_y, w, h))
            cropped_np = np.array(cropped)

            # Run YOLO
            results = model.predict(cropped_np, conf=CONF, verbose=False)

            # Get highest confidence detection
            best_box = None
            best_score = 0.0
            for r in results:
                for box in r.boxes:
                    score = box.conf[0].cpu().item()
                    if score > best_score:
                        best_score = score
                        best_box = box.xyxy[0].cpu().tolist()

            if best_box:
                x0, y0, x1, y1 = best_box
                # Map back to full-image coords
                bbox = [int(x0), int(y0 + mid_y), int(x1), int(y1 + mid_y)]
            else:
                bbox = []

            description_masks.append((filename, bbox))

        except Exception as e:
            tqdm.write(f"  [err] {filename}: {e}")
            description_masks.append((filename, []))

    return [], description_masks
