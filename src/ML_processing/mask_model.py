"""
OCR-mask detection using YOLOv11.

The model was trained on bottom-half-cropped ultrasound images.
At inference we crop the bottom half, run YOLO, then map the
detected box coordinates back to full-image space.
"""

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from tools.storage_adapter import read_image, list_files, StorageClient

env = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(env, "models", "ocr_mask_yolo_2026_4_13.pt")
CONF = 0.25
BATCH_SIZE = 16
NUM_WORKERS = 16
PREFETCH_BATCHES = 2  # max in-flight loads = PREFETCH_BATCHES * BATCH_SIZE


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

    def _load_and_crop(filename):
        try:
            img = read_image(os.path.join(images_dir, filename), use_pil=True)
            if img is None:
                return filename, None, 0
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            mid_y = h // 2
            cropped_np = np.array(img.crop((0, mid_y, w, h)))
            return filename, cropped_np, mid_y
        except Exception as e:
            tqdm.write(f"  [err load] {filename}: {e}")
            return filename, None, 0

    def _best_bbox(result, mid_y):
        best_box = None
        best_score = 0.0
        for box in result.boxes:
            score = box.conf[0].cpu().item()
            if score > best_score:
                best_score = score
                best_box = box.xyxy[0].cpu().tolist()
        if best_box is None:
            return []
        x0, y0, x1, y1 = best_box
        return [int(x0), int(y0 + mid_y), int(x1), int(y1 + mid_y)]

    def _run_batch(items):
        """items: list of (fn, arr_or_None, mid_y). Appends results to description_masks."""
        valid_idx = [i for i, (_, arr, _) in enumerate(items) if arr is not None]
        bboxes = [[] for _ in items]
        if valid_idx:
            try:
                results = model.predict(
                    [items[i][1] for i in valid_idx],
                    conf=CONF,
                    verbose=False,
                )
                for i, r in zip(valid_idx, results):
                    bboxes[i] = _best_bbox(r, items[i][2])
            except Exception as e:
                tqdm.write(f"  [err batch] {e}")
        for (fn, _, _), bbox in zip(items, bboxes):
            description_masks.append((fn, bbox))

    description_masks = []
    chunks = [file_list[i:i + BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        pending = []  # FIFO of list-of-futures, one per prefetched chunk

        # Prime the pipeline
        for chunk in chunks[:PREFETCH_BATCHES]:
            pending.append([executor.submit(_load_and_crop, fn) for fn in chunk])
        next_submit = PREFETCH_BATCHES

        pbar = tqdm(total=len(file_list), desc='Finding OCR Masks')
        while pending:
            current_futs = pending.pop(0)

            # Keep the pool fed — submit another chunk before blocking on inference
            if next_submit < len(chunks):
                pending.append([executor.submit(_load_and_crop, fn) for fn in chunks[next_submit]])
                next_submit += 1

            items = [f.result() for f in current_futs]
            _run_batch(items)
            pbar.update(len(items))
            # Free references so numpy arrays are GC'd before next batch
            del items, current_futs
        pbar.close()

    return [], description_masks
