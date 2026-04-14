"""
Re-run OCR mask detection + OCR on an existing database.

Backs up the DB first, then updates only:
  - description
  - laterality, area, orientation, clock_pos, nipple_dist
  (the columns derived from OCR text)

Uses threaded downloads + batched YOLO for speed on large datasets.

Usage:
    python tools/rerun_ocr.py <image_folder_path>

Example:
    python tools/rerun_ocr.py Databases/database_2026_1_13_main/images
"""

import sys
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from tqdm import tqdm

# Project root
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from config import CONFIG
from src.DB_processing.database import DatabaseManager
from src.DB_processing.image_processing import (
    get_OCR,
    extract_descript_features,
    description_labels_dict,
)
from src.ML_processing.mask_model import MODEL_PATH, CONF
from tools.storage_adapter import StorageClient, read_image, list_files

DB_PATH = "data/cadbusi.db"
BACKUP_PATH = "data/cadbusi_pre_yolo_ocr.db"

OCR_COLUMNS = ["description", "laterality", "area", "orientation", "clock_pos", "nipple_dist"]

BATCH_SIZE = 64
DOWNLOAD_WORKERS = 16


# ── Fast parallel mask detection ──────────────────────────────────────

def _download_and_crop(args):
    """Download one image from GCS, crop bottom half, return numpy array."""
    images_dir, filename = args
    try:
        img = read_image(os.path.join(images_dir, filename), use_pil=True)
        if img is None:
            return (filename, None, 0)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        mid_y = h // 2
        cropped_np = np.array(img.crop((0, mid_y, w, h)))
        return (filename, cropped_np, mid_y)
    except Exception:
        return (filename, None, 0)


def find_masks_fast(images_dir, image_df):
    """
    Threaded download + batched YOLO inference.
    Returns list of (filename, bbox) tuples.
    """
    model = YOLO(MODEL_PATH)

    all_files = list_files(images_dir)
    file_dict = {os.path.basename(img): img for img in all_files}
    file_list = [name for name in image_df['image_name'].values if name in file_dict]

    description_masks = []
    batch = []

    args = [(images_dir, f) for f in file_list]

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for filename, cropped_np, mid_y in tqdm(
            pool.map(_download_and_crop, args),
            total=len(args),
            desc='Finding OCR Masks',
        ):
            if cropped_np is None:
                description_masks.append((filename, []))
                continue

            batch.append((filename, cropped_np, mid_y))

            if len(batch) >= BATCH_SIZE:
                description_masks.extend(_run_batch(model, batch))
                batch = []

    if batch:
        description_masks.extend(_run_batch(model, batch))

    return description_masks


def _run_batch(model, batch):
    """Run YOLO on a batch, return list of (filename, bbox)."""
    filenames = [b[0] for b in batch]
    images = [b[1] for b in batch]
    mid_ys = [b[2] for b in batch]

    results_list = model.predict(images, conf=CONF, verbose=False, batch=BATCH_SIZE)

    out = []
    for i, results in enumerate(results_list):
        best_box, best_score = None, 0.0
        for box in results.boxes:
            score = box.conf[0].cpu().item()
            if score > best_score:
                best_score = score
                best_box = box.xyxy[0].cpu().tolist()

        if best_box:
            x0, y0, x1, y1 = best_box
            bbox = [int(x0), int(y0 + mid_ys[i]), int(x1), int(y1 + mid_ys[i])]
        else:
            bbox = []
        out.append((filenames[i], bbox))

    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/rerun_ocr.py <image_folder_path>")
        print("  e.g. python tools/rerun_ocr.py Databases/database_2026_1_13_main/images")
        sys.exit(1)

    image_folder_path = sys.argv[1]

    # Initialise storage (needed for read_image / list_files)
    StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", "") or CONFIG.get("storage", {}).get("bucket_name", ""),
    )

    # ── Backup ──
    if not os.path.exists(BACKUP_PATH):
        print(f"Backing up DB → {BACKUP_PATH}")
        shutil.copy2(DB_PATH, BACKUP_PATH)
    else:
        print(f"Backup already exists at {BACKUP_PATH}, skipping")

    # ── Load images from DB ──
    with DatabaseManager() as db:
        image_df = db.get_images_dataframe()
        breast_df = db.get_study_cases_dataframe()

    print(f"Loaded {len(image_df)} images from database")

    # ── Run YOLO OCR mask detection (fast: threaded + batched) ──
    print("\nRunning YOLO OCR mask detection ...")
    description_masks = find_masks_fast(image_folder_path, image_df)
    masks_found = sum(1 for _, bbox in description_masks if bbox)
    print(f"Masks found: {masks_found} / {len(description_masks)}")

    # ── Run OCR ──
    print("\nPerforming OCR ...")
    descriptions = get_OCR(image_folder_path, description_masks)
    valid = sum(1 for d in descriptions.values() if d)
    print(f"Valid descriptions: {valid} / {len(descriptions)}")

    # ── Extract features from descriptions ──
    import pandas as pd
    image_df['description'] = image_df['image_name'].map(descriptions)

    temp_df = image_df['description'].apply(
        lambda x: extract_descript_features(x, labels_dict=description_labels_dict)
    ).apply(pd.Series)
    for col in temp_df.columns:
        image_df[col] = temp_df[col]

    # Overwrite non-bilateral cases with known lateralities
    laterality_mapping = (
        breast_df[breast_df['study_laterality'].isin(['LEFT', 'RIGHT'])]
        .set_index('accession_number')['study_laterality']
        .to_dict()
    )
    image_df['laterality'] = image_df.apply(
        lambda row: laterality_mapping.get(row['accession_number'], '').lower()
        if row['accession_number'] in laterality_mapping
        else row['laterality'],
        axis=1,
    )

    # ── Write back only the OCR columns ──
    print(f"\nUpdating {len(image_df)} rows in database ...")
    with DatabaseManager() as db:
        update_records = image_df[['image_name'] + OCR_COLUMNS].to_dict('records')
        updated = db.insert_images_batch(update_records, upsert=True, update_only=True)
        print(f"Updated {updated} images")

    print("\nDone.")
    print(f"  Backup: {BACKUP_PATH}")
    print(f"  Updated columns: {OCR_COLUMNS}")


if __name__ == "__main__":
    main()
