"""
Re-run OCR mask detection + OCR on an existing database.

Backs up the DB first, then updates only:
  - description
  - laterality, area, orientation, clock_pos, nipple_dist
  (the columns derived from OCR text)

Usage:
    python tools/rerun_ocr.py <image_folder_path>

Example:
    python tools/rerun_ocr.py Databases/database_2026_1_13_main/images
"""

import sys
import os
import shutil

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
from src.ML_processing.mask_model import find_masks
from tools.storage_adapter import StorageClient

DB_PATH = "data/cadbusi.db"
BACKUP_PATH = "data/cadbusi_pre_yolo_ocr.db"

OCR_COLUMNS = ["description", "laterality", "area", "orientation", "clock_pos", "nipple_dist"]


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

    # ── Run YOLO OCR mask detection ──
    print("\nRunning YOLO OCR mask detection ...")
    _, description_masks = find_masks(image_folder_path, 'ocr_mask_yolo', image_df, 0, 0)
    masks_found = sum(1 for _, bbox in description_masks if bbox)
    print(f"Masks found: {masks_found} / {len(description_masks)}")

    # ── Run OCR ──
    print("\nPerforming OCR ...")
    descriptions = get_OCR(image_folder_path, description_masks)
    valid = sum(1 for d in descriptions.values() if d)
    print(f"Valid descriptions: {valid} / {len(descriptions)}")

    # ── Extract features from descriptions ──
    image_df['description'] = image_df['image_name'].map(descriptions)

    import pandas as pd
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
