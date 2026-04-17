"""Standalone script: download 100 random dual-region images from GCP,
split into left/right using machine-specific ratios, apply crop, save locally."""
import os
import sqlite3
import cv2
import numpy as np
from google.cloud import storage

BUCKET_NAME = "shared-aif-bucket-87d1"
DB_BLOB = "Databases/database_2026_1_13_main/cadbusi_2026_4_16.db"
IMAGES_PREFIX = "Databases/database_2026_1_13_main/images/"
LOCAL_DB = os.path.join(os.path.dirname(__file__), "_temp_cadbusi.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "split_regions")

SPLIT_RATIOS = {
    'EPIQ 5G':    0.501,
    'EPIQ 7G':    0.501,
    'EPIQ Elite': 0.501,
    'LOGIQE9':    0.4665,
    'LOGIQE10':   0.4665,
}

# RegionDataType values to exclude (spectral doppler)
SPECTRAL_TYPES = {'3', '4'}
SAMPLE_SIZE = 100


def download_db(client):
    """Download the DB file from GCP."""
    if os.path.exists(LOCAL_DB):
        print(f"Using cached DB: {LOCAL_DB}")
        return
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DB_BLOB)
    print(f"Downloading DB from gs://{BUCKET_NAME}/{DB_BLOB} ...")
    blob.download_to_filename(LOCAL_DB)
    print(f"Downloaded to {LOCAL_DB}")


def query_eligible_images():
    """Query DB for eligible dual-region images with crops, sample 100."""
    conn = sqlite3.connect(LOCAL_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    machines = list(SPLIT_RATIOS.keys())
    placeholders = ','.join('?' * len(machines))

    cursor.execute(f"""
        SELECT image_name, dicom_hash, manufacturer_model_name,
               region_data_type, crop_x, crop_y, crop_w, crop_h
        FROM Images
        WHERE region_count = 2
          AND manufacturer_model_name IN ({placeholders})
          AND crop_x IS NOT NULL
          AND crop_y IS NOT NULL
          AND crop_w IS NOT NULL
          AND crop_h IS NOT NULL
          AND region_data_type IS NOT NULL
          AND region_data_type != ''
        ORDER BY RANDOM()
    """, machines)

    rows = []
    for row in cursor:
        # Filter out spectral doppler types
        types = set(row['region_data_type'].split(','))
        if types & SPECTRAL_TYPES:
            continue
        rows.append(dict(row))
        if len(rows) >= SAMPLE_SIZE:
            break

    conn.close()
    print(f"Selected {len(rows)} eligible images")
    return rows


def download_and_process(client, rows):
    """Download each image from GCP, split, crop, and save."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bucket = client.bucket(BUCKET_NAME)

    saved = 0
    for i, row in enumerate(rows):
        image_name = row['image_name']
        model = row['manufacturer_model_name']
        crop_x = int(row['crop_x'])
        crop_y = int(row['crop_y'])
        crop_w = int(row['crop_w'])
        crop_h = int(row['crop_h'])

        blob_path = IMAGES_PREFIX + image_name
        blob = bucket.blob(blob_path)

        try:
            data = blob.download_as_bytes()
        except Exception as e:
            print(f"  [{i+1}] SKIP (download): {image_name} — {e}")
            continue

        arr = np.frombuffer(data, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"  [{i+1}] SKIP (decode): {image_name}")
            continue

        h, w = im.shape[:2]
        split_x = int(w * SPLIT_RATIOS[model])

        # Split
        left_full = im[:, :split_x]
        right_full = im[:, split_x:]

        # Apply crop to each half (clip to valid bounds)
        # Left: crop_x to min(crop_x+crop_w, split_x)
        lx0 = max(crop_x, 0)
        lx1 = min(crop_x + crop_w, split_x)
        left_crop = left_full[crop_y:crop_y+crop_h, lx0:lx1]

        # Right: shift x coords by split_x
        rx0 = max(crop_x - split_x, 0)
        rx1 = max(crop_x + crop_w - split_x, 0)
        right_crop = right_full[crop_y:crop_y+crop_h, rx0:rx1]

        base = os.path.splitext(image_name)[0]
        if left_crop.size > 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_0.png"), left_crop)
        if right_crop.size > 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_1.png"), right_crop)

        saved += 1
        print(f"  [{i+1}/{len(rows)}] {model}: {base}  split@{split_x}  crop=({crop_x},{crop_y},{crop_w},{crop_h})")

    print(f"\nDone. Processed {saved} images -> {OUTPUT_DIR}")


def main():
    client = storage.Client()
    download_db(client)
    rows = query_eligible_images()
    download_and_process(client, rows)


if __name__ == "__main__":
    main()
