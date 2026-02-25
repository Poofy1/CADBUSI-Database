"""Sample val/test images with lesion masks and save locally.

Queries the DB with P2 filters, restricts to validation (valid=1) and test
(valid=2) splits, keeps only images that have a matching mask in GCP
lesion_masks/, randomly samples up to 250 from each split, then downloads
the images + masks and writes a manifest CSV.

Outputs:
  {output_dir}/images/{image_name}   — source images
  {output_dir}/masks/{image_name}    — lesion mask PNGs
  {output_dir}/manifest.csv          — metadata

Usage:
    python 2026_2_24_inject_seg_annotations.py
    python 2026_2_24_inject_seg_annotations.py --dry-run
    python 2026_2_24_inject_seg_annotations.py --output-dir output/my_sample --seed 0
"""

import argparse
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_labeling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_labeling_dir)
sys.path.insert(0, _root)

from config import CONFIG
from tools.storage_adapter import StorageClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_N         = 250
ALLOWED_SCANNERS = ("LOGIQE9", "LOGIQE10", "EPIQ 7G", "EPIQ 5G", "EPIQ Elite")

# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

_SCANNERS_SQL = ", ".join(f"'{s}'" for s in ALLOWED_SCANNERS)

QUERY = f"""
SELECT
    i.image_name,
    i.patient_id,
    i.accession_number,
    i.manufacturer_model_name,
    s.has_malignant,
    s.valid
FROM Images i
JOIN StudyCases s ON i.accession_number = s.accession_number
WHERE i.image_name NOT IN (SELECT image_name FROM BadImages)
  AND s.is_biopsy = 0
  AND s.date >= '2018-01-01'
  AND s.has_malignant IN (0, 1)
  AND s.valid IN (1, 2)
  AND i.manufacturer_model_name IN ({_SCANNERS_SQL})
"""


def load_candidates(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(QUERY, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Mask discovery
# ---------------------------------------------------------------------------

def list_available_masks(storage: StorageClient, lesion_dir: str) -> set[str]:
    """Return set of filenames that have a mask in lesion_dir."""
    available: set[str] = set()
    if storage.is_gcp:
        prefix = lesion_dir.lstrip("/") + "/"
        for blob in storage._bucket.list_blobs(prefix=prefix):
            name = os.path.basename(blob.name)
            if name:
                available.add(name)
    else:
        if os.path.isdir(lesion_dir):
            available = set(os.listdir(lesion_dir))
    return available


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_one(
    image_name: str,
    images_dir: str,
    lesion_dir: str,
    out_images: Path,
    out_masks: Path,
    storage: StorageClient,
) -> tuple[str, bool, str]:
    """Download image + mask for one entry. Returns (image_name, success, error)."""
    try:
        if storage.is_gcp:
            img_bytes  = storage._bucket.blob(f"{images_dir.lstrip('/')}/{image_name}").download_as_bytes()
            mask_bytes = storage._bucket.blob(f"{lesion_dir.lstrip('/')}/{image_name}").download_as_bytes()
        else:
            img_bytes  = Path(os.path.join(images_dir, image_name)).read_bytes()
            mask_bytes = Path(os.path.join(lesion_dir,  image_name)).read_bytes()

        (out_images / image_name).write_bytes(img_bytes)
        (out_masks  / image_name).write_bytes(mask_bytes)
        return (image_name, True, "")
    except Exception as exc:
        return (image_name, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample val/test images with masks and save locally")
    parser.add_argument("--db",         default=os.path.join(_root, "data", "cadbusi.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/2026_2_24_seg_sample"))
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--workers",    type=int, default=16)
    parser.add_argument("--dry-run",    action="store_true", help="Print stats only, no downloads")
    args = parser.parse_args()

    db_dir     = CONFIG.get("DATABASE_DIR", "").rstrip("/\\")
    images_dir = f"{db_dir}/images"
    lesion_dir = f"{db_dir}/lesion_masks"

    storage = StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", ""),
    )
    print(f"Storage mode : {'GCP' if storage.is_gcp else 'local'}")
    print(f"DB           : {args.db}")
    print(f"Output dir   : {args.output_dir}")

    # Query
    print("\nQuerying candidates ...")
    df = load_candidates(args.db)
    print(f"  {len(df):,} images (val + test, P2 filters)")

    # Discover masks
    print("\nListing available masks ...")
    available_masks = list_available_masks(storage, lesion_dir)
    print(f"  {len(available_masks):,} masks in {lesion_dir}")

    # Filter
    df = df[df["image_name"].isin(available_masks)].copy()
    df_val  = df[df["valid"] == 1]
    df_test = df[df["valid"] == 2]
    print(f"  {len(df):,} images with masks  (val: {len(df_val):,} | test: {len(df_test):,})")

    # Sample
    df_val_s  = df_val.sample( n=min(SAMPLE_N, len(df_val)),  random_state=args.seed)
    df_test_s = df_test.sample(n=min(SAMPLE_N, len(df_test)), random_state=args.seed)
    df_sampled = pd.concat([df_val_s, df_test_s]).reset_index(drop=True)

    _split_map = {1: "valid", 2: "test"}
    df_sampled["split"] = df_sampled["valid"].map(_split_map)
    df_sampled["label"] = df_sampled["has_malignant"].astype(int)

    print(f"\nSampled {len(df_val_s)} val + {len(df_test_s)} test = {len(df_sampled):,} total")
    print("Label breakdown:")
    print(df_sampled["label"].value_counts().rename({0: "benign", 1: "malignant"}).to_string())

    if args.dry_run:
        print("\n[DRY RUN] No files downloaded.")
        return

    # Output dirs
    out_images = args.output_dir / "images"
    out_masks  = args.output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"\nDownloading {len(df_sampled):,} image+mask pairs with {args.workers} threads ...")
    success, errors = 0, []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                row["image_name"], images_dir, lesion_dir,
                out_images, out_masks, storage,
            ): row["image_name"]
            for _, row in df_sampled.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            name, ok, err = future.result()
            if ok:
                success += 1
            else:
                errors.append((name, err))

    print(f"\nDone: {success:,} downloaded, {len(errors):,} errors")
    if errors:
        print("First 10 errors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # Manifest — only rows that downloaded successfully
    downloaded = {p.name for p in out_images.iterdir()}
    manifest = df_sampled[df_sampled["image_name"].isin(downloaded)].copy()
    manifest["image_path"] = "images/" + manifest["image_name"]
    manifest["mask_path"]  = "masks/"  + manifest["image_name"]

    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  Manifest: {manifest_path}  ({len(manifest):,} rows)")


if __name__ == "__main__":
    main()
