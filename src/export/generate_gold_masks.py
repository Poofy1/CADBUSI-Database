#!/usr/bin/env python3
"""Download gold lesion mask annotations from GCP/local storage.

Applies the same study/scanner filters as P0.yaml so the gold mask set
is consistent with the P0/P2 training datasets. Each mask is a PNG file
stored alongside the source image metadata in manifest.csv.

Outputs:
  {output_dir}/masks/{mask_image}   — downloaded lesion mask PNGs
  {output_dir}/manifest.csv         — metadata with split, label, bbox, paths

Usage:
  python generate_gold_masks.py
  python generate_gold_masks.py --config configs/P0.yaml --output-dir output/gold_masks
  python generate_gold_masks.py --workers 16 --resume
"""

import argparse
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Set up sys.path: repo root, src/export/config_processing/
_export = Path(__file__).resolve().parent
_root = _export.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export / "config_processing"))

from config import CONFIG
from tools.storage_adapter import StorageClient
from export_configurable import ExportConfig, compute_split
from pipeline_common import resolve_output_dir


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def build_gold_mask_query(config: ExportConfig) -> str:
    """Build SQL query for LesionLabels joined with Images + StudyCases."""
    conditions = [
        "ll.mask_image IS NOT NULL",
        "ll.mask_image != ''",
        "s.has_malignant IN (0, 1)",
        "i.image_name NOT IN (SELECT image_name FROM BadImages)",
    ]

    stf = config.study_filters
    if stf.is_biopsy is not None:
        conditions.append(f"s.is_biopsy = {stf.is_biopsy}")
    if stf.min_year is not None:
        conditions.append(f"s.date >= '{stf.min_year}-01-01'")
    if stf.max_year is not None:
        conditions.append(f"s.date <= '{stf.max_year}-12-31'")

    sf = config.scanner_filters
    if sf.allowed_scanners:
        names = ", ".join(f"'{s}'" for s in sf.allowed_scanners)
        conditions.append(f"i.manufacturer_model_name IN ({names})")
    elif sf.exclude_scanners:
        names = ", ".join(f"'{s}'" for s in sf.exclude_scanners)
        conditions.append(f"i.manufacturer_model_name NOT IN ({names})")

    where = "\n    AND ".join(conditions)
    return f"""
    SELECT
        ll.id          AS lesion_id,
        ll.mask_image,
        ll.x1, ll.y1, ll.x2, ll.y2,
        ll.quality,
        i.image_name,
        i.patient_id,
        i.accession_number,
        i.manufacturer_model_name,
        i.crop_x, i.crop_y, i.crop_w, i.crop_h,
        s.has_malignant,
        s.date
    FROM LesionLabels ll
    JOIN Images i ON ll.dicom_hash = i.dicom_hash
    JOIN StudyCases s ON i.accession_number = s.accession_number
    WHERE {where}
    """


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_one(
    mask_image: str,
    lesion_dir: str,
    output_dir: Path,
    storage: StorageClient,
) -> tuple[str, bool, str]:
    """Download one mask PNG. Returns (mask_image, success, error_msg)."""
    out_path = output_dir / "masks" / mask_image
    try:
        if storage.is_gcp:
            blob_path = f"{lesion_dir}/{mask_image}".replace("//", "/").lstrip("/")
            blob = storage._bucket.blob(blob_path)
            data = blob.download_as_bytes()
        else:
            src = Path(lesion_dir) / mask_image
            data = src.read_bytes()
        out_path.write_bytes(data)
        return (mask_image, True, "")
    except Exception as exc:
        return (mask_image, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download gold lesion masks with P0-equivalent filters"
    )
    parser.add_argument(
        "--db",
        default=str(_root / "data" / "cadbusi.db"),
        help="Path to cadbusi.db",
    )
    parser.add_argument(
        "--config",
        default=str(_export / "configs" / "P0.yaml"),
        help="Dataset YAML config to use for filtering (default: P0.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/gold_masks"),
        help="Output directory (default: output/gold_masks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel download threads (default: 16)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip masks that already exist in output dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without downloading",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N masks, 0 = all (default: 0)",
    )
    args = parser.parse_args()

    # Storage
    db_dir = CONFIG.get("DATABASE_DIR", "").rstrip("/\\")
    lesion_dir = f"{db_dir}/lesion_labels"
    storage = StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", ""),
    )
    print(f"Storage mode : {'GCP' if storage.is_gcp else 'local'}")
    print(f"Lesion dir   : {lesion_dir}")

    # Config + query
    config = ExportConfig.from_yaml(Path(args.config))
    query = build_gold_mask_query(config)

    print(f"\nQuerying {args.db} ...")
    conn = sqlite3.connect(args.db)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  {len(df):,} lesion masks found")

    # Compute split
    df["split"] = df["patient_id"].apply(
        lambda pid: ["test", "val", "train"][compute_split(pid, config.split)]
    )
    df["label"] = df["has_malignant"].astype(int)

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"  Limited to first {args.limit}")

    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    if args.dry_run:
        print(f"\n[DRY RUN] Would download {len(df):,} masks to {args.output_dir}")
        return

    # Output dir
    output_dir = resolve_output_dir(args.output_dir, args.resume)
    if output_dir != args.output_dir:
        print(f"  Output dir   : {output_dir}  (auto-incremented)")
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "export_config.yaml")

    rows = df.to_dict("records")

    if args.resume:
        existing = {p.name for p in (output_dir / "masks").iterdir()}
        before = len(rows)
        rows = [r for r in rows if r["mask_image"] not in existing]
        print(f"\nResume: skipped {before - len(rows):,} existing, {len(rows):,} remaining")

    if not rows:
        print("Nothing to download.")
        return

    print(f"\nDownloading {len(rows):,} masks with {args.workers} threads ...")
    success = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, r["mask_image"], lesion_dir, output_dir, storage): r
            for r in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            mask_image, ok, err = future.result()
            if ok:
                success += 1
            else:
                errors.append((mask_image, err))

    print(f"\nDone: {success:,} downloaded, {len(errors):,} errors")
    if errors:
        print("First 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # Manifest — only include successfully downloaded masks
    masks_dir = output_dir / "masks"
    manifest_rows = [
        {
            **row,
            "mask_path": f"masks/{row['mask_image']}",
        }
        for row in df.to_dict("records")
        if (masks_dir / row["mask_image"]).exists()
    ]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Manifest: {manifest_path} ({len(manifest_df):,} rows)")


if __name__ == "__main__":
    main()
