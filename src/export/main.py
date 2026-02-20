#!/usr/bin/env python3
"""Preprocessing pipeline: SQLite DB → GCP image download → crop/mask → local output.

Pipeline per image:
  1. Query Images + StudyCases from SQLite DB
  2. Download raw PNG from GCP (multithreaded)
  3. Rasterize inclusion mask (FOV minus debris)
  4. Crop image + mask to optimized crop box
  5. Fill non-tissue pixels with gray (configurable)
  6. Resize to target_size (aspect-preserving, top-left aligned)
  7. Compute 16x16 patch tissue count map
  8. Save image, mask, patch map locally

Outputs (all local):
  {output_dir}/images/{image_name}.png        — target_size RGB
  {output_dir}/masks/{image_name}.png         — target_size grayscale (255=tissue)
  {output_dir}/patch_tissue/{image_name}.png  — 16x16 grayscale patch map
  {output_dir}/manifest.csv                   — metadata manifest

Usage:
  python main.py --db /path/to/cadbusi.db --output-dir ./output/preprocessed
  python main.py --db cadbusi.db --labeled-only --workers 24 --resume
"""

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Project root on sys.path so we can import tools/ and config
_root = Path(__file__).resolve().parent.parent
_pipeline = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_pipeline))
sys.path.insert(0, str(_pipeline / "config_processing"))

from config import CONFIG
from tools.storage_adapter import StorageClient
from ui_mask import compute_ui_mask
from export_configurable import ExportConfig


# ---------------------------------------------------------------------------
# Config -> SQL
# ---------------------------------------------------------------------------

def build_query(config: ExportConfig, raw_cfg: dict) -> str:
    """Build the Images+StudyCases SQL query from a structured ExportConfig."""
    conditions = [
        "i.crop_x IS NOT NULL",
        "i.crop_w IS NOT NULL",
        "i.crop_h IS NOT NULL",
        "i.image_name NOT IN (SELECT image_name FROM BadImages)",
    ]

    sf = config.scanner_filters
    if sf.allowed_scanners:
        names = ", ".join(f"'{s}'" for s in sf.allowed_scanners)
        conditions.append(f"i.manufacturer_model_name IN ({names})")
    elif sf.exclude_scanners:
        names = ", ".join(f"'{s}'" for s in sf.exclude_scanners)
        conditions.append(f"i.manufacturer_model_name NOT IN ({names})")

    stf = config.study_filters
    if stf.min_year is not None:
        conditions.append(f"s.date >= '{stf.min_year}-01-01'")
    if stf.max_year is not None:
        conditions.append(f"s.date <= '{stf.max_year}-12-31'")
    if stf.is_biopsy is not None:
        conditions.append(f"s.is_biopsy = {stf.is_biopsy}")

    # Extra fields not in ExportConfig
    raw_stf = raw_cfg.get("study_filters", {})
    if raw_stf.get("exclude_unknown_label"):
        conditions.append("(s.has_malignant != -1 OR s.has_malignant IS NULL)")

    imf = config.image_filters
    if imf.darkness_max is not None:
        conditions.append(f"(i.darkness IS NULL OR i.darkness <= {imf.darkness_max})")

    where = "\n    AND ".join(conditions)
    return f"""
    SELECT
        i.image_id,
        i.image_name,
        i.accession_number,
        i.patient_id,
        i.laterality,
        i.crop_x, i.crop_y, i.crop_w, i.crop_h,
        i.crop_aspect_ratio,
        i.us_polygon,
        i.debris_polygons,
        i.rows,
        i.columns,
        i.manufacturer_model_name,
        i.has_calipers,
        i.label,
        i.darkness,
        i.area,
        i.region_count,
        s.has_malignant,
        s.has_benign,
        s.valid,
        s.bi_rads,
        s.date
    FROM Images i
    LEFT JOIN StudyCases s ON i.accession_number = s.accession_number
    WHERE {where}
    """


def apply_image_filters(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    """Apply image-level filters in Python after SQL load."""
    imf = config.image_filters

    if imf.allowed_areas:
        before = len(df)
        df = df[df["area"].isin(imf.allowed_areas) | df["area"].isna()].copy()
        print(f"  After area filter        : {len(df):,}  (-{before - len(df):,})")

    if imf.region_count_max:
        before = len(df)
        df = df[df["region_count"].fillna(1) <= imf.region_count_max].copy()
        print(f"  After region_count filter: {len(df):,}  (-{before - len(df):,})")

    if imf.aspect_ratio_min is not None and "crop_aspect_ratio" in df.columns:
        before = len(df)
        ar = df["crop_aspect_ratio"].fillna(1.0)
        df = df[ar.between(imf.aspect_ratio_min, imf.aspect_ratio_max)].copy()
        print(f"  After aspect_ratio filter: {len(df):,}  (-{before - len(df):,})")

    if imf.min_dimension is not None:
        before = len(df)
        ok = (df["crop_w"].fillna(0) >= imf.min_dimension) & (df["crop_h"].fillna(0) >= imf.min_dimension)
        df = df[ok].copy()
        print(f"  After min_dimension filter: {len(df):,}  (-{before - len(df):,})")

    return df


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

def load_from_db(db_path: str, query: str) -> pd.DataFrame:
    """Execute a SQL query against the database and return results as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Loaded {len(df):,} images from DB")
    return df


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray, row: dict, target_size: int, fill: int):
    """Crop, mask, resize one image.

    Args:
        img:         BGR numpy array (H, W, 3).
        row:         Dict with crop_x/y/w/h, us_polygon, debris_polygons.
        target_size: Output canvas size in pixels.
        fill:        Gray fill value for non-tissue pixels.

    Returns:
        (canvas_img, canvas_mask, patch_counts)
    """
    img_h, img_w = img.shape[:2]

    # Rasterize inclusion mask
    us_poly = row.get("us_polygon") or None
    debris_poly = row.get("debris_polygons") or None
    if isinstance(us_poly, float):
        us_poly = None
    if isinstance(debris_poly, float):
        debris_poly = None
    mask_full = compute_ui_mask(us_poly, debris_poly, img_h, img_w)

    # Crop (clamped to image bounds)
    cx = max(0, int(row["crop_x"]))
    cy = max(0, int(row["crop_y"]))
    cw = min(int(row["crop_w"]), img_w - cx)
    ch = min(int(row["crop_h"]), img_h - cy)

    img_crop = img[cy : cy + ch, cx : cx + cw].copy()
    mask_crop = mask_full[cy : cy + ch, cx : cx + cw]

    # Fill non-tissue
    img_crop[mask_crop == 0] = fill

    # Aspect-preserving resize
    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Place on canvas (top-left aligned)
    canvas_img = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas_img[:new_h, :new_w] = img_resized
    canvas_mask[:new_h, :new_w] = mask_resized

    # 16x16 patch tissue counts
    ps = target_size // 16
    patch_counts = np.zeros((16, 16), dtype=np.uint8)
    for py in range(16):
        for px in range(16):
            patch = canvas_mask[py * ps : (py + 1) * ps, px * ps : (px + 1) * ps]
            patch_counts[py, px] = min(int(np.count_nonzero(patch)), 255)

    return canvas_img, canvas_mask, patch_counts


def preprocess_p0(img: np.ndarray, row: dict, target_size: int, fill: int) -> np.ndarray:
    """Simple crop + center letterbox. No polygons, no masks, no patch maps."""
    cx = max(0, int(row["crop_x"]))
    cy = max(0, int(row["crop_y"]))
    cw = min(int(row["crop_w"]), img.shape[1] - cx)
    ch = min(int(row["crop_h"]), img.shape[0] - cy)
    img_crop = img[cy:cy + ch, cx:cx + cw]

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img_resized
    return canvas


# ---------------------------------------------------------------------------
# Per-image worker (download + process + save)
# ---------------------------------------------------------------------------

def _download_bytes(image_name: str, image_dir: str, storage: StorageClient) -> bytes:
    """Download image bytes from GCP or local filesystem."""
    if storage.is_gcp:
        blob_path = f"{image_dir}/{image_name}".replace("//", "/").lstrip("/")
        blob = storage._bucket.blob(blob_path)
        return blob.download_as_bytes()
    else:
        file_path = Path(image_dir) / image_name
        return file_path.read_bytes()


def process_one(
    row: dict,
    image_dir: str,
    output_dir: Path,
    target_size: int,
    fill: int,
    storage: StorageClient,
    pipeline: str = "structural_tissue_aware",
) -> tuple[str, bool, str]:
    """Download + preprocess + save one image.

    Returns (image_name, success, error_msg).
    """
    name = row["image_name"]
    try:
        img_bytes = _download_bytes(name, image_dir, storage)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")

        if pipeline == "simple_crop_letterbox":
            canvas_img = preprocess_p0(img, row, target_size, fill)
            cv2.imwrite(str(output_dir / "images" / name), canvas_img)
        else:
            canvas_img, canvas_mask, patch_counts = preprocess_image(img, row, target_size, fill)
            cv2.imwrite(str(output_dir / "images" / name), canvas_img)
            cv2.imwrite(str(output_dir / "masks" / name), canvas_mask)
            cv2.imwrite(str(output_dir / "patch_tissue" / name), patch_counts)

        return (name, True, "")
    except Exception as exc:
        return (name, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline: SQLite DB -> GCP images -> local output"
    )
    parser.add_argument(
        "--db",
        default=str(Path(__file__).parent.parent.parent / "data" / "cadbusi.db"),
        help="Path to cadbusi.db SQLite file (default: <repo-root>/data/cadbusi.db)",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "configs" / "dataset.yaml"),
        help="Path to dataset YAML config (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/preprocessed"),
        help="Local output directory (default: output/preprocessed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download/process threads (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N images, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already exist in output_dir/images/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without processing",
    )
    args = parser.parse_args()

    # Resolve from config
    image_dir = CONFIG.get("DATABASE_DIR", "").rstrip("/\\") + "/images"
    storage = StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", ""),
    )
    mode = "GCP" if storage.is_gcp else "local"
    print(f"Storage mode : {mode}")
    print(f"Image source : {image_dir}")

    # Load dataset config
    with open(args.dataset) as f:
        raw_cfg = yaml.safe_load(f)
    config = ExportConfig.from_yaml(Path(args.dataset))
    pre = raw_cfg.get("preprocessing", {})
    target_size = pre.get("target_size", 256)
    fill = pre.get("fill", 128)
    pipeline = pre.get("pipeline", "structural_tissue_aware")
    print(f"Dataset : {config.name}")
    print(f"          {config.description}")
    print(f"  target_size={target_size}, fill={fill}, pipeline={pipeline}")

    # Load data from DB
    query = build_query(config, raw_cfg)
    print(f"\nLoading DB: {args.db}")
    df = load_from_db(args.db, query)
    df = apply_image_filters(df, config)

    if args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"  After --limit: {len(df):,}")

    if df.empty:
        print("No images to process — exiting.")
        return

    if args.dry_run:
        sample = df.head(1_000).copy()
        print(f"\n[DRY RUN] Running on {len(sample):,} of {len(df):,} total images")
        print(f"  Output  : {args.output_dir}")
        print(f"  Size    : {target_size}px, fill={fill}")
        print(f"  Workers : {args.workers}")
        if "has_malignant" in df.columns:
            vc = df["has_malignant"].value_counts(dropna=False)
            print(f"  has_malignant distribution:\n{vc.to_string()}")
        df = sample

    # Create output subdirectories
    output_dir = args.output_dir
    subdirs = ["images"] if pipeline == "simple_crop_letterbox" else ["images", "masks", "patch_tissue"]
    for sub in subdirs:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = df.to_dict("records")

    if args.resume:
        existing = {p.name for p in (output_dir / "images").iterdir()}
        before = len(rows)
        rows = [r for r in rows if r["image_name"] not in existing]
        print(f"\nResume: skipped {before - len(rows):,} existing, {len(rows):,} remaining")

    if not rows:
        print("Nothing left to process.")
        return

    print(f"\nProcessing {len(rows):,} images with {args.workers} threads ...")

    t0 = time.time()
    success = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one, row, image_dir, output_dir, target_size, fill, storage, pipeline
            ): row["image_name"]
            for row in rows
        }

        for i, future in enumerate(as_completed(futures)):
            name, ok, err = future.result()
            if ok:
                success += 1
            else:
                errors.append((name, err))

            n = i + 1
            if n % 500 == 0 or n == len(rows):
                elapsed = time.time() - t0
                rate = n / elapsed
                eta = (len(rows) - n) / rate if rate > 0 else 0
                print(
                    f"  [{n:,}/{len(rows):,}]  "
                    f"{rate:.1f} img/s  |  "
                    f"elapsed {elapsed:.0f}s  |  "
                    f"ETA {eta:.0f}s  |  "
                    f"errors: {len(errors)}"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {success:,} ok, {len(errors):,} errors")

    if errors:
        print(f"\nFirst 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # Manifest CSV (only successfully written images)
    print("\nWriting manifest ...")
    img_dir = output_dir / "images"
    if pipeline == "simple_crop_letterbox":
        manifest_rows = [
            {**row, "image_path": f"images/{row['image_name']}"}
            for row in df.to_dict("records")
            if (img_dir / row["image_name"]).exists()
        ]
    else:
        manifest_rows = [
            {
                **row,
                "image_path": f"images/{row['image_name']}",
                "mask_path": f"masks/{row['image_name']}",
                "patch_tissue_path": f"patch_tissue/{row['image_name']}",
            }
            for row in df.to_dict("records")
            if (img_dir / row["image_name"]).exists()
        ]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Manifest: {manifest_path} ({len(manifest_df):,} rows)")


if __name__ == "__main__":
    main()
