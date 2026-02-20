#!/usr/bin/env python3
"""Post-process v7 ImageData to tighten crop boxes using signal intensity.

Reads the existing ImageData CSV, loads each image, and trims dark empty rows
from the crop bottom (and top) based on mean pixel intensity within the FOV mask.
Writes an updated CSV with tightened crop_y and crop_h values.

Uses multiprocessing for throughput (~100-150 img/s on 8 cores).

Usage:
    python data/registry/scripts/tighten_crops.py [OPTIONS]

    # Default: process ImageData_v6.csv in v7 dataset
    python data/registry/scripts/tighten_crops.py

    # Dry run (report stats without writing)
    python data/registry/scripts/tighten_crops.py --dry-run

    # Process only a subset (for testing)
    python data/registry/scripts/tighten_crops.py --limit 1000

    # Custom workers
    python data/registry/scripts/tighten_crops.py --workers 12
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import warnings

import cv2
import numpy as np
import polars as pl

warnings.filterwarnings("ignore", "Mean of empty slice")

# Tightening parameters (validated on 1000 images across 5 scanners)
INTENSITY_THRESH = 12
WINDOW = 15
MARGIN = 10
MIN_FOV_PIXELS = 20
MIN_HEIGHT_RATIO = 0.5  # never shrink below 50% of original


def _parse_polygon(s):
    """Parse "x,y;x,y;..." -> Nx2 int32 array for cv2.fillPoly."""
    if not s:
        return None
    points = []
    for pt in s.split(";"):
        x, y = pt.split(",")
        points.append([float(x), float(y)])
    return np.array(points, dtype=np.int32)


def _compute_fov_mask(us_polygon_str, debris_polygons_str, h, w):
    """Rasterize FOV mask. Inlined to avoid import in worker processes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    fov = _parse_polygon(us_polygon_str)
    if fov is None:
        return mask
    cv2.fillPoly(mask, [fov], 255)
    if debris_polygons_str:
        for poly_str in debris_polygons_str.split("|"):
            poly_str = poly_str.strip()
            if poly_str:
                debris = _parse_polygon(poly_str)
                if debris is not None:
                    cv2.fillPoly(mask, [debris], 0)
    return mask


def _tighten_one(gray, fov_mask, crop_y, crop_h):
    """Core tightening: returns (new_y, new_h)."""
    h = gray.shape[0]

    masked = np.where(fov_mask > 0, gray.astype(np.float32), np.nan)
    with np.errstate(all='ignore'):
        row_means = np.nanmean(masked, axis=1)
    row_counts = np.sum(fov_mask > 0, axis=1)

    # Smooth with sliding window
    smoothed = np.full(h, np.nan)
    valid = (row_counts >= MIN_FOV_PIXELS) & ~np.isnan(row_means)
    for r in range(h):
        s = max(0, r - WINDOW // 2)
        e = min(h, r + WINDOW // 2 + 1)
        vals = row_means[s:e][valid[s:e]]
        if len(vals) > 0:
            smoothed[r] = vals.mean()

    # Bottom: scan upward
    signal_bottom = crop_y + crop_h - 1
    for r in range(h - 1, -1, -1):
        if not np.isnan(smoothed[r]) and smoothed[r] >= INTENSITY_THRESH:
            signal_bottom = r
            break

    # Top: scan downward
    signal_top = crop_y
    for r in range(h):
        if not np.isnan(smoothed[r]) and smoothed[r] >= INTENSITY_THRESH:
            signal_top = r
            break

    new_y = max(crop_y, signal_top - MARGIN)
    new_bottom = min(crop_y + crop_h, signal_bottom + MARGIN + 1)
    new_h = new_bottom - new_y

    if new_h < crop_h * MIN_HEIGHT_RATIO:
        return crop_y, crop_h

    return new_y, new_h


# Global for worker processes (set via initializer)
_IMAGE_DIR = None


def _worker_init(image_dir_str):
    global _IMAGE_DIR
    _IMAGE_DIR = Path(image_dir_str)


def _process_one(args):
    """Worker function: process one image, return (image_name, new_y, new_h, reduction)."""
    image_name, us_poly, debris_poly, crop_y, crop_h, crop_w = args

    img_path = _IMAGE_DIR / image_name
    if not img_path.exists():
        return (image_name, None, None, 0.0)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (image_name, None, None, 0.0)

    h, w = img.shape
    fov_mask = _compute_fov_mask(us_poly, debris_poly, h, w)

    new_y, new_h = _tighten_one(img, fov_mask, int(crop_y), int(crop_h))

    orig_area = int(crop_w) * int(crop_h)
    new_area = int(crop_w) * new_h
    reduction = (1 - new_area / orig_area) * 100 if orig_area > 0 else 0.0

    if reduction > 1.0:
        return (image_name, new_y, new_h, reduction)
    else:
        return (image_name, None, None, 0.0)


def main():
    DEFAULT_IMAGEDATA = None  # Must be provided via --imagedata
    DEFAULT_IMAGE_DIR = None  # Must be provided via --image-dir

    parser = argparse.ArgumentParser(description="Tighten v7 crop boxes")
    parser.add_argument("--imagedata", type=Path, default=DEFAULT_IMAGEDATA)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats without writing output")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N images (0=all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: overwrite input)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2),
                        help=f"Worker processes (default: {max(1, mp.cpu_count()-2)})")
    args = parser.parse_args()

    if args.imagedata is None:
        parser.error("--imagedata is required")
    if not args.imagedata.exists():
        print(f"ERROR: {args.imagedata} not found")
        sys.exit(1)

    print(f"Loading {args.imagedata}...")
    t0 = time.time()
    df = pl.read_csv(args.imagedata, infer_schema_length=10000)
    print(f"  {len(df):,} rows, {len(df.columns)} columns in {time.time()-t0:.1f}s")

    # Identify eligible rows (have polygon + valid crop)
    can_tighten = (
        pl.col("us_polygon").is_not_null() & (pl.col("us_polygon") != "") &
        pl.col("crop_w").is_not_null() & (pl.col("crop_w") > 0) &
        pl.col("crop_h").is_not_null() & (pl.col("crop_h") > 0)
    )
    eligible = df.filter(can_tighten)
    print(f"  Eligible for tightening: {len(eligible):,} / {len(df):,}")

    if args.limit > 0:
        eligible = eligible.head(args.limit)
        print(f"  Limited to first {args.limit}")

    # Build work items
    work_items = list(zip(
        eligible["image_name"].to_list(),
        eligible["us_polygon"].to_list(),
        eligible["debris_polygons"].to_list(),
        eligible["crop_y"].to_list(),
        eligible["crop_h"].to_list(),
        eligible["crop_w"].to_list(),
    ))

    # Process with multiprocessing
    n_total = len(work_items)
    print(f"\nProcessing {n_total:,} images with {args.workers} workers...")
    t0 = time.time()

    # Index for updating: image_name -> position in df
    image_names_all = df["image_name"].to_list()
    name_to_idx = {name: i for i, name in enumerate(image_names_all)}

    new_crop_y = df["crop_y"].to_list()
    new_crop_h = df["crop_h"].to_list()
    all_crop_w = df["crop_w"].to_list()  # for pixel savings calc

    n_processed = 0
    n_tightened = 0
    reductions = []
    total_pixels_saved = 0

    report_interval = 20000

    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(str(args.image_dir),)) as pool:
        for result in pool.imap_unordered(_process_one, work_items, chunksize=64):
            image_name, tight_y, tight_h, reduction = result
            n_processed += 1

            if tight_y is not None and reduction > 1.0:
                n_tightened += 1
                reductions.append(reduction)

                idx = name_to_idx[image_name]
                orig_h = int(new_crop_h[idx])
                cw = int(all_crop_w[idx])
                total_pixels_saved += cw * (orig_h - tight_h)

                new_crop_y[idx] = tight_y
                new_crop_h[idx] = tight_h

            if n_processed % report_interval == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"  [{n_processed:,}/{n_total:,}] "
                      f"{rate:.0f} img/s, ETA {eta/60:.1f}min, "
                      f"tightened {n_tightened:,}")

    elapsed = time.time() - t0
    rate = n_processed / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed/60:.1f}min ({rate:.0f} img/s)")
    print(f"  Processed: {n_processed:,}")
    print(f"  Tightened (>1%): {n_tightened:,} "
          f"({n_tightened/n_processed*100:.1f}% of processed)")

    if reductions:
        print(f"  Mean reduction: {np.mean(reductions):.1f}%")
        print(f"  Median reduction: {np.median(reductions):.1f}%")
        print(f"  Max reduction: {np.max(reductions):.1f}%")
        print(f"  Total pixels saved: {total_pixels_saved:,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write updated CSV
    output_path = args.output or args.imagedata
    print(f"\nWriting updated CSV to {output_path}...")

    df = df.with_columns([
        pl.Series("crop_y", new_crop_y),
        pl.Series("crop_h", new_crop_h),
    ])

    df.write_csv(output_path)
    print(f"  Written {len(df):,} rows")

    # Quick verify
    verify = pl.read_csv(output_path, columns=["crop_y", "crop_h"],
                         infer_schema_length=10000)
    print(f"  Verified: {len(verify):,} rows readable")


if __name__ == "__main__":
    main()
