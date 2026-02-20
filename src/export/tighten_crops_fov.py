#!/usr/bin/env python3
"""Post-process v7 ImageData to clip crop boxes to FOV polygon boundaries.

Runs AFTER tighten_crops.py (intensity-based vertical tightening). This pass:
  1. Clips crop_x/crop_w to the FOV polygon's column extent (horizontal tightening)
  2. Clips crop_y/crop_h to the FOV polygon's row extent (fixes margin overextension)

No image loading needed — purely geometric (polygon rasterization on blank canvas).
Runs ~10x faster than the intensity-based pass.

Usage:
    python data/registry/scripts/tighten_crops_fov.py [OPTIONS]

    # Default: process ImageData_v6.csv in v7 dataset
    python data/registry/scripts/tighten_crops_fov.py

    # Dry run (report stats without writing)
    python data/registry/scripts/tighten_crops_fov.py --dry-run

    # Custom output (don't overwrite input)
    python data/registry/scripts/tighten_crops_fov.py --output /tmp/test.csv
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl


# Margin in pixels beyond the last FOV pixel (prevents clipping at polygon edge)
MARGIN_H = 2   # horizontal (columns)
MARGIN_V = 3   # vertical (rows)

# Minimum FOV pixels in a row/column to count it as having FOV coverage.
# Avoids noise from 1-2 stray pixels at polygon vertices.
MIN_FOV_PIXELS_ROW = 5
MIN_FOV_PIXELS_COL = 5

# Safety: never shrink below these fractions of original dimensions
MIN_WIDTH_RATIO = 0.90
MIN_HEIGHT_RATIO = 0.90  # Vertical was already tightened; this is just boundary clipping


def _parse_polygon(s):
    """Parse "x,y;x,y;..." -> Nx2 int32 array for cv2.fillPoly."""
    if not s:
        return None
    points = []
    for pt in s.split(";"):
        x, y = pt.split(",")
        points.append([float(x), float(y)])
    return np.array(points, dtype=np.int32)


def _rasterize_fov(us_polygon_str, debris_polygons_str, h, w):
    """Rasterize FOV mask from polygon strings. No image loading needed."""
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


def _process_one(args):
    """Worker: clip one crop to FOV extent. Returns (idx, new_x, new_y, new_w, new_h, h_reduction, v_reduction)."""
    idx, us_poly, debris_poly, img_rows, img_cols, crop_x, crop_y, crop_w, crop_h = args

    if not us_poly:
        return (idx, None, None, None, None, 0.0, 0.0)

    h, w = int(img_rows), int(img_cols)
    cx, cy, cw, ch = int(crop_x), int(crop_y), int(crop_w), int(crop_h)

    # Clamp crop to image bounds
    cy = max(0, min(cy, h - 1))
    cx = max(0, min(cx, w - 1))
    ch = min(ch, h - cy)
    cw = min(cw, w - cx)
    if cw <= 0 or ch <= 0:
        return (idx, None, None, None, None, 0.0, 0.0)

    # Rasterize FOV mask (purely from geometry)
    fov_mask = _rasterize_fov(us_poly, debris_poly, h, w)

    # Extract crop region of the mask
    mask_crop = fov_mask[cy:cy+ch, cx:cx+cw]
    if mask_crop.size == 0:
        return (idx, None, None, None, None, 0.0, 0.0)

    # --- Horizontal clipping ---
    # Per-column FOV pixel count within crop
    col_counts = np.sum(mask_crop > 0, axis=0)

    # Find first/last columns with sufficient FOV
    first_col = cw  # sentinel
    for c in range(cw):
        if col_counts[c] >= MIN_FOV_PIXELS_COL:
            first_col = c
            break

    last_col = -1  # sentinel
    for c in range(cw - 1, -1, -1):
        if col_counts[c] >= MIN_FOV_PIXELS_COL:
            last_col = c
            break

    if first_col >= last_col:
        # No valid columns found — keep original
        return (idx, None, None, None, None, 0.0, 0.0)

    new_x = cx + max(0, first_col - MARGIN_H)
    new_right = cx + min(cw, last_col + MARGIN_H + 1)
    new_w = new_right - new_x

    # --- Vertical clipping (FOV boundary only, not intensity) ---
    # Per-row FOV pixel count within crop
    row_counts = np.sum(mask_crop > 0, axis=1)

    first_row = ch
    for r in range(ch):
        if row_counts[r] >= MIN_FOV_PIXELS_ROW:
            first_row = r
            break

    last_row = -1
    for r in range(ch - 1, -1, -1):
        if row_counts[r] >= MIN_FOV_PIXELS_ROW:
            last_row = r
            break

    if first_row >= last_row:
        return (idx, None, None, None, None, 0.0, 0.0)

    new_y = cy + max(0, first_row - MARGIN_V)
    new_bottom = cy + min(ch, last_row + MARGIN_V + 1)
    new_h = new_bottom - new_y

    # Safety checks
    if new_w < cw * MIN_WIDTH_RATIO and new_w < cw:
        # Horizontal reduction too aggressive — skip horizontal
        new_x = cx
        new_w = cw

    if new_h < ch * MIN_HEIGHT_RATIO and new_h < ch:
        # Vertical reduction too aggressive — skip vertical
        new_y = cy
        new_h = ch

    # Compute reductions
    h_reduction = (1 - new_w / cw) * 100 if cw > 0 else 0.0
    v_reduction = (1 - new_h / ch) * 100 if ch > 0 else 0.0

    # Only report changes > 0.1%
    changed = (h_reduction > 0.1) or (v_reduction > 0.1)
    if not changed:
        return (idx, None, None, None, None, 0.0, 0.0)

    return (idx, new_x, new_y, new_w, new_h, h_reduction, v_reduction)


def main():
    DEFAULT_IMAGEDATA = None  # Must be provided via --imagedata

    parser = argparse.ArgumentParser(description="Clip v7 crops to FOV polygon boundaries")
    parser.add_argument("--imagedata", type=Path, default=DEFAULT_IMAGEDATA)
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

    # Identify eligible rows
    can_clip = (
        pl.col("us_polygon").is_not_null() & (pl.col("us_polygon") != "") &
        pl.col("crop_w").is_not_null() & (pl.col("crop_w") > 0) &
        pl.col("crop_h").is_not_null() & (pl.col("crop_h") > 0) &
        pl.col("rows").is_not_null() & pl.col("columns").is_not_null()
    )
    n_eligible = df.filter(can_clip).height
    print(f"  Eligible: {n_eligible:,} / {len(df):,}")

    # Build work items with original DataFrame index
    all_us_poly = df["us_polygon"].to_list()
    all_debris = df["debris_polygons"].to_list()
    all_rows = df["rows"].to_list()
    all_cols = df["columns"].to_list()
    all_crop_x = df["crop_x"].to_list()
    all_crop_y = df["crop_y"].to_list()
    all_crop_w = df["crop_w"].to_list()
    all_crop_h = df["crop_h"].to_list()

    work_items = []
    for i in range(len(df)):
        us = all_us_poly[i]
        if not us or us == "":
            continue
        cx, cy, cw, ch = all_crop_x[i], all_crop_y[i], all_crop_w[i], all_crop_h[i]
        if cw is None or cw <= 0 or ch is None or ch <= 0:
            continue
        r, c = all_rows[i], all_cols[i]
        if r is None or c is None:
            continue
        work_items.append((i, us, all_debris[i], r, c, cx, cy, cw, ch))

    if args.limit > 0:
        work_items = work_items[:args.limit]
        print(f"  Limited to first {args.limit}")

    n_total = len(work_items)
    print(f"\nProcessing {n_total:,} images with {args.workers} workers...")
    print(f"  (No image loading — polygon rasterization only)")
    t0 = time.time()

    # Mutable lists for updates
    new_crop_x = list(all_crop_x)
    new_crop_y = list(all_crop_y)
    new_crop_w = list(all_crop_w)
    new_crop_h = list(all_crop_h)

    n_processed = 0
    n_h_tightened = 0
    n_v_tightened = 0
    n_either = 0
    h_reductions = []
    v_reductions = []
    total_pixels_saved = 0

    report_interval = 50000

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_process_one, work_items, chunksize=256):
            idx, nx, ny, nw, nh, h_red, v_red = result
            n_processed += 1

            if nx is not None:
                orig_area = int(all_crop_w[idx]) * int(all_crop_h[idx])

                if h_red > 0.1:
                    n_h_tightened += 1
                    h_reductions.append(h_red)
                if v_red > 0.1:
                    n_v_tightened += 1
                    v_reductions.append(v_red)
                n_either += 1

                new_crop_x[idx] = nx
                new_crop_y[idx] = ny
                new_crop_w[idx] = nw
                new_crop_h[idx] = nh

                new_area = nw * nh
                total_pixels_saved += orig_area - new_area

            if n_processed % report_interval == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"  [{n_processed:,}/{n_total:,}] "
                      f"{rate:.0f} img/s, ETA {eta:.0f}s, "
                      f"h_tight={n_h_tightened:,}, v_tight={n_v_tightened:,}")

    elapsed = time.time() - t0
    rate = n_processed / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({rate:.0f} img/s)")
    print(f"  Processed: {n_processed:,}")
    print(f"  Horizontally tightened: {n_h_tightened:,} "
          f"({n_h_tightened/n_processed*100:.1f}%)")
    print(f"  Vertically clipped (FOV boundary): {n_v_tightened:,} "
          f"({n_v_tightened/n_processed*100:.1f}%)")
    print(f"  Either: {n_either:,} ({n_either/n_processed*100:.1f}%)")

    if h_reductions:
        print(f"\n  Horizontal reduction stats:")
        print(f"    Mean: {np.mean(h_reductions):.1f}%")
        print(f"    Median: {np.median(h_reductions):.1f}%")
        print(f"    Max: {np.max(h_reductions):.1f}%")

    if v_reductions:
        print(f"\n  Vertical FOV-clip stats:")
        print(f"    Mean: {np.mean(v_reductions):.1f}%")
        print(f"    Median: {np.median(v_reductions):.1f}%")
        print(f"    Max: {np.max(v_reductions):.1f}%")

    print(f"\n  Total pixels saved: {total_pixels_saved:,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write updated CSV
    output_path = args.output or args.imagedata
    print(f"\nWriting updated CSV to {output_path}...")

    df = df.with_columns([
        pl.Series("crop_x", new_crop_x),
        pl.Series("crop_y", new_crop_y),
        pl.Series("crop_w", new_crop_w),
        pl.Series("crop_h", new_crop_h),
    ])

    df.write_csv(output_path)
    print(f"  Written {len(df):,} rows")

    # Quick verify
    verify = pl.read_csv(output_path, columns=["crop_x", "crop_y", "crop_w", "crop_h"],
                         infer_schema_length=10000)
    print(f"  Verified: {len(verify):,} rows readable")


if __name__ == "__main__":
    main()
