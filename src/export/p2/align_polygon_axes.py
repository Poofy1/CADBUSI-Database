#!/usr/bin/env python3
"""Axis-align US region polygons to eliminate tilted edges.

Douglas-Peucker polygon approximation produces edges that are *nearly* but not
perfectly horizontal/vertical (1-20px of tilt). When FOV masking is applied,
these tilted edges create gray slivers at crop boundaries.

This script snaps near-axis-aligned edges to perfect alignment, always
contracting towards the polygon interior (conservative — we lose a thin sliver
of tissue rather than gain non-FOV pixels).

Runs BEFORE tighten_crops_fov.py. No image loading — purely geometric.

Usage:
    python data/registry/scripts/align_polygon_axes.py [OPTIONS]

    # Default: process ImageData_v6.csv
    python data/registry/scripts/align_polygon_axes.py

    # Dry run (stats only)
    python data/registry/scripts/align_polygon_axes.py --dry-run

    # Custom input/output
    python data/registry/scripts/align_polygon_axes.py --imagedata path.csv --output /tmp/out.csv
"""

import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl


# --- Constants ---

# Maximum angle (degrees) from horizontal/vertical for an edge to be snapped
DEFAULT_ANGLE_THRESHOLD = 5.0

# Maximum pixel deviation on the non-axis dimension to snap regardless of angle.
# Catches short edges where small absolute tilt produces a large angle.
# E.g., a 40px edge with 4px tilt = 5.7° (over angle threshold) but only 4px deviation.
MAX_PIXEL_DEVIATION = 20

# Minimum IoU between aligned and original polygon to accept alignment
MIN_ALIGNMENT_IOU = 0.96

# Maximum alignment passes to resolve shared-vertex conflicts
MAX_ALIGN_PASSES = 3

# Shape classification by vertex count
SHAPE_RECT = "rectangle"
SHAPE_HEX = "hexagon"
SHAPE_PENT = "pentagon"
SHAPE_FAN = "arc_fan"
SHAPE_OTHER = "other"


# --- Polygon parsing (same format as ui_mask.py) ---

def parse_polygon(s):
    """Parse 'x,y;x,y;...' -> Nx2 float array."""
    if not s:
        return None
    points = []
    for pt in s.split(";"):
        x, y = pt.split(",")
        points.append([float(x), float(y)])
    return np.array(points, dtype=np.float64)


def polygon_to_str(polygon, precision=1):
    """Convert Nx2 array -> 'x,y;x,y;...' string."""
    if polygon is None or len(polygon) == 0:
        return ""
    fmt = f"{{:.{precision}f}}"
    parts = [f"{fmt.format(x)},{fmt.format(y)}" for x, y in polygon]
    return ";".join(parts)


# --- Edge geometry ---

def edge_angle_from_axis(p1, p2):
    """Return the minimum angle (degrees) of edge p1->p2 from horizontal or vertical.

    Returns (angle, axis) where axis is 'h' (horizontal) or 'v' (vertical).
    angle is always in [0, 45].
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx == 0 and dy == 0:
        return 0.0, "h"
    angle_h = math.degrees(math.atan2(dy, dx)) if dx > 0 else 90.0
    angle_v = 90.0 - angle_h
    if angle_h <= angle_v:
        return angle_h, "h"
    else:
        return angle_v, "v"


def polygon_centroid(polygon):
    """Compute centroid of polygon vertices."""
    return polygon.mean(axis=0)


def snap_edge_conservative(p1, p2, axis, centroid):
    """Snap an edge to perfect axis alignment, contracting towards polygon interior.

    For a horizontal edge: snap both y-coords to the one closer to centroid.
    For a vertical edge: snap both x-coords to the one closer to centroid.

    Returns (new_p1, new_p2).
    """
    new_p1 = p1.copy()
    new_p2 = p2.copy()

    if axis == "h":
        # Snap y to the value closer to centroid_y (interior)
        cy = centroid[1]
        if abs(p1[1] - cy) < abs(p2[1] - cy):
            # p1 is closer to interior -> snap to the other (further from interior)
            # Wait, we want conservative = towards interior
            # "closer to centroid" means more interior
            # We want to snap to the y that's more towards the centroid
            snap_y = p1[1] if abs(p1[1] - cy) < abs(p2[1] - cy) else p2[1]
        else:
            snap_y = p2[1] if abs(p2[1] - cy) < abs(p1[1] - cy) else p1[1]
        new_p1[1] = snap_y
        new_p2[1] = snap_y
    else:  # vertical
        cx = centroid[0]
        if abs(p1[0] - cx) < abs(p2[0] - cx):
            snap_x = p1[0]
        else:
            snap_x = p2[0]
        new_p1[0] = snap_x
        new_p2[0] = snap_x

    return new_p1, new_p2


# --- IoU computation ---

def polygon_iou(poly1, poly2, img_shape=(1080, 1920)):
    """Compute IoU between two polygons by rasterization.

    img_shape should be large enough to contain both polygons.
    """
    if poly1 is None or poly2 is None:
        return 0.0
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0

    # Determine canvas size from polygon extents
    all_pts = np.vstack([poly1, poly2])
    max_x = int(np.ceil(all_pts[:, 0].max())) + 2
    max_y = int(np.ceil(all_pts[:, 1].max())) + 2
    h = min(max_y, img_shape[0])
    w = min(max_x, img_shape[1])

    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [poly1.astype(np.int32)], 1)
    cv2.fillPoly(m2, [poly2.astype(np.int32)], 1)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(intersection / union) if union > 0 else 0.0


# --- Core alignment ---

def classify_shape(n_vertices):
    """Classify polygon shape by vertex count."""
    if n_vertices == 4:
        return SHAPE_RECT
    elif n_vertices == 5:
        return SHAPE_PENT
    elif n_vertices == 6:
        return SHAPE_HEX
    elif 10 <= n_vertices <= 15:
        return SHAPE_FAN
    else:
        return SHAPE_OTHER


def _should_snap(p1, p2, angle_threshold):
    """Determine if an edge should be axis-aligned.

    Uses both angle threshold AND absolute pixel deviation to catch short
    edges where a few pixels of tilt produce a large angle.

    Returns (should_snap, axis) where axis is 'h' or 'v', or (False, None).
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx == 0 and dy == 0:
        return False, None

    angle_h = math.degrees(math.atan2(dy, dx)) if dx > 0 else 90.0
    angle_v = 90.0 - angle_h

    if angle_h <= angle_v:
        # Closer to horizontal — deviation is dy
        if angle_h == 0.0:
            return False, None  # already perfect
        if angle_h <= angle_threshold or dy <= MAX_PIXEL_DEVIATION:
            return True, "h"
    else:
        # Closer to vertical — deviation is dx
        if angle_v == 0.0:
            return False, None
        if angle_v <= angle_threshold or dx <= MAX_PIXEL_DEVIATION:
            return True, "v"

    return False, None


def _run_one_pass(aligned, centroid, angle_threshold):
    """Run one pass of edge alignment. Returns number of edges snapped."""
    n = len(aligned)
    n_snapped = 0

    for i in range(n):
        j = (i + 1) % n
        p1, p2 = aligned[i], aligned[j]

        should, axis = _should_snap(p1, p2, angle_threshold)
        if not should:
            continue

        new_p1, new_p2 = snap_edge_conservative(p1, p2, axis, centroid)
        aligned[i] = new_p1
        aligned[j] = new_p2
        n_snapped += 1

    return n_snapped


def align_polygon(polygon_str, angle_threshold=DEFAULT_ANGLE_THRESHOLD,
                  img_shape=(1080, 1920)):
    """Axis-align a polygon's near-axis edges.

    Uses multi-pass alignment to resolve shared-vertex conflicts where two
    consecutive edges snap a shared vertex to different values.

    Args:
        polygon_str: 'x,y;x,y;...' format polygon string
        angle_threshold: max degrees from axis to snap (default 5°)
        img_shape: (height, width) for IoU computation

    Returns:
        (aligned_str, shape_type, n_edges_snapped, iou) or
        (original_str, shape_type, 0, 1.0) if alignment rejected/not applicable.
    """
    polygon = parse_polygon(polygon_str)
    if polygon is None or len(polygon) < 3:
        return polygon_str, "empty", 0, 1.0

    n_verts = len(polygon)
    shape = classify_shape(n_verts)

    # Skip complex polygons (>15 vertices) — too many edges to reason about
    if shape == SHAPE_OTHER:
        return polygon_str, shape, 0, 1.0

    centroid = polygon_centroid(polygon)
    aligned = polygon.copy()
    total_snapped = 0

    # Multi-pass: repeat until no more changes (resolves shared-vertex conflicts)
    for _ in range(MAX_ALIGN_PASSES):
        n_snapped = _run_one_pass(aligned, centroid, angle_threshold)
        total_snapped += n_snapped
        if n_snapped == 0:
            break

    if total_snapped == 0:
        return polygon_str, shape, 0, 1.0

    # Verify IoU — reject if alignment changes area too much
    iou = polygon_iou(polygon, aligned, img_shape)
    if iou < MIN_ALIGNMENT_IOU:
        return polygon_str, shape, 0, iou

    aligned_str = polygon_to_str(aligned)
    return aligned_str, shape, total_snapped, iou


# --- Batch processing ---

def _process_one(args):
    """Worker: align one polygon. Returns (idx, aligned_str, shape, n_snapped, iou)."""
    idx, polygon_str, img_rows, img_cols, angle_threshold = args

    if not polygon_str:
        return (idx, None, "empty", 0, 1.0)

    h = int(img_rows) if img_rows else 1080
    w = int(img_cols) if img_cols else 1920

    aligned_str, shape, n_snapped, iou = align_polygon(
        polygon_str, angle_threshold=angle_threshold, img_shape=(h, w)
    )

    if n_snapped == 0:
        return (idx, None, shape, 0, 1.0)

    return (idx, aligned_str, shape, n_snapped, iou)


def main():
    DEFAULT_IMAGEDATA = None  # Must be provided via --imagedata

    parser = argparse.ArgumentParser(
        description="Axis-align US region polygons to eliminate tilted edges"
    )
    parser.add_argument("--imagedata", type=Path, default=DEFAULT_IMAGEDATA,
                        help="Input ImageData CSV (required)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats without writing output")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N images (0=all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: overwrite input)")
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_ANGLE_THRESHOLD,
                        help=f"Max angle from axis to snap (default: {DEFAULT_ANGLE_THRESHOLD}°)")
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

    # Identify eligible rows (have a polygon)
    can_align = (
        pl.col("us_polygon").is_not_null() & (pl.col("us_polygon") != "")
    )
    n_eligible = df.filter(can_align).height
    print(f"  Eligible (have us_polygon): {n_eligible:,} / {len(df):,}")

    # Build work items
    all_us_poly = df["us_polygon"].to_list()
    all_rows = df["rows"].to_list()
    all_cols = df["columns"].to_list()

    work_items = []
    for i in range(len(df)):
        us = all_us_poly[i]
        if not us or us == "":
            continue
        r = all_rows[i] if all_rows[i] is not None else 1080
        c = all_cols[i] if all_cols[i] is not None else 1920
        work_items.append((i, us, r, c, args.angle_threshold))

    if args.limit > 0:
        work_items = work_items[:args.limit]
        print(f"  Limited to first {args.limit}")

    n_total = len(work_items)
    print(f"\nAligning {n_total:,} polygons with {args.workers} workers "
          f"(angle threshold: {args.angle_threshold}°)...")
    print(f"  (No image loading — geometry only)")
    t0 = time.time()

    # Mutable list for updates
    new_us_poly = list(all_us_poly)

    # Stats
    n_processed = 0
    n_aligned = 0
    n_rejected_iou = 0
    shape_counts = {}
    snapped_edges_total = 0
    ious = []

    per_shape_aligned = {}
    per_shape_total = {}

    report_interval = 100000

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_process_one, work_items, chunksize=512):
            idx, aligned_str, shape, n_snapped, iou = result
            n_processed += 1

            # Track shape distribution
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            per_shape_total[shape] = per_shape_total.get(shape, 0) + 1

            if aligned_str is not None:
                new_us_poly[idx] = aligned_str
                n_aligned += 1
                snapped_edges_total += n_snapped
                ious.append(iou)
                per_shape_aligned[shape] = per_shape_aligned.get(shape, 0) + 1
            elif n_snapped == 0 and iou < MIN_ALIGNMENT_IOU:
                # Was attempted but rejected
                n_rejected_iou += 1

            if n_processed % report_interval == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"  [{n_processed:,}/{n_total:,}] "
                      f"{rate:.0f} poly/s, ETA {eta:.0f}s, "
                      f"aligned={n_aligned:,}")

    elapsed = time.time() - t0
    rate = n_processed / elapsed if elapsed > 0 else 0

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Polygon Axis Alignment Report")
    print(f"{'='*60}")
    print(f"  Processed:       {n_processed:,}")
    print(f"  Aligned:         {n_aligned:,} ({n_aligned/n_processed*100:.1f}%)")
    print(f"  Rejected (IoU):  {n_rejected_iou:,}")
    print(f"  Edges snapped:   {snapped_edges_total:,}")
    print(f"  Time:            {elapsed:.1f}s ({rate:.0f} poly/s)")

    print(f"\nShape Distribution:")
    for shape in sorted(shape_counts.keys()):
        cnt = shape_counts[shape]
        aligned_cnt = per_shape_aligned.get(shape, 0)
        pct = aligned_cnt / cnt * 100 if cnt > 0 else 0
        print(f"  {shape:12s}: {cnt:8,} total, {aligned_cnt:8,} aligned ({pct:.1f}%)")

    if ious:
        ious_arr = np.array(ious)
        print(f"\nAlignment IoU (aligned polygons only):")
        print(f"  Mean:   {ious_arr.mean():.6f}")
        print(f"  Median: {np.median(ious_arr):.6f}")
        print(f"  Min:    {ious_arr.min():.6f}")
        print(f"  <0.99:  {(ious_arr < 0.99).sum():,}")
        print(f"  <0.995: {(ious_arr < 0.995).sum():,}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written.")
        return

    # Write updated CSV
    output_path = args.output or args.imagedata
    print(f"\nWriting updated CSV to {output_path}...")

    df = df.with_columns([
        pl.Series("us_polygon", new_us_poly),
    ])

    df.write_csv(output_path)
    print(f"  Written {len(df):,} rows")

    # Quick verify
    verify = pl.read_csv(output_path, columns=["us_polygon"],
                         infer_schema_length=10000)
    n_has_poly = verify.filter(
        pl.col("us_polygon").is_not_null() & (pl.col("us_polygon") != "")
    ).height
    print(f"  Verified: {n_has_poly:,} rows with polygons")


if __name__ == "__main__":
    main()
