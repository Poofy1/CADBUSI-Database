#!/usr/bin/env python3
"""Transform gold lesion masks into P0/P2 256x256 space.

Applies the exact same spatial transform (crop + resize + canvas) as the
P0 and P2 image pipelines so the gold masks align pixel-for-pixel with
the processed images.

Outputs:
  {output_dir}/masks/{image_name}   — 256x256 transformed gold mask
  {output_dir}/manifest.csv         — original manifest columns + mask_path

Usage:
  python src/export/transform_gold_masks.py --pipeline p0
  python src/export/transform_gold_masks.py --pipeline p2
  python src/export/transform_gold_masks.py --pipeline p0 --output-dir output/gold_masks_p0
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_export = Path(__file__).resolve().parent
_root   = _export.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_export / "p2"))

from ui_mask import parse_polygon

_P0_REGION_VERSION = "2025_11_25_crops"
_TARGET_SIZE = 256


# ---------------------------------------------------------------------------
# Spatial transforms — mirrors main.py exactly, mask-only
# ---------------------------------------------------------------------------

def _shrink_polygon(poly_str: str, factor: float = 0.97) -> str:
    pts = parse_polygon(poly_str)
    if len(pts) == 0:
        return poly_str
    centroid = pts.mean(axis=0)
    shrunk = centroid + factor * (pts - centroid)
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in shrunk)


def transform_mask_p0(
    mask: np.ndarray, row: dict, target_size: int = _TARGET_SIZE,
) -> np.ndarray | None:
    """Apply P0 crop + resize to a gold mask. Returns 256x256 single-channel."""
    mask_h, mask_w = mask.shape[:2]

    cx = max(0, int(row["p0_crop_x"]))
    cy = max(0, int(row["p0_crop_y"]))
    cw = min(int(row["p0_crop_w"]), mask_w - cx)
    ch = min(int(row["p0_crop_h"]), mask_h - cy)
    if cw <= 0 or ch <= 0:
        return None

    crop = mask[cy : cy + ch, cx : cx + cw]

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas


def transform_mask_p2(
    mask: np.ndarray, row: dict, target_size: int = _TARGET_SIZE,
) -> np.ndarray | None:
    """Apply P2 crop + resize to a gold mask. Mirrors preprocess_p2 crop logic."""
    mask_h, mask_w = mask.shape[:2]

    us_poly = row.get("us_polygon") or None
    if isinstance(us_poly, float):
        us_poly = None
    if us_poly:
        us_poly = _shrink_polygon(us_poly)
        pts = parse_polygon(us_poly)
        if len(pts) > 0:
            cx = max(0, int(np.floor(pts[:, 0].min())))
            cy = max(0, int(np.floor(pts[:, 1].min())))
            cw = min(mask_w - cx, int(np.ceil(pts[:, 0].max())) - cx)
            ch = min(mask_h - cy, int(np.ceil(pts[:, 1].max())) - cy)
        else:
            cx = max(0, int(row["crop_x"]))
            cy = max(0, int(row["crop_y"]))
            cw = min(int(row["crop_w"]), mask_w - cx)
            ch = min(int(row["crop_h"]), mask_h - cy)
    else:
        cx = max(0, int(row["crop_x"]))
        cy = max(0, int(row["crop_y"]))
        cw = min(int(row["crop_w"]), mask_w - cx)
        ch = min(int(row["crop_h"]), mask_h - cy)

    if cw <= 0 or ch <= 0:
        return None

    crop = mask[cy : cy + ch, cx : cx + cw]

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Per-mask worker
# ---------------------------------------------------------------------------

def process_one(
    row: dict,
    gold_masks_dir: Path,
    output_dir: Path,
    pipeline: str,
) -> tuple[str, bool, str]:
    """Read source mask, transform, write. Returns (image_name, success, err)."""
    image_name = row["image_name"]
    mask_file  = row["mask_image"]
    src_path   = gold_masks_dir / "masks" / mask_file
    try:
        mask = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {src_path}")

        if pipeline == "p0":
            result = transform_mask_p0(mask, row)
        else:
            result = transform_mask_p2(mask, row)

        if result is None:
            return (image_name, False, "transform returned None (bad crop coords)")

        cv2.imwrite(str(output_dir / "masks" / image_name), result)
        return (image_name, True, "")
    except Exception as exc:
        return (image_name, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transform gold lesion masks into P0/P2 256x256 space"
    )
    parser.add_argument("--pipeline", required=True, choices=["p0", "p2"])
    parser.add_argument("--db", default=str(_root / "data" / "cadbusi.db"))
    parser.add_argument(
        "--gold-masks-dir", type=Path,
        default=_export / "gold_masks",
        help="Directory containing the raw gold_masks/ with masks/ and manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: output/gold_masks_{pipeline})")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(f"output/gold_masks_{args.pipeline}")

    # --- Load gold masks manifest ---
    manifest_path = args.gold_masks_dir / "manifest.csv"
    print(f"Gold masks manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"  {len(df):,} gold mask entries")

    # --- Enrich with DB data needed for transforms ---
    conn = sqlite3.connect(args.db)

    if args.pipeline == "p0":
        # Need RegionLabels crop coords keyed by dicom_hash
        # First get dicom_hash for each image_name
        image_names = df["image_name"].unique().tolist()
        placeholders = ",".join("?" * len(image_names))
        img_df = pd.read_sql_query(
            f"SELECT image_name, dicom_hash FROM Images WHERE image_name IN ({placeholders})",
            conn, params=image_names,
        )
        df = df.merge(img_df, on="image_name", how="left")

        # Now get RegionLabels for those dicom_hashes
        hashes = df["dicom_hash"].dropna().unique().tolist()
        placeholders = ",".join("?" * len(hashes))
        rl = pd.read_sql_query(
            f"SELECT dicom_hash, crop_x AS p0_crop_x, crop_y AS p0_crop_y, "
            f"crop_w AS p0_crop_w, crop_h AS p0_crop_h "
            f"FROM RegionLabels WHERE version = ? AND dicom_hash IN ({placeholders})",
            conn, params=[_P0_REGION_VERSION] + hashes,
        )
        df = df.merge(rl, on="dicom_hash", how="left")

        missing = df["p0_crop_x"].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} masks have no RegionLabels entry — will be skipped")
            df = df[df["p0_crop_x"].notna()].copy()

    elif args.pipeline == "p2":
        # Need us_polygon and debris_polygons from Images table
        image_names = df["image_name"].unique().tolist()
        placeholders = ",".join("?" * len(image_names))
        img_df = pd.read_sql_query(
            f"SELECT image_name, us_polygon, debris_polygons, "
            f"crop_x, crop_y, crop_w, crop_h "
            f"FROM Images WHERE image_name IN ({placeholders})",
            conn, params=image_names,
        )
        # manifest already has crop_x etc. — use DB values to stay consistent
        df = df.drop(columns=["crop_x", "crop_y", "crop_w", "crop_h"], errors="ignore")
        df = df.merge(img_df, on="image_name", how="left")

    conn.close()
    print(f"  After enrichment: {len(df):,} masks to transform")

    if df.empty:
        print("Nothing to process.")
        return

    # --- Output dirs ---
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows = df.to_dict("records")

    # --- Process ---
    print(f"\nTransforming {len(rows):,} masks ({args.pipeline.upper()}) with {args.workers} workers ...")
    t0 = time.time()
    success = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, row, args.gold_masks_dir, args.output_dir, args.pipeline): row["image_name"]
            for row in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            name, ok, err = future.result()
            if ok:
                success += 1
            else:
                errors.append((name, err))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {success:,} ok, {len(errors):,} errors")
    if errors:
        print("First 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # --- Manifest ---
    masks_dir = args.output_dir / "masks"
    out_rows = [
        {**row, "mask_path": f"masks/{row['image_name']}"}
        for row in rows
        if (masks_dir / row["image_name"]).exists()
    ]
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"  Manifest: {args.output_dir / 'manifest.csv'} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
