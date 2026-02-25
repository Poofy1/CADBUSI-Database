#!/usr/bin/env python3
"""Unified two-pipeline export: P0 and P2 in a single pass.

Each image is downloaded from GCP exactly once and processed by both
pipelines before the next image is fetched — maximising throughput at scale.

Pipeline summary
----------------
P0  Baseline          — crop + top-left-aligned resize, no masking
P2  Structural        — axis-align + tighten + FOV-clip, then FOV mask + resize
                        + binary tissue mask + 16×16 patch tissue count map

Outputs
-------
{output_dir}/
  P0/images/{name}               P0 processed image
  P0/manifest.csv
  P2/images/{name}               P2 processed image
  P2/masks/{name}                binary tissue mask  (255=tissue)
  P2/patch_tissue/{name}         16×16 patch tissue count map
  P2/manifest.csv

Usage
-----
  python main.py
  python main.py --dry-run
  python main.py --limit 100 --output-dir output/test_all
  python main.py --workers 24 --resume
"""

import argparse
import sqlite3
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_export = Path(__file__).resolve().parent          # src/export/
_root   = _export.parent.parent                    # repo root
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_export / "p2"))
sys.path.insert(0, str(_export / "config_processing"))

from config import CONFIG
from tools.storage_adapter import StorageClient
from export_configurable import ExportConfig
from pipeline_common import (
    build_query, load_from_db, apply_image_filters,
    download_bytes, resolve_output_dir,
)
from ui_mask import compute_ui_mask
from align_polygon_axes import align_polygon
from tighten_crops import tighten_crop_intensity
from tighten_crops_fov import clip_crop_to_fov

_P0_REGION_VERSION = "2026_1_13_crops"


# ---------------------------------------------------------------------------
# RegionLabels loader (P0 crop coords)
# ---------------------------------------------------------------------------

def load_region_labels(db_path: str) -> pd.DataFrame:
    """Return crop coords from RegionLabels for version='2026_1_13_crops'.

    Columns returned: dicom_hash, p0_crop_x, p0_crop_y, p0_crop_w, p0_crop_h
    """
    conn = sqlite3.connect(db_path)
    rl = pd.read_sql_query(
        f"SELECT dicom_hash, crop_x, crop_y, crop_w, crop_h "
        f"FROM RegionLabels WHERE version = '{_P0_REGION_VERSION}'",
        conn,
    )
    conn.close()
    rl = rl.rename(columns={
        "crop_x": "p0_crop_x",
        "crop_y": "p0_crop_y",
        "crop_w": "p0_crop_w",
        "crop_h": "p0_crop_h",
    })
    print(f"  RegionLabels ({_P0_REGION_VERSION}): {len(rl):,} entries")
    return rl


# ---------------------------------------------------------------------------
# P0 — baseline crop + resize
# ---------------------------------------------------------------------------

def preprocess_p0(img: np.ndarray, row: dict, target_size: int, fill: int) -> np.ndarray:
    """Simple crop + top-left-aligned resize. No masking of any kind."""
    cx = max(0, int(row["crop_x"]))
    cy = max(0, int(row["crop_y"]))
    cw = min(int(row["crop_w"]), img.shape[1] - cx)
    ch = min(int(row["crop_h"]), img.shape[0] - cy)
    img_crop = img[cy : cy + ch, cx : cx + cw]

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas[:new_h, :new_w] = img_resized
    return canvas


# ---------------------------------------------------------------------------
# P2 — structural tissue-aware (stages 7a-c + 8)
# ---------------------------------------------------------------------------

def preprocess_p2(
    img: np.ndarray,
    row: dict,
    target_size: int,
    fill: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply P2 structural refinement + tissue masking.

    Returns (canvas_img, canvas_mask, patch_counts).
    The row dict is not modified (a local copy is used for crop adjustments).
    """
    row = dict(row)   # mutable local copy — do NOT mutate the original
    img_h, img_w = img.shape[:2]

    # Stage 7a: snap near-axis polygon edges to perfect H/V alignment
    us_poly = row.get("us_polygon") or ""
    if us_poly and not isinstance(us_poly, float):
        aligned, _, _, _ = align_polygon(us_poly, img_shape=(img_h, img_w))
        row["us_polygon"] = aligned
    else:
        us_poly = ""
        row["us_polygon"] = ""

    # Stage 7b: trim dark empty rows from crop top/bottom via intensity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fov_mask = compute_ui_mask(
        row.get("us_polygon"), row.get("debris_polygons"), img_h, img_w
    )
    new_y, new_h = tighten_crop_intensity(
        gray, fov_mask, int(row["crop_y"]), int(row["crop_h"])
    )
    row["crop_y"], row["crop_h"] = new_y, new_h

    # Stage 7c: clip crop box to FOV polygon extent (geometric, no image needed)
    row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"] = clip_crop_to_fov(
        row.get("us_polygon"), row.get("debris_polygons"),
        img_h, img_w,
        row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"],
    )

    # Stage 8: crop + mask + fill non-tissue + resize + 16×16 patch map
    debris_str = row.get("debris_polygons")
    if isinstance(debris_str, float):
        debris_str = None

    mask_full = compute_ui_mask(row.get("us_polygon"), debris_str, img_h, img_w)

    cx = max(0, int(row["crop_x"]))
    cy = max(0, int(row["crop_y"]))
    cw = min(int(row["crop_w"]), img_w - cx)
    ch = min(int(row["crop_h"]), img_h - cy)

    img_crop  = img[cy : cy + ch, cx : cx + cw].copy()
    mask_crop = mask_full[cy : cy + ch, cx : cx + cw]
    img_crop[mask_crop == 0] = fill

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized  = cv2.resize(img_crop,  (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img  = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas_img[:new_h, :new_w]  = img_resized
    canvas_mask[:new_h, :new_w] = mask_resized

    ps = target_size // 16
    patch_counts = np.zeros((16, 16), dtype=np.uint8)
    for py in range(16):
        for px in range(16):
            patch = canvas_mask[py * ps : (py + 1) * ps, px * ps : (px + 1) * ps]
            patch_counts[py, px] = min(int(np.count_nonzero(patch)), 255)

    return canvas_img, canvas_mask, patch_counts


# ---------------------------------------------------------------------------
# Per-image worker — download once, write all three pipelines
# ---------------------------------------------------------------------------

def process_one(
    row: dict,
    image_dir: str,
    dirs: dict[str, Path],
    target_size: int,
    fill: int,
    storage: StorageClient,
) -> tuple[str, bool, str]:
    """Download + process for P0 and P2 in one shot.

    dirs  — {"p0": Path, "p2": Path}

    Returns (image_name, success, error_msg).
    """
    name = row["image_name"]
    try:
        # --- single GCP download ---
        img_bytes = download_bytes(name, image_dir, storage)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")

        # P0 — only for images that have a RegionLabels entry
        p0_cx = row.get("p0_crop_x")
        if p0_cx is not None and not (isinstance(p0_cx, float) and p0_cx != p0_cx):
            row_p0 = {**row,
                      "crop_x": row["p0_crop_x"], "crop_y": row["p0_crop_y"],
                      "crop_w": row["p0_crop_w"], "crop_h": row["p0_crop_h"]}
            p0 = preprocess_p0(img, row_p0, target_size, fill)
            cv2.imwrite(str(dirs["p0"] / "images" / name), p0)

        # P2 (takes an internal copy of row; original unchanged)
        p2_img, p2_mask, p2_patch = preprocess_p2(img, row, target_size, fill)
        cv2.imwrite(str(dirs["p2"] / "images"       / name), p2_img)
        cv2.imwrite(str(dirs["p2"] / "masks"        / name), p2_mask)
        cv2.imwrite(str(dirs["p2"] / "patch_tissue" / name), p2_patch)

        return (name, True, "")
    except Exception as exc:
        return (name, False, str(exc))


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------

_SPLIT_MAP = {0: "train", 1: "valid", 2: "test"}


def write_manifest(df: pd.DataFrame, out_dir: Path, extra_cols: dict[str, str]) -> None:
    """Write manifest.csv; only includes images that actually exist on disk."""
    img_dir = out_dir / "images"
    rows = []
    for row in df.to_dict("records"):
        if not (img_dir / row["image_name"]).exists():
            continue
        entry = dict(row)
        entry["label"] = int(row["has_malignant"])
        entry["split"] = _SPLIT_MAP.get(
            int(row["valid"]) if row.get("valid") is not None else 0, "train"
        )
        entry["image_path"] = f"images/{row['image_name']}"
        for col, val_tmpl in extra_cols.items():
            entry[col] = val_tmpl.format(name=row["image_name"])
        rows.append(entry)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"  {out_dir.name}/manifest.csv  ({len(rows):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified P0/P2 export — reads each image from GCP exactly once"
    )
    parser.add_argument("--db", default=str(_root / "data" / "cadbusi.db"))
    parser.add_argument(
        "--dataset",
        default=str(_export / "configs" / "P2.yaml"),
        help="YAML config (P2 config used as superset filter; default: configs/P2.yaml)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/all_pipelines"))
    parser.add_argument("--workers",    type=int, default=16)
    parser.add_argument("--limit",      type=int, default=0, help="Process first N images only (debug)")
    parser.add_argument("--resume",     action="store_true", help="Skip images already in P0/images/")
    parser.add_argument("--dry-run",    action="store_true", help="Print stats and exit")
    args = parser.parse_args()

    # Storage
    image_dir = CONFIG.get("DATABASE_DIR", "").rstrip("/\\") + "/images"
    storage = StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", ""),
    )
    print(f"Storage mode : {'GCP' if storage.is_gcp else 'local'}")
    print(f"Image source : {image_dir}")

    # Config (P2 yaml is the superset — same filters, plus polygon/debris columns)
    config = ExportConfig.from_yaml(Path(args.dataset))
    target_size = config.preprocessing.target_size
    fill        = config.preprocessing.fill
    print(f"Config : {config.name} — target_size={target_size}, fill={fill}")

    # Load + filter
    print(f"\nLoading DB: {args.db}")
    df = load_from_db(args.db, build_query(config))
    df = apply_image_filters(df, config)

    # Attach P0 crop coords from RegionLabels (LEFT JOIN — unmatched rows get NaN)
    print("\nLoading RegionLabels ...")
    rl = load_region_labels(args.db)
    df = df.merge(rl, on="dicom_hash", how="left")
    n_p0 = df["p0_crop_x"].notna().sum()
    print(f"  P0 coverage: {n_p0:,}/{len(df):,} images have RegionLabels crop coords")

    if args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"  After --limit: {len(df):,}")

    if df.empty:
        print("No images to process — exiting.")
        return

    if args.dry_run:
        sample = df.head(1_000).copy()
        print(f"\n[DRY RUN] {len(sample):,} of {len(df):,} images")
        print(f"  Output  : {args.output_dir}")
        print(f"  Size    : {target_size}px, fill={fill}")
        print(f"  Workers : {args.workers}")
        if "has_malignant" in df.columns:
            print(f"  has_malignant:\n{df['has_malignant'].value_counts(dropna=False).to_string()}")
        n_poly    = df["us_polygon"].notna().sum() if "us_polygon" in df.columns else 0
        n_debris  = df["debris_polygons"].notna().sum() if "debris_polygons" in df.columns else 0
        n_p0_     = df["p0_crop_x"].notna().sum() if "p0_crop_x" in df.columns else 0
        print(f"  us_polygon coverage   : {n_poly:,}/{len(df):,}")
        print(f"  debris_polygon rows   : {n_debris:,}/{len(df):,}")
        print(f"  P0 RegionLabels match : {n_p0_:,}/{len(df):,}")
        return

    # Output dirs
    base = resolve_output_dir(args.output_dir, args.resume)
    dirs: dict[str, Path] = {
        "p0": base / "P0",
        "p2": base / "P2",
    }
    for d in dirs.values():
        (d / "images").mkdir(parents=True, exist_ok=True)
    for sub in ("masks", "patch_tissue"):
        (dirs["p2"] / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.dataset, base / "export_config.yaml")

    rows = df.to_dict("records")

    # Resume: use P2/images/ as reference (every image gets P2; P0 is a subset)
    if args.resume:
        existing = {p.name for p in (dirs["p2"] / "images").iterdir()}
        before = len(rows)
        rows = [r for r in rows if r["image_name"] not in existing]
        print(f"\nResume: skipped {before - len(rows):,}, {len(rows):,} remaining")

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
                process_one, row, image_dir, dirs, target_size, fill, storage
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
                rate    = n / elapsed
                eta     = (len(rows) - n) / rate if rate > 0 else 0
                print(
                    f"  [{n:,}/{len(rows):,}]  {rate:.1f} img/s"
                    f"  |  elapsed {elapsed:.0f}s  |  ETA {eta:.0f}s"
                    f"  |  errors: {len(errors)}"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {success:,} ok, {len(errors):,} errors")
    if errors:
        print("\nFirst 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # Manifests
    print("\nWriting manifests ...")
    write_manifest(df, dirs["p0"], {})
    write_manifest(df, dirs["p2"], {
        "mask_path":         "masks/{name}",
        "patch_tissue_path": "patch_tissue/{name}",
    })


if __name__ == "__main__":
    main()
