#!/usr/bin/env python3
"""P0 preprocessing pipeline: simple crop + center letterbox.

No masks, no patch maps. Fastest baseline preprocessing.

Outputs:
  {output_dir}/images/{image_name}.png   — 256px center-letterboxed RGB
  {output_dir}/manifest.csv             — metadata manifest
  {output_dir}/export_config.yaml       — copy of config for reproducibility

Usage:
  python main.py
  python main.py --dataset ../configs/P0.yaml --output-dir ./output/P0
  python main.py --workers 24 --resume
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Set up sys.path: repo root, src/export/, src/export/config_processing/
_export = Path(__file__).resolve().parent.parent
_root = _export.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_export / "config_processing"))

from config import CONFIG
from tools.storage_adapter import StorageClient
from export_configurable import ExportConfig
from pipeline_common import build_query, load_from_db, apply_image_filters, download_bytes, resolve_output_dir


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def preprocess_p0(img: np.ndarray, row: dict, target_size: int, fill: int) -> np.ndarray:
    """Simple crop + top-left aligned resize. No polygons, no masks, no patch maps."""
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
    canvas[:new_h, :new_w] = img_resized
    return canvas


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------

def process_one(
    row: dict,
    image_dir: str,
    output_dir: Path,
    target_size: int,
    fill: int,
    storage: StorageClient,
) -> tuple[str, bool, str]:
    """Download + preprocess + save one image. Returns (image_name, success, error_msg)."""
    name = row["image_name"]
    try:
        img_bytes = download_bytes(name, image_dir, storage)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")

        canvas = preprocess_p0(img, row, target_size, fill)
        cv2.imwrite(str(output_dir / "images" / name), canvas)
        return (name, True, "")
    except Exception as exc:
        return (name, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P0 pipeline: simple crop + center letterbox"
    )
    parser.add_argument(
        "--db",
        default=str(_root / "data" / "cadbusi.db"),
        help="Path to cadbusi.db (default: <repo-root>/data/cadbusi.db)",
    )
    parser.add_argument(
        "--dataset",
        default=str(_export / "configs" / "P0.yaml"),
        help="Path to dataset YAML config (default: configs/P0.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/P0"),
        help="Output directory (default: output/P0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel download/process threads (default: 16)",
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

    # Storage
    image_dir = CONFIG.get("DATABASE_DIR", "").rstrip("/\\") + "/images"
    storage = StorageClient.get_instance(
        windir=CONFIG.get("WINDIR", ""),
        bucket_name=CONFIG.get("BUCKET", ""),
    )
    print(f"Storage mode : {'GCP' if storage.is_gcp else 'local'}")
    print(f"Image source : {image_dir}")

    # Config
    config = ExportConfig.from_yaml(Path(args.dataset))
    target_size = config.preprocessing.target_size
    fill = config.preprocessing.fill
    print(f"Dataset : {config.name}")
    print(f"          {config.description}")
    print(f"  target_size={target_size}, fill={fill}")

    # Load + filter
    print(f"\nLoading DB: {args.db}")
    df = load_from_db(args.db, build_query(config))
    df = apply_image_filters(df, config)

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
        df = sample

    # Output dirs
    output_dir = resolve_output_dir(args.output_dir, args.resume)
    if output_dir != args.output_dir:
        print(f"  Output dir   : {output_dir}  (auto-incremented)")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.dataset, output_dir / "export_config.yaml")

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
            executor.submit(process_one, row, image_dir, output_dir, target_size, fill, storage): row["image_name"]
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
                print(f"  [{n:,}/{len(rows):,}]  {rate:.1f} img/s  |  elapsed {elapsed:.0f}s  |  ETA {eta:.0f}s  |  errors: {len(errors)}")

    print(f"\nDone in {time.time() - t0:.1f}s — {success:,} ok, {len(errors):,} errors")
    if errors:
        print("\nFirst 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # Manifest
    print("\nWriting manifest ...")
    img_dir = output_dir / "images"
    _split_map = {0: "train", 1: "valid", 2: "test"}
    manifest_rows = [
        {
            **row,
            "label": int(row["has_malignant"]),
            "split": _split_map.get(int(row["valid"]) if row.get("valid") is not None else 0, "train"),
            "image_path": f"images/{row['image_name']}",
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
