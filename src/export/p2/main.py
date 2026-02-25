#!/usr/bin/env python3
"""P2 preprocessing pipeline: polygon mask + crop + fill + patch map.

Outputs:
  {output_dir}/images/{image_name}.png        — 256px top-left aligned RGB (gray fill)
  {output_dir}/masks/{image_name}.png         — 256px tissue mask (255=tissue)
  {output_dir}/patch_tissue/{image_name}.png  — 16x16 patch tissue count map
  {output_dir}/manifest.csv                   — metadata manifest
  {output_dir}/export_config.yaml             — copy of config for reproducibility

Usage:
  python main.py
  python main.py --dataset ../configs/P2.yaml --output-dir ./output/P2
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

# Set up sys.path: repo root, src/export/, src/export/p2/, src/export/config_processing/
_p2 = Path(__file__).resolve().parent
_export = _p2.parent
_root = _export.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_p2))
sys.path.insert(0, str(_export / "config_processing"))

from config import CONFIG
from tools.storage_adapter import StorageClient
from ui_mask import compute_ui_mask, parse_polygon
from export_configurable import ExportConfig
from pipeline_common import build_query, load_from_db, apply_image_filters, download_bytes, resolve_output_dir
from align_polygon_axes import align_polygon
from tighten_crops import tighten_crop_intensity
from tighten_crops_fov import clip_crop_to_fov


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray, row: dict, target_size: int, fill: int):
    """Crop, mask, resize one image.

    Returns (canvas_img, canvas_mask, patch_counts).
    """
    img_h, img_w = img.shape[:2]

    us_poly = row.get("us_polygon") or None
    debris_poly = row.get("debris_polygons") or None
    if isinstance(us_poly, float):
        us_poly = None
    if isinstance(debris_poly, float):
        debris_poly = None
    mask_full = compute_ui_mask(us_poly, debris_poly, img_h, img_w)

    cx = max(0, int(row["crop_x"]))
    cy = max(0, int(row["crop_y"]))
    cw = min(int(row["crop_w"]), img_w - cx)
    ch = min(int(row["crop_h"]), img_h - cy)

    img_crop = img[cy : cy + ch, cx : cx + cw].copy()
    mask_crop = mask_full[cy : cy + ch, cx : cx + cw]

    img_crop[mask_crop == 0] = fill

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas_img[:new_h, :new_w] = img_resized
    canvas_mask[:new_h, :new_w] = mask_resized

    ps = target_size // 16
    patch_counts = np.zeros((16, 16), dtype=np.uint8)
    for py in range(16):
        for px in range(16):
            patch = canvas_mask[py * ps : (py + 1) * ps, px * ps : (px + 1) * ps]
            patch_counts[py, px] = min(int(np.count_nonzero(patch)), 255)

    return canvas_img, canvas_mask, patch_counts


# ---------------------------------------------------------------------------
# Stage visualization
# ---------------------------------------------------------------------------

_VIS_PANEL_H = 300   # pixel height of each panel in the montage
_CROP_COLOR  = (0, 220, 0)    # green  — crop box
_POLY_COLOR  = (0, 100, 255)  # orange — us_polygon


def _make_panel(img: np.ndarray, cx: int, cy: int, cw: int, ch: int,
                poly_str: str | None, label: str) -> np.ndarray:
    """Draw crop box + polygon on a scaled-down copy of the image."""
    vis = img.copy()

    # Polygon (orange)
    if poly_str and not isinstance(poly_str, float):
        poly = parse_polygon(poly_str)
        if len(poly) >= 3:
            cv2.polylines(vis, [poly.astype(np.int32).reshape(-1, 1, 2)],
                          True, _POLY_COLOR, 2)

    # Crop box (green)
    x1, y1 = max(0, int(cx)), max(0, int(cy))
    x2, y2 = x1 + max(1, int(cw)), y1 + max(1, int(ch))
    cv2.rectangle(vis, (x1, y1), (x2, y2), _CROP_COLOR, 2)

    # Scale to fixed height
    h, w = vis.shape[:2]
    scale = _VIS_PANEL_H / h
    vis = cv2.resize(vis, (max(1, round(w * scale)), _VIS_PANEL_H))

    # Label bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def save_stage_visualization(
    img: np.ndarray,
    final_img: np.ndarray,
    row_orig: dict,
    row_7a: dict,
    row_7b: dict,
    row_7c: dict,
    name: str,
    vis_dir: Path,
) -> None:
    """Save a 5-panel montage: Original | 7a | 7b | 7c | P2 output.

    Green rectangle = crop box at that stage.
    Orange outline  = us_polygon at that stage (7a shows the aligned version
                      in all subsequent panels so the polygon change is visible).
    """
    def _r(row, key, fallback=0):
        return row.get(key, fallback) or fallback

    orig_poly    = row_orig.get("us_polygon")
    aligned_poly = row_7a.get("us_polygon")

    panels = [
        _make_panel(img,
                    _r(row_orig, "crop_x"), _r(row_orig, "crop_y"),
                    _r(row_orig, "crop_w"), _r(row_orig, "crop_h"),
                    orig_poly, "Original"),
        _make_panel(img,
                    _r(row_orig, "crop_x"), _r(row_orig, "crop_y"),
                    _r(row_orig, "crop_w"), _r(row_orig, "crop_h"),
                    aligned_poly, "7a: Axis align"),
        _make_panel(img,
                    _r(row_7b, "crop_x"), _r(row_7b, "crop_y"),
                    _r(row_7b, "crop_w"), _r(row_7b, "crop_h"),
                    aligned_poly, "7b: Intensity tighten"),
        _make_panel(img,
                    _r(row_7c, "crop_x"), _r(row_7c, "crop_y"),
                    _r(row_7c, "crop_w"), _r(row_7c, "crop_h"),
                    aligned_poly, "7c: FOV clip"),
    ]

    # Final P2 output panel — upscale 256px to match panel height
    out_h, out_w = final_img.shape[:2]
    scale = _VIS_PANEL_H / out_h
    out_panel = cv2.resize(final_img, (max(1, round(out_w * scale)), _VIS_PANEL_H))
    cv2.rectangle(out_panel, (0, 0), (out_panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out_panel, "P2 output", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    panels.append(out_panel)

    cv2.imwrite(str(vis_dir / name), np.hstack(panels))


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
    vis_dir: Path | None = None,
) -> tuple[str, bool, str]:
    """Download + Stage 7 refinement + Stage 8 preprocessing + save one image.

    Returns (image_name, success, error_msg).
    """
    name = row["image_name"]
    try:
        img_bytes = download_bytes(name, image_dir, storage)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")

        img_h, img_w = img.shape[:2]
        row = dict(row)       # mutable copy
        row_orig = dict(row)  # snapshot before any stage 7 changes

        # Stage 7a: snap near-axis polygon edges to perfect alignment (geometric)
        us_poly = row.get("us_polygon") or ""
        if us_poly and not isinstance(us_poly, float):
            aligned, _, _, _ = align_polygon(us_poly, img_shape=(img_h, img_w))
            row["us_polygon"] = aligned
        row_7a = dict(row)

        # Stage 7b: trim dark empty rows from crop top/bottom via intensity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fov_mask = compute_ui_mask(row.get("us_polygon"), row.get("debris_polygons"), img_h, img_w)
        new_y, new_h = tighten_crop_intensity(gray, fov_mask, int(row["crop_y"]), int(row["crop_h"]))
        row["crop_y"], row["crop_h"] = new_y, new_h
        row_7b = dict(row)

        # Stage 7c: clip crop box to FOV polygon boundary (geometric)
        row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"] = clip_crop_to_fov(
            row.get("us_polygon"), row.get("debris_polygons"),
            img_h, img_w,
            row["crop_x"], row["crop_y"], row["crop_w"], row["crop_h"],
        )
        row_7c = dict(row)

        # Stage 8: crop + mask + fill + resize + 16x16 patch map
        canvas_img, canvas_mask, patch_counts = preprocess_image(img, row, target_size, fill)
        cv2.imwrite(str(output_dir / "images" / name), canvas_img)
        cv2.imwrite(str(output_dir / "masks" / name), canvas_mask)
        cv2.imwrite(str(output_dir / "patch_tissue" / name), patch_counts)

        if vis_dir is not None:
            save_stage_visualization(img, canvas_img, row_orig, row_7a, row_7b, row_7c, name, vis_dir)

        return (name, True, "")
    except Exception as exc:
        return (name, False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P2 pipeline: polygon mask + crop + fill + patch map"
    )
    parser.add_argument(
        "--db",
        default=str(_root / "data" / "cadbusi.db"),
        help="Path to cadbusi.db (default: <repo-root>/data/cadbusi.db)",
    )
    parser.add_argument(
        "--dataset",
        default=str(_export / "configs" / "P2.yaml"),
        help="Path to dataset YAML config (default: configs/P2.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/P2"),
        help="Output directory (default: output/P2)",
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
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="Save per-image stage montages (Original|7a|7b|7c|P2) to this directory",
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
        n_poly = df["us_polygon"].notna().sum() if "us_polygon" in df.columns else 0
        print(f"  us_polygon coverage: {n_poly:,}/{len(df):,}")
        df = sample

    # Output dirs
    output_dir = resolve_output_dir(args.output_dir, args.resume)
    if output_dir != args.output_dir:
        print(f"  Output dir   : {output_dir}  (auto-incremented)")
    for sub in ["images", "masks", "patch_tissue"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.dataset, output_dir / "export_config.yaml")

    vis_dir = args.vis_dir
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Visualizations: {vis_dir}")

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
            executor.submit(process_one, row, image_dir, output_dir, target_size, fill, storage, vis_dir): row["image_name"]
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
