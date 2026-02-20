#!/usr/bin/env python3
"""Build v7 preprocessed dataset: 256px, masked, patch-aligned.

Pipeline per image:
  1. Load raw PNG
  2. Rasterize inclusion mask (FOV minus debris)
  3. Crop image + mask to optimized crop box
  4. Fill non-tissue pixels with gray (or configurable fill)
  5. Resize to fit target_size (aspect-preserving)
  6. Place on canvas top-left aligned (asymmetric letterbox)
  7. Compute 16x16 patch tissue count map
  8. Save image, mask, patch map

Outputs:
  images/{image_name}.png   — 256x256 RGB (gray fill)
  masks/{image_name}.png    — 256x256 grayscale (255=tissue, 0=non-tissue)
  patch_tissue/{image_name}.png — 16x16 grayscale (pixel = tissue count per patch)
  ImageData_v7_labeled.csv  — manifest with paths
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_mask import compute_ui_mask


def preprocess_one(row_dict, image_dir, target_size=256, fill=128):
    """Process a single image. Returns (canvas_img, canvas_mask, patch_counts).

    Args:
        row_dict: Dict with image_name, crop_x/y/w/h, rows, columns,
                  us_polygon, debris_polygons.
        image_dir: Path to raw images.
        target_size: Output canvas size (default 256).
        fill: Fill value for non-tissue pixels (default 128).

    Returns:
        Tuple of (canvas_img, canvas_mask, patch_counts) or raises on error.
    """
    image_path = image_dir / row_dict["image_name"]
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    img_h, img_w = img.shape[:2]

    # Rasterize inclusion mask at raw resolution
    us_poly = row_dict.get("us_polygon") or None
    debris_poly = row_dict.get("debris_polygons") or None
    # Handle pandas NaN
    if isinstance(us_poly, float):
        us_poly = None
    if isinstance(debris_poly, float):
        debris_poly = None
    mask_full = compute_ui_mask(us_poly, debris_poly, img_h, img_w)

    # Crop
    cx = int(row_dict["crop_x"])
    cy = int(row_dict["crop_y"])
    cw = int(row_dict["crop_w"])
    ch = int(row_dict["crop_h"])
    # Clamp to image bounds
    cx = max(0, cx)
    cy = max(0, cy)
    cw = min(cw, img_w - cx)
    ch = min(ch, img_h - cy)

    img_crop = img[cy : cy + ch, cx : cx + cw]
    mask_crop = mask_full[cy : cy + ch, cx : cx + cw]

    # Apply mask — fill non-tissue pixels
    img_crop[mask_crop == 0] = fill

    # Scale to fit target_size (aspect-preserving)
    scale = min(target_size / cw, target_size / ch)
    new_w = round(cw * scale)
    new_h = round(ch * scale)
    # Clamp to target_size
    new_w = min(new_w, target_size)
    new_h = min(new_h, target_size)

    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Place on canvas (top-left aligned)
    canvas_img = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas_img[:new_h, :new_w] = img_resized
    canvas_mask[:new_h, :new_w] = mask_resized

    # Patch tissue counts (16x16 grid of 16x16 patches for 256px)
    patch_size = target_size // 16
    patch_counts = np.zeros((16, 16), dtype=np.uint8)
    for py in range(16):
        for px in range(16):
            patch = canvas_mask[
                py * patch_size : (py + 1) * patch_size,
                px * patch_size : (px + 1) * patch_size,
            ]
            count = int(np.count_nonzero(patch))
            patch_counts[py, px] = min(count, 255)

    return canvas_img, canvas_mask, patch_counts


def _worker(args):
    """Multiprocessing worker. Returns (image_name, success, error_msg)."""
    row_dict, image_dir, output_dir, target_size, fill = args
    image_name = row_dict["image_name"]
    try:
        canvas_img, canvas_mask, patch_counts = preprocess_one(
            row_dict, image_dir, target_size, fill
        )

        img_path = output_dir / "images" / image_name
        mask_path = output_dir / "masks" / image_name
        patch_path = output_dir / "patch_tissue" / image_name

        cv2.imwrite(str(img_path), canvas_img)
        cv2.imwrite(str(mask_path), canvas_mask)
        cv2.imwrite(str(patch_path), patch_counts)

        return (image_name, True, "")
    except Exception as e:
        return (image_name, False, str(e))


def make_preview_composite(
    row_dict, image_dir, target_size=256, fill=128, upscale_patch=True
):
    """Create a 4-panel preview composite for one image.

    Panels: [Raw + crop box] [Preprocessed] [Mask] [Patch tissue map]
    Returns a single BGR image.
    """
    image_path = image_dir / row_dict["image_name"]
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Panel 1: Raw with crop box overlay
    panel1 = raw.copy()
    cx = int(row_dict["crop_x"])
    cy = int(row_dict["crop_y"])
    cw = int(row_dict["crop_w"])
    ch = int(row_dict["crop_h"])
    cv2.rectangle(panel1, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

    # Resize panel1 to target_size for uniform layout
    p1_h, p1_w = panel1.shape[:2]
    p1_scale = target_size / max(p1_h, p1_w)
    p1_new_w = round(p1_w * p1_scale)
    p1_new_h = round(p1_h * p1_scale)
    panel1_resized = cv2.resize(panel1, (p1_new_w, p1_new_h))
    panel1_canvas = np.full((target_size, target_size, 3), 200, dtype=np.uint8)
    panel1_canvas[:p1_new_h, :p1_new_w] = panel1_resized

    # Panels 2-4: preprocessed outputs
    canvas_img, canvas_mask, patch_counts = preprocess_one(
        row_dict, image_dir, target_size, fill
    )

    # Panel 2: preprocessed image
    panel2 = canvas_img

    # Panel 3: mask as 3-channel for display
    panel3 = cv2.cvtColor(canvas_mask, cv2.COLOR_GRAY2BGR)

    # Panel 4: patch tissue map upscaled with grid + counts
    if upscale_patch:
        patch_vis = cv2.resize(
            patch_counts, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )
        panel4 = cv2.cvtColor(patch_vis, cv2.COLOR_GRAY2BGR)
        # Draw grid lines
        ps = target_size // 16
        for i in range(1, 16):
            cv2.line(panel4, (i * ps, 0), (i * ps, target_size), (0, 100, 0), 1)
            cv2.line(panel4, (0, i * ps), (target_size, i * ps), (0, 100, 0), 1)
        # Overlay counts
        for py in range(16):
            for px in range(16):
                val = int(patch_counts[py, px])
                if val > 0:
                    txt = str(val)
                    tx = px * ps + 2
                    ty = py * ps + ps - 3
                    cv2.putText(
                        panel4,
                        txt,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (0, 255, 0),
                        1,
                    )
    else:
        panel4 = cv2.cvtColor(
            cv2.resize(
                patch_counts, (target_size, target_size), interpolation=cv2.INTER_NEAREST
            ),
            cv2.COLOR_GRAY2BGR,
        )

    # Labels
    label_h = 20
    labels = ["Raw + Crop Box", "Preprocessed", "Inclusion Mask", "Patch Tissue Map"]
    panels = [panel1_canvas, panel2, panel3, panel4]

    # Add label bar on top of each panel
    labeled_panels = []
    for label, panel in zip(labels, panels):
        bar = np.full((label_h, target_size, 3), 40, dtype=np.uint8)
        cv2.putText(bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        labeled_panels.append(np.vstack([bar, panel]))

    composite = np.hstack(labeled_panels)
    return composite


def run_preview(df, image_dir, preview_dir, preview_n, target_size, fill):
    """Generate preview composites: N random labeled images per scanner."""
    preview_dir = Path(preview_dir)

    scanner_col = "manufacturer_model_name"
    scanners = df[scanner_col].unique()
    print(f"Generating preview: {preview_n} images per scanner, {len(scanners)} scanners")

    total = 0
    for scanner in sorted(scanners):
        scanner_df = df[df[scanner_col] == scanner]
        n = min(preview_n, len(scanner_df))
        sample = scanner_df.sample(n=n, random_state=42)

        # Sanitize scanner name for directory
        scanner_dir_name = scanner.replace(" ", "_").replace("/", "_")
        scanner_dir = preview_dir / scanner_dir_name
        scanner_dir.mkdir(parents=True, exist_ok=True)

        for idx, (_, row) in enumerate(sample.iterrows()):
            row_dict = row.to_dict()
            try:
                composite = make_preview_composite(
                    row_dict, image_dir, target_size, fill
                )
                out_path = scanner_dir / f"{idx:03d}_{row_dict['image_name']}"
                cv2.imwrite(str(out_path), composite)
                total += 1
            except Exception as e:
                print(f"  SKIP {row_dict['image_name']}: {e}")

        print(f"  {scanner}: {n} composites → {scanner_dir}")

    print(f"Preview complete: {total} composites in {preview_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build v7 preprocessed dataset (256px, masked, patch-aligned)"
    )
    parser.add_argument(
        "--imagedata",
        type=Path,
        default=Path("data/splits/v7/v7_dataset/ImageData_v6_mbox.csv"),
        help="Input CSV with crop boxes and polygons",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/mnt/e/mayo_dataset/export_01_22_2026_00_24_52/images"),
        help="Raw image directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed/mayo_v7_256_masked"),
        help="Output directory for preprocessed dataset",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Only process images with has_malignant in [0, 1]",
    )
    parser.add_argument("--fill", type=int, default=128, help="Fill value (default: 128)")
    parser.add_argument(
        "--target-size", type=int, default=256, help="Target image size (default: 256)"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip images that already exist"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N images (0=all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report stats without writing"
    )
    # Preview mode
    parser.add_argument(
        "--preview", action="store_true", help="Generate preview composites only"
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("/tmp/v7_preview"),
        help="Directory for preview composites",
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=50,
        help="Number of images per scanner for preview",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.imagedata} ...")
    df = pd.read_csv(args.imagedata, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    if args.labeled_only:
        df = df[df["has_malignant"].isin([0, 1])].copy()
        print(f"  After labeled filter: {len(df):,}")

    if args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"  After limit: {len(df):,}")

    # Preview mode
    if args.preview:
        run_preview(
            df, args.image_dir, args.preview_dir, args.preview_n,
            args.target_size, args.fill,
        )
        return

    # Dry run
    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(df):,} images")
        print(f"  Output: {args.output_dir}")
        print(f"  Target size: {args.target_size}")
        print(f"  Fill: {args.fill}")
        print(f"  Workers: {args.workers}")
        # Compute aspect ratio stats
        ar = df["crop_w"] / df["crop_h"]
        print(f"  Aspect ratio: mean={ar.mean():.2f}, min={ar.min():.2f}, max={ar.max():.2f}")
        return

    # Create output directories
    output_dir = args.output_dir
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "patch_tissue").mkdir(parents=True, exist_ok=True)

    # Build work list
    rows = df.to_dict("records")
    if args.resume:
        existing = set(p.name for p in (output_dir / "images").iterdir())
        before = len(rows)
        rows = [r for r in rows if r["image_name"] not in existing]
        print(f"  Resume: skipping {before - len(rows):,} existing, {len(rows):,} remaining")

    if not rows:
        print("Nothing to process.")
        return

    print(f"\nProcessing {len(rows):,} images with {args.workers} workers ...")
    work = [
        (row, args.image_dir, output_dir, args.target_size, args.fill) for row in rows
    ]

    t0 = time.time()
    success = 0
    errors = []

    with Pool(args.workers) as pool:
        for i, (name, ok, err) in enumerate(pool.imap_unordered(_worker, work, chunksize=64)):
            if ok:
                success += 1
            else:
                errors.append((name, err))

            if (i + 1) % 5000 == 0 or (i + 1) == len(work):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(work) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1:,}/{len(work):,}] "
                    f"{rate:.0f} img/s | "
                    f"elapsed {elapsed:.0f}s | "
                    f"ETA {eta:.0f}s | "
                    f"errors: {len(errors)}"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {success:,} ok, {len(errors):,} errors")

    if errors:
        print(f"\nFirst 20 errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    # Write manifest CSV
    print("\nWriting manifest ...")
    manifest_rows = []
    for _, row in df.iterrows():
        name = row["image_name"]
        img_p = output_dir / "images" / name
        if img_p.exists():
            manifest_row = row.to_dict()
            manifest_row["image_path"] = f"images/{name}"
            manifest_row["mask_path"] = f"masks/{name}"
            manifest_row["patch_tissue_path"] = f"patch_tissue/{name}"
            manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "ImageData_v7_labeled.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Manifest: {manifest_path} ({len(manifest_df):,} rows)")

    # Stats
    print(f"\n--- Stats ---")
    print(f"  Total processed: {success:,}")
    print(f"  Errors: {len(errors):,}")
    if len(manifest_df) > 0:
        # Compute patch tissue coverage from a sample
        sample_paths = list((output_dir / "patch_tissue").iterdir())[:1000]
        tissue_patches = []
        for p in sample_paths:
            pm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                tissue_patches.append(int(np.sum(pm > 128)))
        if tissue_patches:
            print(
                f"  Patch tissue >50%: mean={np.mean(tissue_patches):.1f}/256, "
                f"min={np.min(tissue_patches)}, max={np.max(tissue_patches)}"
            )


if __name__ == "__main__":
    main()
