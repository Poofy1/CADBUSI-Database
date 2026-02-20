#!/usr/bin/env python3
"""Infer US region polygons from full-resolution images.

Takes a CSV of image paths, runs U-Net segmentation, simplifies to efficient
polygons (rectangle/hexagon/arc_fan), and outputs results.

Usage:
    python infer_us_region.py input.csv -o output.csv
    python infer_us_region.py input.csv -o output.csv --erode 5
    python infer_us_region.py input.csv -o output.csv --scanner-col scanner

Input CSV must have column 'image_path' (or specify with --image-col).
Optional 'scanner' column helps classify EPIQ 5G hexagons.

Output CSV columns:
    image_path: Original image path
    shape_type: rectangle, hexagon, arc_fan, other, or empty
    n_vertices: Number of polygon vertices
    iou: IoU between simplified polygon and raw mask
    polygon: Semicolon-separated "x,y" vertices (e.g., "10.0,20.0;30.0,40.0;...")
"""

import argparse
import csv
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplify_region import simplify_us_region, polygon_to_storage


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load U-Net model from checkpoint."""
    import segmentation_models_pytorch as smp

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    encoder_name = checkpoint.get("encoder_name", "mobilenet_v2")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    img_size = checkpoint.get("img_size", 256)
    return model, img_size


def predict_mask_from_array(
    model: torch.nn.Module,
    img: np.ndarray,
    img_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Run U-Net inference on a BGR numpy array.

    Returns binary mask at original resolution (H, W), uint8 0/255.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_f = resized.astype(np.float32) / 255.0
    img_3ch = np.stack([img_f, img_f, img_f], axis=0)
    with torch.no_grad():
        x = torch.from_numpy(img_3ch).unsqueeze(0).to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask > 0.5).astype(np.uint8) * 255


def predict_mask(
    model: torch.nn.Module,
    image_path: str,
    img_size: int = 256,
    device: str = "cuda",
) -> Optional[np.ndarray]:
    """Run U-Net inference on a single image file.

    Returns binary mask at original resolution (H, W), uint8 0/255.
    Returns None if the image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    return predict_mask_from_array(model, img, img_size, device)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Erode binary mask by specified number of pixels."""
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.erode(mask, kernel, iterations=1)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate binary mask by specified number of pixels.

    Useful for recovering slightly under-segmented tissue at boundaries.
    1-2px is safe within the 10px UI safety margin.
    """
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def process_image(
    model: torch.nn.Module,
    image_path: str,
    img_size: int,
    device: str,
    scanner: Optional[str] = None,
    erode_pixels: int = 0,
    dilate_pixels: int = 0,
) -> Tuple[str, int, float, str]:
    """Process single image and return (shape_type, n_vertices, iou, polygon_str)."""
    mask = predict_mask(model, image_path, img_size, device)
    if mask is None:
        return "error", 0, 0.0, ""

    # Apply erosion if requested
    if erode_pixels > 0:
        mask = erode_mask(mask, erode_pixels)

    # Apply dilation if requested (after erosion)
    if dilate_pixels > 0:
        mask = dilate_mask(mask, dilate_pixels)

    # Simplify to polygon
    shape_type, polygon, iou = simplify_us_region(mask, scanner)

    if len(polygon) == 0:
        return shape_type, 0, iou, ""

    # Convert to storage string
    polygon_str = polygon_to_storage(polygon, precision=1)

    return shape_type, len(polygon), iou, polygon_str


def main():
    parser = argparse.ArgumentParser(
        description="Infer US region polygons from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_csv", help="Input CSV with image paths")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--checkpoint",
        default="segmentation/detection/crop_region/model/us_region_mobilenet_v2_v3_best.pth",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--image-col",
        default="image_path",
        help="Column name for image paths (default: image_path)",
    )
    parser.add_argument(
        "--scanner-col",
        default="scanner",
        help="Column name for scanner info (default: scanner)",
    )
    parser.add_argument(
        "--erode",
        type=int,
        default=0,
        help="Erode mask by N pixels before fitting polygon (default: 0)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="Dilate mask by N pixels after erosion (default: 0, 1-2px safe)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (currently only 1 supported)",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, img_size = load_model(args.checkpoint, args.device)
    print(f"  Model loaded, img_size={img_size}, device={args.device}")

    # Read input CSV
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.image_col not in rows[0]:
        print(f"Error: Column '{args.image_col}' not found in input CSV")
        print(f"  Available columns: {list(rows[0].keys())}")
        sys.exit(1)

    has_scanner = args.scanner_col in rows[0]
    if has_scanner:
        print(f"  Using scanner column: {args.scanner_col}")
    else:
        print(f"  No scanner column '{args.scanner_col}' found, will skip scanner hints")

    print(f"Processing {len(rows)} images (erode={args.erode}px, dilate={args.dilate}px)...")

    # Process images
    results = []
    stats = {"rectangle": 0, "hexagon": 0, "arc_fan": 0, "other": 0, "empty": 0, "error": 0}

    for row in tqdm(rows, desc="Inference"):
        image_path = row[args.image_col]
        scanner = row.get(args.scanner_col) if has_scanner else None

        shape_type, n_verts, iou, polygon_str = process_image(
            model, image_path, img_size, args.device, scanner, args.erode, args.dilate
        )

        stats[shape_type] += 1
        results.append({
            "image_path": image_path,
            "shape_type": shape_type,
            "n_vertices": n_verts,
            "iou": f"{iou:.4f}",
            "polygon": polygon_str,
        })

    # Write output
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "shape_type", "n_vertices", "iou", "polygon"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {args.output}")
    print(f"Shape distribution: {stats}")

    # Summary stats
    total = len(results)
    avg_verts = sum(r["n_vertices"] for r in results) / total if total > 0 else 0
    print(f"Average vertices per image: {avg_verts:.1f}")


if __name__ == "__main__":
    main()
