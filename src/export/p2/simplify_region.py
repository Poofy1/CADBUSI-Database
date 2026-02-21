"""Simplify US region masks to efficient polygonal representations.

Classifies regions into shape types and returns minimal vertex polygons:
- rectangle: 4 vertices (93% of images)
- hexagon: 6 vertices (EPIQ 5G fan shapes)
- arc_fan: 10-14 vertices (curved bottom fans)
- other: 20-40 vertices (panoramic, unusual shapes)

Usage:
    from simplify_region import simplify_us_region

    shape_type, polygon, iou = simplify_us_region(mask, scanner="EPIQ 5G")
    # polygon is Nx2 array of (x, y) vertices
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def _fit_polygon(contour: np.ndarray, n_vertices: int) -> Optional[np.ndarray]:
    """Approximate contour to exactly n vertices using binary search on epsilon."""
    peri = cv2.arcLength(contour, True)
    lo, hi = 0.0001, 0.1
    best = None

    for _ in range(20):
        mid = (lo + hi) / 2
        approx = cv2.approxPolyDP(contour, mid * peri, True)
        if len(approx) == n_vertices:
            return approx.reshape(-1, 2)
        elif len(approx) > n_vertices:
            lo = mid
        else:
            hi = mid
        if best is None or abs(len(approx) - n_vertices) < abs(len(best) - n_vertices):
            best = approx

    return best.reshape(-1, 2) if best is not None else None


def _polygon_iou(mask: np.ndarray, polygon: np.ndarray) -> float:
    """Compute IoU between binary mask and polygon."""
    h, w = mask.shape[:2]
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2, [polygon.astype(np.int32)], 255)
    m1 = mask > 127
    m2 = mask2 > 127
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0


def simplify_us_region(
    mask: np.ndarray,
    scanner: Optional[str] = None,
    min_iou: float = 0.96,
) -> Tuple[str, np.ndarray, float]:
    """
    Simplify US region mask to efficient polygon representation.

    Args:
        mask: Binary mask (H, W) uint8, 255=foreground
        scanner: Scanner model name (helps classify EPIQ 5G hexagons)
        min_iou: Minimum acceptable IoU for simplified polygon

    Returns:
        (shape_type, polygon, iou) where:
        - shape_type: "rectangle", "hexagon", "arc_fan", "other", or "empty"
        - polygon: Nx2 array of (x, y) vertices
        - iou: IoU between simplified polygon and original mask
    """
    # Find largest contour
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "empty", np.array([]).reshape(0, 2), 0.0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_ratio = area / (w * h) if w * h > 0 else 0

    # For EPIQ 5G: check hexagon first (fan shapes are common)
    if scanner == "EPIQ 5G":
        hex_poly = _fit_polygon(contour, 6)
        rect_poly = _fit_polygon(contour, 4)

        hex_iou = _polygon_iou(mask, hex_poly) if hex_poly is not None else 0
        rect_iou = _polygon_iou(mask, rect_poly) if rect_poly is not None else 0

        # Prefer hexagon if it's a better fit (even slightly)
        if hex_iou > rect_iou and hex_iou > min_iou:
            return "hexagon", hex_poly, hex_iou
        elif rect_iou > max(min_iou, 0.97) and rect_ratio > 0.95:
            # Only use rectangle if very rectangular
            return "rectangle", rect_poly, rect_iou
        elif hex_iou > min_iou:
            return "hexagon", hex_poly, hex_iou

    # Test 1: Rectangle (most common, 93% of images)
    rect_poly = _fit_polygon(contour, 4)
    if rect_poly is not None:
        rect_iou = _polygon_iou(mask, rect_poly)
        if rect_iou > max(min_iou, 0.97) and rect_ratio > 0.92:
            return "rectangle", rect_poly, rect_iou

    # Test 2: Hexagon (non-EPIQ 5G, rare)
    hex_poly = _fit_polygon(contour, 6)
    if hex_poly is not None:
        hex_iou = _polygon_iou(mask, hex_poly)
        if hex_iou > min_iou:
                return "hexagon", hex_poly, hex_iou

    # Test 3: Arc fan (curved bottom, 10-15 vertices)
    for n in [12, 13, 14, 11, 10, 15]:
        arc_poly = _fit_polygon(contour, n)
        if arc_poly is not None:
            arc_iou = _polygon_iou(mask, arc_poly)
            if arc_iou > max(min_iou, 0.97):
                return "arc_fan", arc_poly, arc_iou

    # Test 4: Other (panoramic, unusual) - use more vertices
    for n in [20, 25, 30, 40]:
        other_poly = _fit_polygon(contour, n)
        if other_poly is not None:
            other_iou = _polygon_iou(mask, other_poly)
            if other_iou > min_iou:
                return "other", other_poly, other_iou

    # Fallback: approxPolyDP with small epsilon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.002 * peri, True)
    poly = approx.reshape(-1, 2)
    iou = _polygon_iou(mask, poly)
    return "fallback", poly, iou


def polygon_to_storage(polygon: np.ndarray, precision: int = 1) -> str:
    """
    Convert polygon to compact storage string.

    Format: "x1,y1;x2,y2;..." with configurable decimal precision.
    For 1080p images with precision=1: ~8 bytes per vertex.

    Args:
        polygon: Nx2 array of (x, y) vertices
        precision: Decimal places (0=integer, 1=0.1 pixel)

    Returns:
        Compact string representation
    """
    if len(polygon) == 0:
        return ""
    fmt = f"{{:.{precision}f}}"
    points = [f"{fmt.format(x)},{fmt.format(y)}" for x, y in polygon]
    return ";".join(points)


def storage_to_polygon(storage: str) -> np.ndarray:
    """Convert storage string back to polygon array."""
    if not storage:
        return np.array([]).reshape(0, 2)
    points = []
    for pt in storage.split(";"):
        x, y = pt.split(",")
        points.append([float(x), float(y)])
    return np.array(points)


if __name__ == "__main__":
    # Demo: storage size comparison
    import sys

    print("Storage Size Comparison:")
    print("=" * 60)
    print()

    # Typical vertex counts by shape type
    shapes = [
        ("rectangle", 4),
        ("hexagon", 6),
        ("arc_fan", 12),
        ("other", 30),
    ]

    # Assume 1080p image (1920x1080), worst case coordinates
    for shape_name, n_verts in shapes:
        # Create sample polygon
        poly = np.array([[1920, 1080]] * n_verts, dtype=np.float32)

        # Storage formats
        compact = polygon_to_storage(poly, precision=1)
        json_flat = str(poly.flatten().tolist())

        print(f"{shape_name} ({n_verts} vertices):")
        print(f"  Compact string: {len(compact):4d} bytes")
        print(f"  JSON array:     {len(json_flat):4d} bytes")
        print(f"  Savings:        {100*(1 - len(compact)/len(json_flat)):.0f}%")
        print()

    # Dataset-level savings
    print("Dataset-Level Storage (962 frames, typical distribution):")
    print("-" * 60)

    # Based on actual distribution from task 3
    dist = {"rectangle": 894, "hexagon": 25, "arc_fan": 41, "other": 2}
    verts = {"rectangle": 4, "hexagon": 6, "arc_fan": 12, "other": 30}

    total_compact = 0
    total_json = 0
    for shape, count in dist.items():
        n = verts[shape]
        poly = np.array([[1920, 1080]] * n, dtype=np.float32)
        compact = polygon_to_storage(poly, precision=1)
        json_flat = str(poly.flatten().tolist())
        total_compact += len(compact) * count
        total_json += len(json_flat) * count

    print(f"  Compact storage: {total_compact/1024:.1f} KB")
    print(f"  JSON storage:    {total_json/1024:.1f} KB")
    print(f"  Savings:         {100*(1 - total_compact/total_json):.0f}%")
