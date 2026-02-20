"""UI Mask utilities for v7 dataset.

Parse polygon storage strings and rasterize composite UI masks
that represent clean ultrasound-only pixels (FOV minus UI debris).

Storage format:
  us_polygon:      "x,y;x,y;..."          (single polygon)
  debris_polygons:  "x,y;x,y;...|x,y;..."  (pipe-separated polygons)
"""

import numpy as np


def parse_polygon(s: str) -> np.ndarray:
    """Parse "x,y;x,y;..." → Nx2 float array.

    Returns empty (0,2) array if input is None/empty.
    """
    if not s:
        return np.array([]).reshape(0, 2)
    points = []
    for pt in s.split(";"):
        x, y = pt.split(",")
        points.append([float(x), float(y)])
    return np.array(points)


def parse_debris_polygons(s: str) -> list[np.ndarray]:
    """Parse pipe-separated debris polygons → list of Nx2 arrays.

    Returns empty list if input is None/empty.
    """
    if not s:
        return []
    return [parse_polygon(poly) for poly in s.split("|") if poly.strip()]


def compute_ui_mask(
    us_polygon_str: str | None,
    debris_polygons_str: str | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize composite UI mask (uint8, 0/255).

    1. Start with zeros (H, W)
    2. Fill us_polygon → 255 (FOV region)
    3. Fill each debris polygon → 0 (UI elements to exclude)

    Returns all-zeros mask if us_polygon is None/empty.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill FOV region
    fov = parse_polygon(us_polygon_str)
    if fov.size == 0:
        return mask
    cv2.fillPoly(mask, [fov.astype(np.int32)], 255)

    # Punch out debris regions
    for debris in parse_debris_polygons(debris_polygons_str):
        if debris.size > 0:
            cv2.fillPoly(mask, [debris.astype(np.int32)], 0)

    return mask


if __name__ == "__main__":
    # Quick self-test
    us = "10,10;10,100;100,100;100,10"
    debris = "20,20;20,40;40,40;40,20|60,60;60,80;80,80;80,60"

    mask = compute_ui_mask(us, debris, 120, 120)
    total_fov = np.count_nonzero(mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Non-zero pixels: {total_fov}")
    print(f"Expected: ~8100 (90x90 FOV) minus ~800 (two 20x20 debris) = ~7300")

    # Null safety
    null_mask = compute_ui_mask(None, None, 100, 100)
    assert null_mask.sum() == 0, "Null polygon should produce all-zeros mask"
    print("Null safety: OK")
