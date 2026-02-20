# FOV Crop Pipeline — End-to-End Documentation

> Last updated: 2026-02-12

This document describes the complete pipeline from raw ultrasound image to preprocessed,
cropped training input. The pipeline spans two codebases:

- **CADBUSI-Database** (developer's repo) — Stages 2, 4, 5, 6
- **BUS_framework** (this repo) — Stages 1, 3, 7, 8

---

## Stage 1: U-Net Mask Prediction

**File**: `segmentation/detection/crop_region/infer_us_region.py`

MobileNetV2 U-Net predicts a binary FOV (field-of-view) mask for each ultrasound image.

- **Architecture**: MobileNetV2 encoder → U-Net decoder
- **Training set**: 962 images across 5 scanners (LOGIQ E9, E10, EPIQ 5G, 7G, Elite)
- **Input**: Grayscale 256x256
- **Inference**: Sigmoid → threshold 0.5 → upscale to original resolution
- **Output**: Binary mask (H, W), uint8 0/255
- **Optional**: `erode_mask(pixels)` / `dilate_mask(pixels)` — currently unused by default
- **Checkpoint**: `model/us_region_mobilenet_v2_v3_best.pth` (v3, 77MB, Feb 4)

---

## Stage 2: Scanner-Specific Exclusion Boxes

**File**: `CADBUSI-Database/src/ML_processing/ultrasound_cropping.py`

Hard-coded per-scanner UI exclusion boxes applied to the mask before polygon fitting.
These remove scanner-specific UI elements (colorbars, rulers, probe indicators).

```python
EXCLUSION_BOXES = {
    'LOGIQE9':    {'right': 140, 'left_ratio': 0.04},
    'LOGIQE10':   {'right': 130, 'left_ratio': 0.04},
    'EPIQ 5G':    {'right': 135, 'left': 102, 'post_boxes': [(0, 0.6, 153, 1.0)]},
    'EPIQ 7G':    {'right': 135, 'left': 102},
    'EPIQ Elite': {'right': 135, 'left': 102},
}
```

| Parameter | Meaning |
|-----------|---------|
| `right` | Zero out this many pixel columns from the right edge |
| `left` | Zero out this many pixel columns from the left edge |
| `left_ratio` | Zero out this fraction of width from the left (LOGIQ colorbar/ruler) |
| `post_boxes` | Regions blacked out *after* crop decision (EPIQ 5G probe indicator) |

---

## Stage 3: Polygon Simplification

**File**: `segmentation/detection/crop_region/simplify_region.py` (225 lines)

Douglas-Peucker binary search to approximate the contour at a target vertex count.
The function tries progressively more complex shapes until IoU is acceptable.

**Classification hierarchy**:

1. **EPIQ 5G special path**: Try hexagon (6v) first, then rectangle (4v)
   - Hexagon preferred if `hex_iou > rect_iou` and `hex_iou > 0.96`
   - Rectangle only if `rect_iou > 0.97` AND `rect_ratio > 0.95`
2. **Rectangle (4v)**: `IoU > 0.97` AND `fill_ratio > 0.92` (~93% of images)
3. **Hexagon (6v)**: `IoU > 0.96`
4. **Arc fan (10-15v)**: `IoU > 0.97` (curved bottom fans)
5. **Other (20-40v)**: `IoU > 0.96` (panoramic, unusual shapes)
6. **Fallback**: `approxPolyDP` with `epsilon = 0.002 * perimeter`

**Output**: `(shape_type, polygon_Nx2, iou)`

**Storage format**: `polygon_to_storage()` → `"x,y;x,y;..."` compact string (~8 bytes/vertex)

---

## Stage 4: UI Detection & Debris Collection

**File**: `CADBUSI-Database/src/ML_processing/ultrasound_cropping.py` → `process_single_image_postprocessing()`

Three sources of "debris" polygons (UI elements inside the FOV that should be masked):

| Source | Scanner | Description |
|--------|---------|-------------|
| Exclusion boxes | All | Rectangles from Stage 2 |
| YOLO orientation marker | LOGIQ only | Probe icons in bottom half of image |
| OCR text detection | All | EasyOCR on bottom quarter + left panel (LOGIQ) |

- **YOLO model**: `LOGIQE_ori_yolo_2026_02_06.pt` (HuggingFace: `poofy38/CADBUSI`)
- **OCR region**: Restricted to non-excluded x-range
- **LOGIQ mbox heuristic**: Text detected in right third gets 30px upward extension

Debris polygons stored pipe-separated: `"x,y;x,y;...|x,y;x,y;..."`

---

## Stage 5: Crop Box Computation

**File**: `CADBUSI-Database/src/ML_processing/ultrasound_cropping.py` → `mask_to_bbox()` + OCR height adjustment

1. Bounding box of exclusion-cleaned binary mask
2. Bottom adjusted upward if OCR text overlaps FOV
3. Output: `(crop_x, crop_y, crop_w, crop_h)`

**Known issue**: Crop box is the axis-aligned bounding box of the raw mask. If mask polygon edges are slightly tilted (1-20px from axis), the crop includes slivers of non-FOV background. This is fixed in Stage 7.

---

## Stage 6: Database Storage

**File**: `CADBUSI-Database` → Updates the ImageData table

```sql
UPDATE Images SET crop_x=?, crop_y=?, crop_w=?, crop_h=?,
    us_polygon=?, debris_polygons=? WHERE image_name=?
```

All downstream stages read from this table (exported as `ImageData.csv`).

---

## Stage 7: v7 Post-Processing (Sliver Fix)

Three scripts run in sequence to fix the axis-alignment / sliver problem from Stage 5.

### 7a. Axis Alignment

**File**: `data/registry/scripts/align_polygon_axes.py`

Snaps near-axis polygon edges to perfect horizontal/vertical alignment.

- **Conservative**: Contracts toward polygon interior (lose thin tissue sliver, not gain background)
- **Threshold**: 5 degrees from axis OR ≤20px absolute deviation
- **Multi-pass**: Resolves shared-vertex conflicts
- **IoU gate**: ≥0.96 or reject alignment
- **Output**: Writes aligned polygons back to ImageData CSV

### 7b. Intensity-Based Crop Tightening

**File**: `data/registry/scripts/tighten_crops.py`

Loads actual images and trims dark empty rows from crop top/bottom.

| Parameter | Value |
|-----------|-------|
| `intensity_thresh` | 12 |
| `window` | 15 |
| `margin` | 10 |
| Safety | Never shrink below 50% of original height |

### 7c. FOV-Based Crop Clipping

**File**: `data/registry/scripts/tighten_crops_fov.py`

Purely geometric (no image loading) — rasterizes FOV polygon and clips crop to FOV extent.

| Parameter | Value |
|-----------|-------|
| Min FOV pixels per row/col | 5 |
| Horizontal margin | 2px |
| Vertical margin | 3px |
| Safety | Never shrink below 90% of original dimensions |

---

## Stage 8: Final Preprocessing

**File**: `data/registry/scripts/build_preprocessed_v7.py`

Produces the training-ready dataset:

1. Load raw PNG
2. Rasterize FOV mask via `ui_mask.compute_ui_mask(us_polygon, debris_polygons, h, w)`
3. Crop to optimized box (from Stage 7)
4. Fill non-tissue pixels with gray (128)
5. Resize to 256px (aspect-preserving, top-left aligned)
6. Compute 16x16 patch tissue count map
7. Save image + mask + patch map

---

## Local Reimplementations

We maintain local versions of alternative crop methods for comparison:

| File | Method |
|------|--------|
| `cadbusi_crop.py` | Faithful CADBUSI reimplementation (threshold → erode → hull → OCR) |
| `apply_busclean_crop.py` | BUSClean method (mode+10 threshold, morph, median-of-thirds) |
| `texture_crop.py` | Local variance speckle detection (aggressive morph open/close) |
| `compute_crop_boxes_v4.py` | Our improved version (colorbar + OCR + sector + blackout tracking) |
| `ui_detector_rules.py` | Rule-based UI detector (templates + OCR patterns + scanner profiles) |
| `detect_colorbar.py` | LOGIQ colorbar template matching |

---

## Typical Shape Distribution (962 training images)

| Shape | Count | Percentage |
|-------|-------|------------|
| rectangle | ~894 | ~93% |
| hexagon | ~25 | ~3% |
| arc_fan | ~41 | ~4% |
| other | ~2 | <1% |

---

## Design Decisions & History

### Why DP-based simplify_region.py (not structural fitting)

A structural fitting approach was attempted (Feb 2026, archived as `simplify_region_structural_v1.py.bak`)
that tried to detect sector shapes by fitting lines to contour segments. This caused regressions:
EPIQ 7G and Elite sectors were misclassified as trapezoids, producing worse polygons.

The DP approach works reliably because:
1. It's agnostic to shape geometry — just binary-searches for vertex count
2. The IoU gates catch bad fits before they propagate
3. The v7 post-processing (Stage 7) already fixes the sliver problem downstream

### Why two-codebase split

The developer's CADBUSI-Database handles the initial crop pipeline (Stages 2, 4-6) because it has
access to the full image database and scanner metadata. Our BUS_framework handles the ML
prediction (Stage 1), polygon simplification (Stage 3), and post-processing (Stage 7-8)
because these are research components that evolve independently.
