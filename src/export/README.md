# Preprocessing Pipeline

Standalone pipeline that takes US region polygons + debris polygons as input and produces
cropped, masked, 256px training images with tissue maps.

## Pipeline Overview

```
US polygon + debris polygons (from CADBUSI)
    |
    v
[1] align_polygon_axes.py    -- Snap near-axis edges to perfect H/V alignment
    |
    v
[2] tighten_crops.py         -- Trim dark empty rows (intensity-based, needs images)
    |
    v
[3] tighten_crops_fov.py     -- Clip crop box to FOV polygon extent (geometric, no images)
    |
    v
[4] build_preprocessed_v7.py -- Final: crop + mask + resize to 256px + patch tissue map
```

Steps 1-3 update the crop box coordinates in `ImageData.csv`. Step 4 reads the
refined crop boxes and produces the final dataset.

## Requirements

```
numpy
opencv-python
polars        # steps 1-3 (CSV processing)
pandas        # step 4 (CSV processing)
```

## Input Format

All scripts read an `ImageData.csv` with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_name` | str | Filename (e.g. `abc123.png`) |
| `us_polygon` | str | FOV polygon as `"x,y;x,y;..."` |
| `debris_polygons` | str | UI debris as `"x,y;x,y;...\|x,y;..."` (pipe-separated) |
| `crop_x`, `crop_y` | int | Crop box origin |
| `crop_w`, `crop_h` | int | Crop box dimensions |
| `rows`, `columns` | int | Original image dimensions (steps 1, 3) |

## Usage

### Step 1: Axis Alignment (geometric, no images needed)

Snaps polygon edges within 5 degrees of H/V to perfect alignment. Contracts
inward (conservative: lose thin tissue sliver, never gain background).

```bash
python align_polygon_axes.py \
    --imagedata ImageData.csv \
    --output ImageData_aligned.csv

# Dry run (stats only)
python align_polygon_axes.py --imagedata ImageData.csv --dry-run
```

### Step 2: Intensity-Based Crop Tightening (needs images)

Trims dark empty rows from top/bottom of crop box based on pixel intensity
within the FOV mask. Safety: never shrinks below 50% of original height.

```bash
python tighten_crops.py \
    --imagedata ImageData_aligned.csv \
    --image-dir /path/to/raw/images \
    --output ImageData_tightened.csv \
    --workers 8
```

**Parameters** (tuned on 1000 images across 5 scanners):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INTENSITY_THRESH` | 12 | Min mean intensity to count as signal |
| `WINDOW` | 15 | Smoothing window (rows) |
| `MARGIN` | 10 | Keep this many px beyond last signal row |
| `MIN_HEIGHT_RATIO` | 0.5 | Safety floor |

### Step 3: FOV-Based Crop Clipping (geometric, no images)

Clips crop box to the FOV polygon's actual extent. Runs ~10x faster than
step 2 since it only rasterizes the polygon (no image loading).

```bash
python tighten_crops_fov.py \
    --imagedata ImageData_tightened.csv \
    --output ImageData_final.csv \
    --workers 8
```

**Parameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_FOV_PIXELS_ROW` | 5 | Min FOV pixels per row to count as coverage |
| `MIN_FOV_PIXELS_COL` | 5 | Min FOV pixels per column |
| `MARGIN_H` | 2px | Horizontal margin beyond FOV |
| `MARGIN_V` | 3px | Vertical margin beyond FOV |
| `MIN_WIDTH_RATIO` | 0.90 | Never shrink width below 90% |
| `MIN_HEIGHT_RATIO` | 0.90 | Never shrink height below 90% |

### Step 4: Final Preprocessing

Produces training-ready 256px images with tissue masks and patch maps.

```bash
python build_preprocessed_v7.py \
    --imagedata ImageData_final.csv \
    --image-dir /path/to/raw/images \
    --output-dir /path/to/output \
    --workers 10

# Preview mode: 4-panel composites (raw + crop box, preprocessed, mask, patch map)
python build_preprocessed_v7.py \
    --imagedata ImageData_final.csv \
    --image-dir /path/to/raw/images \
    --preview --preview-dir /tmp/preview --preview-n 50
```

**Output structure**:
```
output_dir/
  images/{image_name}.png        -- 256x256 RGB, gray (128) fill
  masks/{image_name}.png         -- 256x256 binary (255=tissue, 0=non-tissue)
  patch_tissue/{image_name}.png  -- 16x16, each pixel = tissue count in that patch
  ImageData_v7_labeled.csv       -- manifest
```

**What `preprocess_one()` does**:
1. Load raw image
2. Rasterize FOV mask from `us_polygon` minus `debris_polygons` (via `ui_mask.py`)
3. Crop to optimized box (`crop_x/y/w/h`)
4. Fill non-tissue pixels with gray (128)
5. Resize to fit 256px (aspect-preserving, `INTER_LANCZOS4`)
6. Place on 256x256 canvas, top-left aligned
7. Compute 16x16 patch tissue count map (each cell = count of tissue pixels in that 16x16 patch)

## Core Function (for programmatic use)

```python
from ui_mask import compute_ui_mask
from build_preprocessed_v7 import preprocess_one

row = {
    "image_name": "abc123.png",
    "us_polygon": "10,10;10,800;1900,800;1900,10",
    "debris_polygons": "50,700;50,800;200,800;200,700",
    "crop_x": 10, "crop_y": 10, "crop_w": 1890, "crop_h": 790,
}

canvas_img, canvas_mask, patch_counts = preprocess_one(
    row, image_dir=Path("/path/to/images"), target_size=256, fill=128
)
# canvas_img:    (256, 256, 3) uint8 BGR
# canvas_mask:   (256, 256) uint8 binary
# patch_counts:  (16, 16) uint8
```

## Reference Files

| File | Purpose |
|------|---------|
| `ui_mask.py` | Parse polygon strings, rasterize FOV - debris = tissue mask |
| `align_polygon_axes.py` | Snap near-axis polygon edges to perfect H/V |
| `tighten_crops.py` | Intensity-based vertical crop tightening |
| `tighten_crops_fov.py` | Geometric FOV-based crop clipping |
| `build_preprocessed_v7.py` | Final preprocessing (crop + mask + resize + patch map) |
| `simplify_region.py` | Polygon simplification (reference; runs upstream in CADBUSI) |
| `PIPELINE.md` | Full 8-stage pipeline documentation |

## Polygon Storage Format

```
us_polygon:      "x1,y1;x2,y2;x3,y3;x4,y4"
debris_polygons:  "x1,y1;x2,y2;...|x1,y1;x2,y2;..."
                  ^-- polygon 1 --|^-- polygon 2 --|
```

Coordinates are in original image pixel space (before cropping). The `simplify_region.py`
module can convert between polygon arrays and this storage format.
