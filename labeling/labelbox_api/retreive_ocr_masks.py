"""
Retrieve 'text_region' bounding-box annotations from a Labelbox project
and build a YOLO object-detection dataset.

Images come from  ocr_mask_raw_set/  (with its manifest.csv for splits).
The top half of every image is trimmed off before saving, and bounding-box
coordinates are adjusted to match the cropped geometry.

Output (zipped):
    data/ocr_mask_yolo/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── dataset.yaml
"""

import labelbox as lb
import pandas as pd
import os
import sys
import csv
import yaml
import shutil
from tqdm import tqdm
from PIL import Image

# project root (two levels up from labeling/labelbox_api/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIG

# ── Config ────────────────────────────────────────────────────────────

PROJECT_ID = "cmntjs7vx022o07xq9qwzbd2i"
RAW_SET_DIR = "labeling/ocr_mask_raw_set"
MANIFEST_CSV = os.path.join(RAW_SET_DIR, "manifest.csv")
IMAGES_SRC = os.path.join(RAW_SET_DIR, "images")

TRIM_TOP_HALF = False  # False → keep full image, output to ocr_mask_yolo_untrimmed

OUTPUT_DIR = "labeling/ocr_mask_yolo" if TRIM_TOP_HALF else "labeling/ocr_mask_yolo_untrimmed"

CLASS_NAMES = ["text_region"]  # single class, id=0


# ── Labelbox export ───────────────────────────────────────────────────

def export_annotations(project_id: str, client: lb.Client) -> list[dict]:
    """Export all annotations from a Labelbox project."""
    project = client.get_project(project_id)

    export_params = {
        "data_row_details": True,
        "label_details": True,
        "project_details": True,
    }

    print("Starting Labelbox export ...")
    export_task = project.export(params=export_params)
    export_task.wait_till_done()

    export_data = []
    for data_row in export_task.get_buffered_stream():
        export_data.append(data_row.json)

    print(f"Retrieved {len(export_data)} data rows")
    return export_data


def parse_text_region_boxes(export_data: list[dict]) -> dict[str, list[dict]]:
    """
    Parse 'text_region' bounding-box annotations.

    Returns:
        { image_name: [ {left, top, width, height}, ... ] }
    """
    results: dict[str, list[dict]] = {}
    total_boxes = 0
    no_labels = 0

    for item in tqdm(export_data, desc="Parsing annotations"):
        image_name = item["data_row"]["external_id"]
        boxes = []

        for _pid, project_data in item["projects"].items():
            for label in project_data.get("labels", []):
                for obj in label.get("annotations", {}).get("objects", []):
                    if obj.get("name") == "text_region" and "bounding_box" in obj:
                        bbox = obj["bounding_box"]
                        boxes.append({
                            "left":   bbox["left"],
                            "top":    bbox["top"],
                            "width":  bbox["width"],
                            "height": bbox["height"],
                        })

        if boxes:
            results[image_name] = boxes
            total_boxes += len(boxes)
        else:
            no_labels += 1

    print(f"Parsed {total_boxes} boxes across {len(results)} images "
          f"({no_labels} images had no labels)")
    return results


# ── Coordinate transform ─────────────────────────────────────────────

def transform_boxes_for_bottom_half(
    boxes: list[dict],
    orig_w: int,
    orig_h: int,
) -> list[tuple[float, float, float, float]]:
    """
    Given bounding boxes in original image coords, return YOLO-format
    labels (cx, cy, w, h) normalised to the *bottom half* of the image.

    Boxes fully above the midpoint are dropped.
    Boxes crossing the midpoint are clipped.
    """
    mid_y = orig_h / 2
    new_h = orig_h - mid_y  # height of bottom half

    yolo_boxes = []
    for b in boxes:
        bx1 = b["left"]
        by1 = b["top"]
        bx2 = bx1 + b["width"]
        by2 = by1 + b["height"]

        # Skip boxes entirely in the top half
        if by2 <= mid_y:
            continue

        # Clip to bottom half
        by1 = max(by1, mid_y)

        # Shift origin to top of bottom-half crop
        by1 -= mid_y
        by2 -= mid_y

        # Normalise to YOLO format
        cx = ((bx1 + bx2) / 2) / orig_w
        cy = ((by1 + by2) / 2) / new_h
        bw = (bx2 - bx1) / orig_w
        bh = (by2 - by1) / new_h

        # Clamp
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        yolo_boxes.append((cx, cy, bw, bh))

    return yolo_boxes


def boxes_to_yolo(
    boxes: list[dict],
    img_w: int,
    img_h: int,
) -> list[tuple[float, float, float, float]]:
    """Convert Labelbox boxes to YOLO format on the full (untrimmed) image."""
    yolo_boxes = []
    for b in boxes:
        cx = (b["left"] + b["width"] / 2) / img_w
        cy = (b["top"] + b["height"] / 2) / img_h
        bw = b["width"] / img_w
        bh = b["height"] / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        yolo_boxes.append((cx, cy, bw, bh))
    return yolo_boxes


# ── Dataset building ──────────────────────────────────────────────────

def build_yolo_dataset(
    annotations: dict[str, list[dict]],
    manifest: pd.DataFrame,
):
    """
    Crop images (remove top half), write YOLO labels, and organise into
    train/val/test folders based on the manifest.
    """
    # Prepare output dirs
    splits = manifest["split"].unique()
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    written = 0
    skipped_no_img = 0
    skipped_no_anno = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building dataset"):
        img_name = row["image_name"]
        split = row["split"]

        src_path = os.path.join(IMAGES_SRC, img_name)
        if not os.path.exists(src_path):
            skipped_no_img += 1
            continue

        # Skip images with no annotations
        boxes = annotations.get(img_name, [])
        if not boxes:
            skipped_no_anno += 1
            continue

        img = Image.open(src_path)
        orig_w, orig_h = img.size

        if TRIM_TOP_HALF:
            mid_y = orig_h // 2
            out_img = img.crop((0, mid_y, orig_w, orig_h))
            yolo_boxes = transform_boxes_for_bottom_half(boxes, orig_w, orig_h)
        else:
            out_img = img
            yolo_boxes = boxes_to_yolo(boxes, orig_w, orig_h)

        dst_img = os.path.join(OUTPUT_DIR, "images", split, img_name)
        out_img.save(dst_img)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        dst_lbl = os.path.join(OUTPUT_DIR, "labels", split, label_name)
        with open(dst_lbl, "w") as f:
            for (cx, cy, bw, bh) in yolo_boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        written += 1

    print(f"\nDataset written: {written} images")
    print(f"  Missing source image: {skipped_no_img}")
    print(f"  No annotations (saved as negative): {skipped_no_anno}")

    # Write dataset.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    dataset_cfg = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    # Add test split if present
    if "test" in splits:
        dataset_cfg["test"] = "images/test"

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"Wrote {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Read manifest
    if not os.path.exists(MANIFEST_CSV):
        print(f"Manifest not found: {MANIFEST_CSV}")
        return
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"Manifest: {len(manifest)} images  "
          f"({manifest['split'].value_counts().to_dict()})")

    # Export from Labelbox
    client = lb.Client(api_key=CONFIG["LABELBOX_API_KEY"])
    export_data = export_annotations(PROJECT_ID, client)

    # Parse bounding boxes
    annotations = parse_text_region_boxes(export_data)

    # Build YOLO dataset (crop top half, write labels)
    build_yolo_dataset(annotations, manifest)

    print(f"\nDone → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
