"""
Train YOLOv11-small to detect OCR text regions in ultrasound images.
Uses the dataset built by labeling/labelbox_api/retreive_ocr_masks.py
(images are already bottom-half cropped, grayscale).

After training, runs a thorough evaluation on val (and test if present).
"""

from ultralytics import YOLO
import torch
import os

# Debug GPU availability
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

# ── Paths ─────────────────────────────────────────────────────────────

DATA_YAML = "labeling/ocr_mask_yolo/dataset.yaml"
PROJECT_DIR = "training/ocr_mask_runs"
RUN_NAME = "yolo11s_ocr_masks"


# ── Training ──────────────────────────────────────────────────────────

def train(data_yaml: str, checkpoint: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from checkpoint: {checkpoint}")
        model = YOLO(checkpoint)
        resume = True
    else:
        print("Starting fresh training with yolo11s.pt")
        model = YOLO("yolo11s.pt")
        resume = False

    model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        device=device,
        patience=15,
        project=PROJECT_DIR,
        name=RUN_NAME,
        resume=resume,
        save_period=10,

        # B&W ultrasound augmentations
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=0.0,        # text regions don't rotate
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="",
        erasing=0.1,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Conservative LR for medical data
        lr0=0.0005,
        lrf=0.01,
    )

    return model


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(model: YOLO, data_yaml: str):
    """Thorough evaluation: val metrics, per-class stats, confusion matrix."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("EVALUATION — val set")
    print("=" * 60)

    metrics = model.val(
        data=data_yaml,
        split="val",
        device=device,
        plots=True,         # confusion matrix, PR curve, F1 curve
        save_json=True,     # COCO-format results
        conf=0.25,
        iou=0.5,
    )

    print(f"\n  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")

    # Per-class breakdown (useful even with 1 class — confirms nothing odd)
    if hasattr(metrics.box, "maps") and len(metrics.box.maps):
        print("\n  Per-class AP@50:")
        class_names = metrics.names if hasattr(metrics, "names") else {0: "text_region"}
        for i, ap in enumerate(metrics.box.maps):
            name = class_names.get(i, f"class_{i}")
            print(f"    {name}: {ap:.4f}")

    # Sweep confidence thresholds
    print("\n  Confidence sweep:")
    for conf in [0.1, 0.25, 0.5, 0.75]:
        m = model.val(data=data_yaml, split="val", device=device,
                      plots=False, conf=conf, iou=0.5, verbose=False)
        print(f"    conf={conf:.2f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}  "
              f"mAP50={m.box.map50:.3f}")

    # Test set if available
    try:
        print("\n" + "=" * 60)
        print("EVALUATION — test set")
        print("=" * 60)
        test_metrics = model.val(
            data=data_yaml,
            split="test",
            device=device,
            plots=True,
            conf=0.25,
            iou=0.5,
        )
        print(f"\n  Precision : {test_metrics.box.mp:.4f}")
        print(f"  Recall    : {test_metrics.box.mr:.4f}")
        print(f"  mAP@50    : {test_metrics.box.map50:.4f}")
        print(f"  mAP@50-95 : {test_metrics.box.map:.4f}")
    except Exception:
        print("  (no test split found, skipping)")

    print("\n" + "=" * 60)
    print(f"Plots & results saved to: {PROJECT_DIR}/{RUN_NAME}/")
    print("=" * 60)

    return metrics


# ── Failure visualisation ─────────────────────────────────────────────

def save_failure_images(model: YOLO, data_yaml: str, conf: float = 0.25, iou_thresh: float = 0.5):
    """
    Run prediction on val (and test) sets, find false positives, false
    negatives, and poor-IoU detections, then save annotated debug images.
    """
    from PIL import Image, ImageDraw
    import yaml as _yaml

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(data_yaml) as f:
        cfg = _yaml.safe_load(f)
    base_path = cfg["path"]

    out_dir = os.path.join(PROJECT_DIR, RUN_NAME, "failures")
    os.makedirs(out_dir, exist_ok=True)

    splits_to_check = ["val"]
    if "test" in cfg:
        splits_to_check.append("test")

    total_saved = 0

    for split in splits_to_check:
        img_dir = os.path.join(base_path, cfg[split])
        lbl_dir = img_dir.replace("images", "labels")
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)

        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        print(f"\nChecking {len(img_files)} {split} images for failures ...")

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")

            # Ground-truth boxes (pixel coords)
            gt_boxes = []
            img = Image.open(img_path)
            w, h = img.size
            if os.path.exists(lbl_path):
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append((x1, y1, x2, y2))

            # Predictions
            results = model.predict(img_path, device=device, conf=conf, verbose=False)
            pred_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    score = box.conf[0].cpu().item()
                    pred_boxes.append((x1, y1, x2, y2, score))

            # Classify failures
            gt_matched = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)

            for pi, (px1, py1, px2, py2, _) in enumerate(pred_boxes):
                best_iou, best_gi = 0, -1
                for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = ((px2 - px1) * (py2 - py1) +
                             (gx2 - gx1) * (gy2 - gy1) - inter)
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_thresh and best_gi >= 0:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True

            false_negatives = [gt_boxes[i] for i, m in enumerate(gt_matched) if not m]
            false_positives = [pred_boxes[i] for i, m in enumerate(pred_matched) if not m]

            if not false_negatives and not false_positives:
                continue  # all good, skip

            # Draw debug image
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)

            # GT boxes in green
            for (x1, y1, x2, y2) in gt_boxes:
                for o in range(2):
                    draw.rectangle((x1 - o, y1 - o, x2 + o, y2 + o), outline=(0, 255, 0))

            # False negatives — missed GT in yellow
            for (x1, y1, x2, y2) in false_negatives:
                for o in range(2):
                    draw.rectangle((x1 - o, y1 - o, x2 + o, y2 + o), outline=(255, 255, 0))
                draw.text((x1, y1 - 12), "MISS", fill=(255, 255, 0))

            # False positives — wrong predictions in red
            for (x1, y1, x2, y2, score) in false_positives:
                for o in range(2):
                    draw.rectangle((x1 - o, y1 - o, x2 + o, y2 + o), outline=(255, 0, 0))
                draw.text((x1, y1 - 12), f"FP {score:.2f}", fill=(255, 0, 0))

            # Correct predictions in blue
            for i, (x1, y1, x2, y2, score) in enumerate(pred_boxes):
                if pred_matched[i]:
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 120, 255))

            img.save(os.path.join(split_out, img_file))
            total_saved += 1

    print(f"\nSaved {total_saved} failure images → {out_dir}/")
    print("  Green = GT | Yellow+MISS = false negative | Red+FP = false positive | Blue = correct pred")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = train(DATA_YAML)
    evaluate(model, DATA_YAML)
    save_failure_images(model, DATA_YAML)
