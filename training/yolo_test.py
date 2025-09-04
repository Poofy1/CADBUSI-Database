from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import yaml
from tqdm import tqdm

def load_model_and_data(base_dir, data_yaml_path):
    """Load the trained model and dataset information"""
    
    base_path = Path(base_dir)
    
    # Look for the trained model in various possible locations
    possible_model_paths = [
        base_path / "train1" / "weights" / "best.pt",
        base_path / "ultrasound_lesion_detection" / "yolo11m_lesions11" / "weights" / "best.pt",
        base_path / "yolo11m_lesions11" / "weights" / "best.pt",
        base_path / "weights" / "best.pt",
        base_path / "best.pt",
    ]
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            print(f"Found model at: {model_path}")
            break
    
    if model_path is None:
        print("Could not find trained model. Available files in base directory:")
        for item in base_path.iterdir():
            print(f"  {item}")
        
        # Look for any .pt files
        pt_files = list(base_path.glob("**/*.pt"))
        if pt_files:
            print("\nFound .pt files:")
            for pt_file in pt_files:
                print(f"  {pt_file}")
            model_path = pt_files[0]  # Use the first one found
            print(f"Using: {model_path}")
        else:
            return None, None
    
    model = YOLO(str(model_path))
    
    # Load dataset info
    if Path(data_yaml_path).exists():
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    else:
        print(f"Could not find data.yaml at {data_yaml_path}")
        return model, None
    
    return model, data_config

def get_all_images_from_directory(image_dir, max_images=None):
    """Get all images from specified directory, optionally limited to max_images"""
    
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Image directory does not exist: {image_dir}")
        return []
    
    print(f"Looking for images in: {image_dir}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    # Sort for consistent ordering
    image_files.sort()
    
    print(f"Total images found: {len(image_files)}")
    
    # Limit to max_images if specified
    if max_images is not None and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Limited to first {max_images} images")
    
    return image_files

def load_ground_truth_annotations(image_path, labels_dir):
    """Load ground truth annotations for an image"""
    
    # Convert image path to corresponding label path
    image_name = Path(image_path).stem
    label_path = Path(labels_dir) / f"{image_name}.txt"
    
    annotations = []
    
    if label_path.exists():
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        annotations.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
    
    return annotations

def convert_yolo_to_bbox(annotation, img_width, img_height):
    """Convert YOLO format (normalized) to bounding box coordinates"""
    
    center_x = annotation['center_x'] * img_width
    center_y = annotation['center_y'] * img_height
    width = annotation['width'] * img_width
    height = annotation['height'] * img_height
    
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0
    
    return intersection / union

def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5):
    """Match predictions to ground truth annotations using IoU threshold"""
    
    matched_predictions = []
    matched_gt = []
    
    # Convert predictions to the same format as ground truth for comparison
    pred_boxes = []
    for pred in predictions:
        pred_boxes.append(pred['bbox'])
    
    gt_boxes = []
    for gt in ground_truths:
        gt_boxes.append(gt['bbox'])
    
    # Track which GT boxes have been matched
    gt_matched = [False] * len(ground_truths)
    
    # For each prediction, find the best matching GT box
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:  # Skip already matched GT boxes
                continue
                
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j
        
        if best_gt_idx >= 0:
            matched_predictions.append(i)
            matched_gt.append(best_gt_idx)
            gt_matched[best_gt_idx] = True
    
    return matched_predictions, matched_gt, gt_matched

def draw_ground_truth_boxes(image, annotations, class_names, img_width, img_height):
    """Draw ground truth bounding boxes on the image"""
    
    # Ground truth color (green with dashed-like effect)
    gt_color = (0, 255, 0)  # Green for ground truth
    
    for annotation in annotations:
        x1, y1, x2, y2 = convert_yolo_to_bbox(annotation, img_width, img_height)
        
        # Draw dashed rectangle for ground truth (simulate dashed line)
        dash_length = 10
        
        # Top line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), gt_color, 3)
        
        # Bottom line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), gt_color, 3)
        
        # Left line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), gt_color, 3)
        
        # Right line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), gt_color, 3)
        
        # Add ground truth label
        class_id = annotation['class_id']
        if class_id < len(class_names):
            label = f"GT: {class_names[class_id]}"
        else:
            label = f"GT: Class {class_id}"
        
        # Calculate text size and background
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw background rectangle for text (offset to avoid overlap with prediction labels)
        text_y = y1 - 35 if y1 > 35 else y2 + text_height + 10
        cv2.rectangle(image, (x1, text_y - text_height - baseline - 5), 
                    (x1 + text_width, text_y + baseline), gt_color, -1)
        cv2.putText(image, label, (x1, text_y - baseline), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def add_legend(image):
    """Add a legend to distinguish between predictions and ground truth"""
    
    height, width = image.shape[:2]
    
    # Legend position (top-right corner)
    legend_x = width - 250
    legend_y = 30
    
    # Background for legend
    cv2.rectangle(image, (legend_x - 10, legend_y - 25), (width - 10, legend_y + 40), (0, 0, 0), -1)
    cv2.rectangle(image, (legend_x - 10, legend_y - 25), (width - 10, legend_y + 40), (255, 255, 255), 2)
    
    # Prediction legend (solid red line)
    cv2.rectangle(image, (legend_x, legend_y - 15), (legend_x + 30, legend_y - 5), (255, 0, 0), 3)
    cv2.putText(image, "Predictions", (legend_x + 40, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Ground truth legend (dashed green line approximation)
    for x in range(legend_x, legend_x + 30, 6):
        cv2.line(image, (x, legend_y + 10), (min(x + 3, legend_x + 30), legend_y + 10), (0, 255, 0), 3)
    cv2.putText(image, "Ground Truth", (legend_x + 40, legend_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def calculate_performance_metrics(all_predictions, all_ground_truths, iou_threshold=0.5):
    """Calculate comprehensive performance metrics"""
    
    total_predictions = len(all_predictions)
    total_ground_truths = len(all_ground_truths)
    
    if total_ground_truths == 0:
        print("No ground truth annotations found. Cannot calculate performance metrics.")
        return None
    
    # Match all predictions to ground truths
    matched_preds, matched_gts, gt_matched = match_predictions_to_gt(
        all_predictions, all_ground_truths, iou_threshold
    )
    
    # Calculate metrics
    true_positives = len(matched_preds)
    false_positives = total_predictions - true_positives
    false_negatives = total_ground_truths - true_positives
    
    # Precision: TP / (TP + FP)
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    
    # Recall (Sensitivity): TP / (TP + FN)
    recall = true_positives / total_ground_truths
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Miss Rate: FN / (FN + TP)
    miss_rate = false_negatives / total_ground_truths
    
    # False Positive Rate per image (approximation)
    total_images_with_gt = len(set(gt['image_name'] for gt in all_ground_truths))
    fp_rate_per_image = false_positives / total_images_with_gt if total_images_with_gt > 0 else 0
    
    return {
        'total_predictions': total_predictions,
        'total_ground_truths': total_ground_truths,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'miss_rate': miss_rate,
        'fp_rate_per_image': fp_rate_per_image,
        'iou_threshold': iou_threshold
    }

def process_all_images(model, image_paths, class_names, base_dir, confidence_threshold=0.3, 
                      output_dir="C:/Users/Tristan/Desktop/test_images_output", save_debug_images=True):
    """Process all images and optionally save annotated outputs with both predictions and ground truth"""
    
    # Create output directory only if saving debug images
    if save_debug_images:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine labels directory
    labels_dir = None
    possible_label_dirs = [
        Path(base_dir) / "labels" / "val",
        Path(base_dir) / "labels",
        Path(base_dir).parent / "labels" / "val",
        Path(base_dir).parent / "labels"
    ]
    
    for label_dir in possible_label_dirs:
        if label_dir.exists():
            labels_dir = label_dir
            print(f"Found labels directory: {labels_dir}")
            break
    
    if labels_dir is None:
        print("Warning: No labels directory found. Only predictions will be shown.")
        print("Looked in:")
        for dir_path in possible_label_dirs:
            print(f"  {dir_path}")
    
    print(f"\nProcessing {len(image_paths)} images...")
    if save_debug_images:
        print(f"Output directory: {output_dir}")
    else:
        print("Debug images will NOT be saved (performance metrics only)")
    
    # Collections for performance metrics
    all_predictions = []
    all_ground_truths = []
    
    total_detections = 0
    total_ground_truth = 0
    processed_count = 0
    images_with_gt = 0
    images_with_predictions = 0
    
    # Set progress bar description based on whether saving images
    progress_desc = "Processing images" if save_debug_images else "Calculating metrics"
    
    # Use tqdm for progress bar
    for idx, image_path in enumerate(tqdm(image_paths, desc=progress_desc, unit="img")):
        # Load image (only load if we need to save debug images, otherwise just get dimensions)
        if save_debug_images:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image_rgb.shape[:2]
        else:
            # For metrics-only mode, we still need image dimensions for bbox conversion
            # Use opencv to get image dimensions without loading full image data
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            img_height, img_width = image.shape[:2]
        
        # Load ground truth annotations
        gt_annotations = []
        if labels_dir:
            gt_annotations = load_ground_truth_annotations(image_path, labels_dir)
            if gt_annotations:
                images_with_gt += 1
                total_ground_truth += len(gt_annotations)
                
                # Add to performance tracking
                for gt in gt_annotations:
                    x1, y1, x2, y2 = convert_yolo_to_bbox(gt, img_width, img_height)
                    all_ground_truths.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': gt['class_id'],
                        'image_name': image_path.stem
                    })
        
        # Run inference
        results = model(str(image_path), conf=confidence_threshold)
        
        # Process predictions for metrics
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            images_with_predictions += 1
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Add to performance tracking
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                all_predictions.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls,
                    'confidence': conf,
                    'image_name': image_path.stem
                })
        
        # Only create and save debug images if requested
        if save_debug_images:
            # Start with a copy of the original image
            annotated_image = image_rgb.copy()
            
            # Draw ground truth first (so predictions appear on top)
            if gt_annotations:
                draw_ground_truth_boxes(annotated_image, gt_annotations, class_names, img_width, img_height)
            
            # Draw predictions
            if num_detections > 0:
                # Prediction colors (red variations)
                pred_colors = [(255, 0, 0), (255, 100, 0), (255, 0, 100), (200, 0, 0), (255, 50, 50)]
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    color = pred_colors[cls % len(pred_colors)]
                    
                    # Draw solid bounding box for predictions
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    
                    # Add prediction label
                    if cls < len(class_names):
                        label = f"PRED: {class_names[cls]}: {conf:.2f}"
                    else:
                        label = f"PRED: Class {cls}: {conf:.2f}"
                        
                    # Calculate text size and background
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(annotated_image, (x1, y1 - text_height - baseline - 10), 
                                (x1 + text_width, y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - baseline - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Add legend
            add_legend(annotated_image)
            
            # Save annotated image
            output_filename = f"{image_path.stem}_with_gt{image_path.suffix}"
            output_path = Path(output_dir) / output_filename
            
            # Convert RGB back to BGR for saving with cv2
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), annotated_bgr)
        
        processed_count += 1
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Images processed: {processed_count}")
    print(f"Images with ground truth: {images_with_gt}")
    print(f"Images with predictions: {images_with_predictions}")
    print(f"Total predictions: {total_detections}")
    print(f"Total ground truth annotations: {total_ground_truth}")
    if processed_count > 0:
        print(f"Average predictions per image: {total_detections/processed_count:.2f}")
        if images_with_gt > 0:
            print(f"Average ground truth per image (with GT): {total_ground_truth/images_with_gt:.2f}")
    
    if save_debug_images:
        print(f"Debug images saved to: {output_dir}")
        print("\nLegend:")
        print("  - RED solid boxes: Model predictions")
        print("  - GREEN dashed boxes: Ground truth annotations")
    else:
        print("Debug images were not saved (metrics only mode)")
    
    # Calculate and display performance metrics
    if all_ground_truths:
        print(f"\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        # Calculate metrics at different IoU thresholds
        iou_thresholds = [0.3, 0.5, 0.7]
        
        for iou_thresh in iou_thresholds:
            metrics = calculate_performance_metrics(all_predictions, all_ground_truths, iou_thresh)
            
            if metrics:
                print(f"\nAt IoU threshold {iou_thresh}:")
                print(f"  Precision:           {metrics['precision']:.3f} ({metrics['true_positives']}/{metrics['total_predictions']})")
                print(f"  Recall (Sensitivity): {metrics['recall']:.3f} ({metrics['true_positives']}/{metrics['total_ground_truths']})")
                print(f"  F1-Score:            {metrics['f1_score']:.3f}")
                print(f"  Miss Rate:           {metrics['miss_rate']:.3f} ({metrics['false_negatives']}/{metrics['total_ground_truths']})")
                print(f"  False Positives:     {metrics['false_positives']} predictions")
                print(f"  FP Rate per image:   {metrics['fp_rate_per_image']:.2f}")
        
        # Overall detection statistics
        print(f"\nOVERALL DETECTION STATISTICS:")
        print(f"  Detection Rate:      {images_with_predictions/processed_count:.3f} ({images_with_predictions}/{processed_count} images)")
        if images_with_gt > 0:
            images_with_both = len(set(pred['image_name'] for pred in all_predictions) & 
                                 set(gt['image_name'] for gt in all_ground_truths))
            print(f"  Coverage:            {images_with_both/images_with_gt:.3f} (detected in {images_with_both}/{images_with_gt} GT images)")
    
    return all_predictions, all_ground_truths

def main():
    """Main function to run inference"""
    
    # ========== CONFIGURATION ==========
    IMAGE_DIR = "C:/Users/Tristan/Desktop/testing"
    OUTPUT_DIR = "C:/Users/Tristan/Desktop/test_images_output"
    BASE_DIR = "D:/DATA/CASBUSI/training_sets/Yolo5/train2/"
    DATA_YAML_PATH = "D:/DATA/CASBUSI/training_sets/Yolo5/train2/args.yaml"
    CONFIDENCE_THRESHOLD = 0.3
    SAVE_DEBUG_IMAGES = True  # Set to False for faster processing (metrics only)
    MAX_IMAGES = None  # Set to None to process all images, or specify a number
    # ===================================
    
    print("Loading model...")
    model, data_config = load_model_and_data(BASE_DIR, DATA_YAML_PATH)
    
    if model is None:
        print("Could not load model. Please check your file paths.")
        return
    
    # Get class names
    class_names = []
    if data_config and 'names' in data_config:
        class_names = data_config['names']
        if isinstance(class_names, dict):
            class_names = list(class_names.values())
    else:
        print("No class names found in data config. Using generic names.")
        class_names = ["lesion"]
    
    print(f"Classes: {class_names}")
    
    # Get all images (limited by MAX_IMAGES)
    image_paths = get_all_images_from_directory(IMAGE_DIR, MAX_IMAGES)
    
    if not image_paths:
        print("No images found!")
        return
        
    # Process all images and optionally save outputs with ground truth
    all_predictions, all_ground_truths = process_all_images(
        model, image_paths, class_names, BASE_DIR, CONFIDENCE_THRESHOLD, OUTPUT_DIR, SAVE_DEBUG_IMAGES
    )
    
    print("\nDONE!")

if __name__ == "__main__":
    main()