from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
import re
from tools.storage_adapter import *
current_dir = os.path.dirname(os.path.dirname(__file__))

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
env = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_DF_DIR = f"{env}/labeling/labelbox_annotations.csv"
IMAGE_DF_DIR = f"{CONFIG['DATABASE_DIR']}/ImageData.csv"
IMAGE_DIR = f"{CONFIG['DATABASE_DIR']}images/"

def load_model():
    """Load the trained model"""
    model_path = f"{env}/src/ML_processing/models/yolo_lesion_detect.pt"
    model = YOLO(str(model_path))
    return model

def load_image_data_mapping():
    """Load ImageData.csv to get crop coordinates for each DicomHash"""
    try:
        image_df = read_csv(IMAGE_DF_DIR)
        print(f"Loaded ImageData.csv with {len(image_df)} rows")
        
        # Create mapping from DicomHash to crop coordinates and ImageName
        crop_mapping = {}
        for _, row in image_df.iterrows():
            dicom_hash = row['DicomHash']
            crop_mapping[dicom_hash] = {
                'crop_x': row.get('crop_x', 0),
                'crop_y': row.get('crop_y', 0), 
                'crop_w': row.get('crop_w', None),
                'crop_h': row.get('crop_h', None),
                'image_name': row.get('ImageName', ''),
                'original_rows': row.get('Rows', None),
                'original_columns': row.get('Columns', None),
                'has_calipers': row.get('has_calipers', False)
            }
        
        print(f"Created crop mapping for {len(crop_mapping)} DicomHash entries")
        return crop_mapping
        
    except Exception as e:
        print(f"Error loading ImageData.csv: {e}")
        return {}

def adjust_bbox_for_crop(bbox_absolute, crop_x, crop_y, crop_w, crop_h):
    """
    Adjust bounding box coordinates for cropped image
    
    Args:
        bbox_absolute: [left, top, width, height] in original image coordinates
        crop_x, crop_y, crop_w, crop_h: crop parameters
    
    Returns:
        Adjusted bbox in crop coordinates, or None if bbox is outside crop area
    """
    left, top, width, height = bbox_absolute
    
    # Convert to x1, y1, x2, y2 format for easier calculation
    x1, y1 = left, top
    x2, y2 = left + width, top + height
    
    # Crop boundaries
    crop_x1, crop_y1 = crop_x, crop_y
    crop_x2, crop_y2 = crop_x + crop_w, crop_y + crop_h
    
    # Check if bbox intersects with crop area
    if x2 <= crop_x1 or x1 >= crop_x2 or y2 <= crop_y1 or y1 >= crop_y2:
        return None  # No intersection
    
    # Clip bbox to crop boundaries
    clipped_x1 = max(x1, crop_x1)
    clipped_y1 = max(y1, crop_y1)
    clipped_x2 = min(x2, crop_x2)
    clipped_y2 = min(y2, crop_y2)
    
    # Convert to crop coordinate system
    new_x1 = clipped_x1 - crop_x
    new_y1 = clipped_y1 - crop_y
    new_x2 = clipped_x2 - crop_x
    new_y2 = clipped_y2 - crop_y
    
    # Convert back to [left, top, width, height] format
    new_left = new_x1
    new_top = new_y1
    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1
    
    # Only return bbox if it has meaningful size
    if new_width > 5 and new_height > 5:
        return [new_left, new_top, new_width, new_height]
    
    return None

def crop_image(image, crop_x, crop_y, crop_w, crop_h):
    """
    Crop image using the specified coordinates
    
    Args:
        image: Input image (numpy array)
        crop_x, crop_y, crop_w, crop_h: crop parameters
    
    Returns:
        Cropped image
    """
    if crop_w is None or crop_h is None:
        return image  # Return original if no crop info
    
    # Convert all parameters to integers, handling NaN and None values
    try:
        crop_x = int(float(crop_x)) if crop_x is not None and not pd.isna(crop_x) else 0
        crop_y = int(float(crop_y)) if crop_y is not None and not pd.isna(crop_y) else 0
        crop_w = int(float(crop_w)) if crop_w is not None and not pd.isna(crop_w) else None
        crop_h = int(float(crop_h)) if crop_h is not None and not pd.isna(crop_h) else None
    except (ValueError, TypeError):
        print(f"Warning: Invalid crop parameters: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        return image
    
    # Return original if crop dimensions are invalid
    if crop_w is None or crop_h is None or crop_w <= 0 or crop_h <= 0:
        return image
    
    # Ensure crop coordinates are within image bounds
    h, w = image.shape[:2]
    crop_x = max(0, min(crop_x, w))
    crop_y = max(0, min(crop_y, h))
    crop_w = min(crop_w, w - crop_x)
    crop_h = min(crop_h, h - crop_y)
    
    # Final validation
    if crop_w <= 0 or crop_h <= 0:
        return image
    
    # Perform crop
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    return cropped

def load_ground_truth_from_csv():
    """Load ground truth annotations from the Labelbox CSV file and adjust for cropping"""
    
    print(f"Loading ground truth from: {LABEL_DF_DIR}")
    print(f"Loading image data from: {IMAGE_DF_DIR}")
    
    try:
        # Load the labelbox annotations
        df = pd.read_csv(LABEL_DF_DIR)
        print(f"Loaded {len(df)} annotations from CSV")
        
        # Load crop mapping from ImageData.csv
        crop_mapping = load_image_data_mapping()
        if not crop_mapping:
            print("Warning: No crop mapping loaded. Proceeding without cropping.")
        
        ground_truth_data = {}
        images_with_crop_info = 0
        images_processed = 0
        bboxes_adjusted = 0
        bboxes_removed = 0
        
        for _, row in df.iterrows():
            dicom_hash = row['dicom_hash']
            lesion_coords = row['lesion_coords']
            axillary_coords = row['axillary_node_coords']
            
            # Get crop information for this DicomHash
            crop_info = crop_mapping.get(dicom_hash)
            if not crop_info:
                print(f"Warning: No crop info found for DicomHash {dicom_hash}")
                continue
            
            image_name = crop_info['image_name']
            if not image_name:
                continue
            
            # Remove extension for consistent matching
            image_stem = Path(image_name).stem
            
            # Get crop parameters
            crop_x = crop_info.get('crop_x', 0) or 0
            crop_y = crop_info.get('crop_y', 0) or 0
            crop_w = crop_info.get('crop_w')
            crop_h = crop_info.get('crop_h')
            
            if crop_w and crop_h:
                images_with_crop_info += 1
            
            annotations = []
            
            # Parse lesion coordinates and treat as class_id = 0
            if lesion_coords and pd.notna(lesion_coords):
                lesion_boxes = parse_coordinate_string(lesion_coords)
                for box in lesion_boxes:
                    if crop_w and crop_h:
                        adjusted_box = adjust_bbox_for_crop(box, crop_x, crop_y, crop_w, crop_h)
                        if adjusted_box is not None:
                            annotations.append({
                                'class_id': 0,
                                'bbox_absolute': adjusted_box,
                                'class_name': 'detection',
                                'original_class': 'lesion'
                            })
                            bboxes_adjusted += 1
                        else:
                            bboxes_removed += 1
                    else:
                        # No crop info, use original coordinates
                        annotations.append({
                            'class_id': 0,
                            'bbox_absolute': box,
                            'class_name': 'detection',
                            'original_class': 'lesion'
                        })
            
            # Parse axillary node coordinates and also treat as class_id = 0
            if axillary_coords and pd.notna(axillary_coords):
                axillary_boxes = parse_coordinate_string(axillary_coords)
                for box in axillary_boxes:
                    if crop_w and crop_h:
                        adjusted_box = adjust_bbox_for_crop(box, crop_x, crop_y, crop_w, crop_h)
                        if adjusted_box is not None:
                            annotations.append({
                                'class_id': 0,
                                'bbox_absolute': adjusted_box,
                                'class_name': 'detection',
                                'original_class': 'axillary_node'
                            })
                            bboxes_adjusted += 1
                        else:
                            bboxes_removed += 1
                    else:
                        # No crop info, use original coordinates
                        annotations.append({
                            'class_id': 0,
                            'bbox_absolute': box,
                            'class_name': 'detection', 
                            'original_class': 'axillary_node'
                        })
            
            if crop_info and image_name:  # Store all images with valid crop info
                ground_truth_data[image_stem] = {
                    'annotations': annotations,  # This could be an empty list
                    'crop_info': crop_info
                }
                images_processed += 1
        
        print(f"Parsed ground truth for {images_processed} images")
        print(f"Images with crop info: {images_with_crop_info}")
        print(f"Bounding boxes adjusted for cropping: {bboxes_adjusted}")
        print(f"Bounding boxes removed (outside crop): {bboxes_removed}")
        
        # Print summary of combined annotations
        total_lesions = sum(1 for image_data in ground_truth_data.values() 
                           for ann in image_data['annotations'] if ann['original_class'] == 'lesion')
        total_axillary = sum(1 for image_data in ground_truth_data.values() 
                            for ann in image_data['annotations'] if ann['original_class'] == 'axillary_node')
        total_annotations = total_lesions + total_axillary
        
        print(f"Final annotations: {total_lesions} lesions + {total_axillary} axillary nodes = {total_annotations} total detections")
        
        return ground_truth_data
        
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        return {}

def parse_coordinate_string(coord_string):
    """Parse coordinate string like '[242,269,334,178];[242,269,342,188]' into list of boxes"""
    
    if not coord_string or pd.isna(coord_string):
        return []
    
    boxes = []
    # Find all coordinate groups in brackets
    pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
    matches = re.findall(pattern, coord_string)
    
    for match in matches:
        left, top, width, height = map(int, match)
        boxes.append([left, top, width, height])
    
    return boxes

def get_images_with_ground_truth(ground_truth_data, max_images=None):
    """Get images that have ground truth annotations using existing image names"""
    print(f"Building image paths from ground truth data...")
    image_files = []
    
    # Get image names directly from ground truth data
    for image_stem, data in ground_truth_data.items():
        crop_info = data.get('crop_info', {})
        image_name = crop_info.get('image_name', '')
        
        if image_name:
            # Construct full path
            image_path = Path(IMAGE_DIR) / image_name
            image_files.append(image_path)
            
    # Sort for consistent ordering
    image_files.sort()
    
    print(f"Found {len(image_files)} existing image files")
    
    # Limit to max_images if specified
    if max_images is not None and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Limited to first {max_images} images")
    
    return image_files

def convert_absolute_to_xyxy(bbox_absolute):
    """Convert [left, top, width, height] to [x1, y1, x2, y2]"""
    left, top, width, height = bbox_absolute
    return [left, top, left + width, top + height]

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


def save_visual_examples(all_predictions, all_ground_truths, ground_truth_data, current_dir, max_examples=10):
    """Save visual examples showing ground truth vs predictions"""
    
    print(f"\nGenerating visual examples...")
    
    # Group by image
    image_predictions = {}
    image_ground_truths = {}
    
    for pred in all_predictions:
        img_name = pred['image_name']
        if img_name not in image_predictions:
            image_predictions[img_name] = []
        image_predictions[img_name].append(pred)
    
    for gt in all_ground_truths:
        img_name = gt['image_name']
        if img_name not in image_ground_truths:
            image_ground_truths[img_name] = []
        image_ground_truths[img_name].append(gt)
    
    # Select diverse examples
    examples = []
    
    # Get images with both predictions and ground truth
    common_images = set(image_predictions.keys()) & set(image_ground_truths.keys())
    
    # Get some images with only ground truth (missed detections)
    gt_only_images = set(image_ground_truths.keys()) - set(image_predictions.keys())
    
    # Get some images with only predictions (false positives)
    pred_only_images = set(image_predictions.keys()) - set(image_ground_truths.keys())
    
    # Mix different types of examples
    examples.extend(list(common_images)[:max_examples//2])  # Mixed cases
    examples.extend(list(gt_only_images)[:2])  # Missed detections
    examples.extend(list(pred_only_images)[:2])  # False positives only
    
    # Limit total examples
    examples = examples[:max_examples]
    
    print(f"Creating {len(examples)} visual examples...")
    
    for i, img_name in enumerate(examples):
        try:
            # Get ground truth data for this image
            gt_data = ground_truth_data.get(img_name, {})
            crop_info = gt_data.get('crop_info', {})
            image_name = crop_info.get('image_name', '')
            
            if not image_name:
                continue
            
            # Load and crop the image
            image_path = Path(IMAGE_DIR) / image_name
            image = read_image(str(image_path))
            if image is None:
                continue
            
            # Apply cropping if available
            if crop_info.get('crop_w') and crop_info.get('crop_h'):
                crop_x = crop_info.get('crop_x', 0) or 0
                crop_y = crop_info.get('crop_y', 0) or 0
                crop_w = crop_info['crop_w']
                crop_h = crop_info['crop_h']
                image = crop_image(image, crop_x, crop_y, crop_w, crop_h)
            
            # Convert to RGB for matplotlib
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            # Draw ground truth boxes (green)
            gt_lesions = []
            gt_nodes = []
            if img_name in image_ground_truths:
                for gt in image_ground_truths[img_name]:
                    x1, y1, x2, y2 = gt['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    if gt['original_class'] == 'lesion':
                        color = 'lime'
                        gt_lesions.append((x1, y1, width, height))
                    else:  # axillary_node
                        color = 'green'
                        gt_nodes.append((x1, y1, width, height))
                    
                    rect = plt.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
            
            # Draw prediction boxes (red/orange)
            pred_count = 0
            if img_name in image_predictions:
                for pred in image_predictions[img_name]:
                    x1, y1, x2, y2 = pred['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    conf = pred['confidence']
                    
                    # Color by confidence
                    if conf > 0.7:
                        color = 'red'
                    elif conf > 0.5:
                        color = 'orange'
                    else:
                        color = 'yellow'
                    
                    rect = plt.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    
                    # Add confidence label
                    ax.text(x1, y1-5, f'{conf:.2f}', color=color, fontsize=10, fontweight='bold')
                    pred_count += 1
            
            # Add title with counts and class info
            title = f'Example {i+1}: {img_name}\n'
            title += f'GT: {len(gt_lesions)} lesions, {len(gt_nodes)} nodes | Pred: {pred_count} detections'
            ax.set_title(title, fontsize=12)
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, linewidth=2, edgecolor='lime', facecolor='none', label='GT Lesions'),
                plt.Rectangle((0,0),1,1, linewidth=2, edgecolor='green', facecolor='none', label='GT Nodes'),
                plt.Rectangle((0,0),1,1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Predictions')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save the figure
            save_path = f"{current_dir}/example_{i+1:02d}_{img_name}.png"
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Saved example {i+1}: {save_path}")
            
        except Exception as e:
            print(f"Error creating example {i+1} for {img_name}: {e}")
            continue
    
    print(f"Visual examples saved to {current_dir}/")
    
    
def create_confusion_matrix_data(all_predictions, ground_truths_for_class, class_name, iou_threshold=0.3):
    """Create confusion matrix data for a specific class"""
    
    # Match predictions to this class's ground truth
    matched_preds, matched_gts, gt_matched = match_predictions_to_gt(
        all_predictions, ground_truths_for_class, iou_threshold
    )
    
    tp = len(matched_gts)  # Counts how many ground truth boxes were successfully matched to predictions
    fn = len(ground_truths_for_class) - tp  # Total ground truth boxes minus the matched ones
    fp = len(all_predictions) - len(matched_preds)  # Total predictions minus those that matched ground truth
    
    # For TN, we'll use 0 since it's hard to define in object detection
    tn = 0
    
    return {
        'TP': tp,
        'FP': fp, 
        'FN': fn,
        'TN': tn,
        'class_name': class_name
    }

def plot_confusion_matrix(cm_data, save_path):
    """Plot and save confusion matrix"""
    
    # Create 2x2 matrix
    cm = np.array([
        [cm_data['TP'], cm_data['FP']],
        [cm_data['FN'], cm_data['TN']]
    ])
    
    # Calculate percentages
    total = cm_data['TP'] + cm_data['FP'] + cm_data['FN']
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot with both counts and percentages
    labels = np.array([
        [f"TP\n{cm_data['TP']}\n({cm_data['TP']/total*100:.1f}%)" if total > 0 else f"TP\n{cm_data['TP']}", 
         f"FP\n{cm_data['FP']}\n({cm_data['FP']/total*100:.1f}%)" if total > 0 else f"FP\n{cm_data['FP']}"],
        [f"FN\n{cm_data['FN']}\n({cm_data['FN']/total*100:.1f}%)" if total > 0 else f"FN\n{cm_data['FN']}", 
         "TN\n0\n(N/A)"]
    ])
    
    # Create heatmap
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True, 
                square=True, annot_kws={'size': 12})
    
    plt.title(f'Confusion Matrix - Overall Detection\n(IoU ≥ 0.3)', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    
    # Set tick labels
    plt.gca().set_yticklabels(['Detected', 'Not Detected'], rotation=0)
    plt.gca().set_xticklabels(['Match', 'No Match'], rotation=0)
    
    # Add summary text
    precision = cm_data['TP'] / (cm_data['TP'] + cm_data['FP']) if (cm_data['TP'] + cm_data['FP']) > 0 else 0
    recall = cm_data['TP'] / (cm_data['TP'] + cm_data['FN']) if (cm_data['TP'] + cm_data['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.02, 0.02, f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to: {save_path}")
    
    
def save_confusion_matrices(all_predictions, all_ground_truths, current_dir, iou_threshold=0.3):
    """Generate and save a single confusion matrix for overall detection performance"""
    
    if not all_ground_truths:
        print("No ground truth data, skipping confusion matrix")
        return
    
    # Calculate overall detection metrics (all classes combined)
    matched_preds, matched_gts, _ = match_predictions_to_gt(all_predictions, all_ground_truths, iou_threshold)
    
    tp = len(matched_gts)  # Ground truth annotations detected
    fn = len(all_ground_truths) - tp  # Ground truth annotations missed
    fp = len(all_predictions) - len(matched_preds)  # Predictions that didn't match any ground truth
    tn = 0  # Not meaningful in object detection
    
    cm_data = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'class_name': 'overall_detection'
    }
    
    # Save the combined confusion matrix
    save_path = f"{current_dir}/confusion_matrix_overall_detection.png"
    plot_confusion_matrix(cm_data, save_path)
    
    print(f"\nConfusion Matrix Summary:")
    print(f"  True Positives (TP): {tp} - Ground truth detections found")
    print(f"  False Positives (FP): {fp} - Predictions with no matching ground truth") 
    print(f"  False Negatives (FN): {fn} - Ground truth detections missed")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
        
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

def calculate_ap_at_iou(predictions, ground_truths, iou_threshold):
    """Calculate Average Precision at a specific IoU threshold"""
    if not ground_truths:
        return 0.0
    
    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Match predictions to ground truth
    matched_preds, matched_gts, gt_matched = match_predictions_to_gt(
        sorted_preds, ground_truths, iou_threshold
    )
    
    # Calculate precision and recall at each prediction
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    
    for i, pred in enumerate(sorted_preds):
        if i in matched_preds:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
        recall = tp_cumsum / len(ground_truths)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for recall_thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Find max precision for recalls >= recall_thresh
        max_precision = 0.0
        for r, p in zip(recalls, precisions):
            if r >= recall_thresh:
                max_precision = max(max_precision, p)
        ap += max_precision
    
    return ap / 11.0

def calculate_map30_95(predictions, ground_truths):
    """Calculate mAP30-95 (mean AP from IoU 0.3 to 0.95)"""
    iou_thresholds = [0.3 + 0.05 * i for i in range(14)]  # 0.3 to 0.95 in steps of 0.05
    
    aps = []
    for iou_thresh in iou_thresholds:
        ap = calculate_ap_at_iou(predictions, ground_truths, iou_thresh)
        aps.append(ap)
    
    return sum(aps) / len(aps), aps

def calculate_map50_95(predictions, ground_truths):
    """Calculate mAP50-95 (mean AP from IoU 0.5 to 0.95)"""
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95 in steps of 0.05
    
    aps = []
    for iou_thresh in iou_thresholds:
        ap = calculate_ap_at_iou(predictions, ground_truths, iou_thresh)
        aps.append(ap)
    
    return sum(aps) / len(aps), aps

def calculate_metrics_only(model, image_paths, class_names, ground_truth_data, confidence_threshold=0.3):
    """Process all images for performance metrics with mAP50-95, including subset analysis"""
    
    print(f"\nCalculating performance metrics for {len(image_paths)} images...")
    print(f"Confidence threshold: {confidence_threshold}")
    
    all_predictions = []
    all_ground_truths = []
    
    # Track subsets
    calipers_predictions = []
    calipers_ground_truths = []
    calipers_images = set()
    
    multi_gt_predictions = []
    multi_gt_ground_truths = []
    multi_gt_images = set()
    
    single_lesion_predictions = []
    single_lesion_ground_truths = []
    single_lesion_images = set()

    single_node_predictions = []
    single_node_ground_truths = []
    single_node_images = set()
    
    processed_count = 0
    images_cropped = 0
    images_with_zero_gt = 0
    images_with_one_gt = 0
    images_with_multi_gt = 0
    images_with_calipers = 0
    
    # Process images
    for image_path in tqdm(image_paths, desc="Processing images"):
        image_stem = image_path.stem
        
        try:
            gt_data = ground_truth_data.get(image_stem, {})
            gt_annotations = gt_data.get('annotations', [])
            crop_info = gt_data.get('crop_info', {})
            
            # Track calipers info
            has_calipers = crop_info.get('has_calipers', False)
            if has_calipers:
                images_with_calipers += 1
                calipers_images.add(image_stem)
            
            # Track ground truth statistics
            num_annotations = len(gt_annotations)
            if num_annotations == 0:
                images_with_zero_gt += 1
            elif num_annotations == 1:
                images_with_one_gt += 1
                # NEW: Track single annotation by class
                annotation_class = gt_annotations[0]['original_class']
                if annotation_class == 'lesion':
                    single_lesion_images.add(image_stem)
                elif annotation_class == 'axillary_node':
                    single_node_images.add(image_stem)
            else:  # > 1
                images_with_multi_gt += 1
                multi_gt_images.add(image_stem)
                
            # Add ground truth to overall
            for gt in gt_annotations:
                bbox_xyxy = convert_absolute_to_xyxy(gt['bbox_absolute'])
                gt_entry = {
                    'bbox': bbox_xyxy,
                    'class_id': 0,
                    'image_name': image_stem,
                    'original_class': gt['original_class']
                }
                all_ground_truths.append(gt_entry)
                
                # Add to subset collections
                if has_calipers:
                    calipers_ground_truths.append(gt_entry)
                if num_annotations > 1:
                    multi_gt_ground_truths.append(gt_entry)
                # NEW: Add to single annotation subsets
                if image_stem in single_lesion_images:
                    single_lesion_ground_truths.append(gt_entry)
                if image_stem in single_node_images:
                    single_node_ground_truths.append(gt_entry)
            
            # Load and crop image
            image = read_image(str(image_path))
            if image is None:
                continue
            
            if crop_info.get('crop_w') and crop_info.get('crop_h'):
                crop_x = crop_info.get('crop_x', 0) or 0
                crop_y = crop_info.get('crop_y', 0) or 0
                crop_w = crop_info['crop_w']
                crop_h = crop_info['crop_h']
                image = crop_image(image, crop_x, crop_y, crop_w, crop_h)
                images_cropped += 1
            
            # Run inference
            results = model(image, conf=confidence_threshold, verbose=False)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    pred_entry = {
                        'bbox': (x1, y1, x2, y2),
                        'class_id': 0,
                        'confidence': conf,
                        'image_name': image_stem
                    }
                    all_predictions.append(pred_entry)
                    
                    # Add to subset collections
                    if has_calipers:
                        calipers_predictions.append(pred_entry)
                    if num_annotations > 1:
                        multi_gt_predictions.append(pred_entry)
                    # NEW: Add to single annotation subsets
                    if image_stem in single_lesion_images:
                        single_lesion_predictions.append(pred_entry)
                    if image_stem in single_node_images:
                        single_node_predictions.append(pred_entry)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Helper function to calculate and print metrics for a subset
    def calculate_and_print_metrics(predictions, ground_truths, subset_name, subset_images_count=None):
        if not ground_truths:
            print(f"\n{subset_name.upper()}:")
            print("  No ground truth annotations found")
            return
            
        print(f"\n{subset_name.upper()}:")
        if subset_images_count:
            print(f"  Images: {subset_images_count}")
        print(f"  Ground Truth Count: {len(ground_truths)}")
        print(f"  Predictions Count: {len(predictions)}")
        
        # mAP metrics
        map30_95, _ = calculate_map30_95(predictions, ground_truths)
        map50_95, _ = calculate_map50_95(predictions, ground_truths)
        print(f"  mAP30-95: {map30_95:.3f}")
        print(f"  mAP50-95: {map50_95:.3f}")
        
        # Standard metrics at IoU 0.3
        matched_preds, matched_gts, _ = match_predictions_to_gt(predictions, ground_truths, 0.3)
        tp = len(matched_preds)
        fp = len(predictions) - tp
        fn = len(ground_truths) - tp
        
        precision = tp / len(predictions) if len(predictions) > 0 else 0
        recall = tp / len(ground_truths) if len(ground_truths) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  IoU 0.3 Metrics:")
        tp_pct = (tp/len(ground_truths)*100) if len(ground_truths) > 0 else 0
        fp_pct = (fp/len(predictions)*100) if len(predictions) > 0 else 0
        fn_pct = (fn/len(ground_truths)*100) if len(ground_truths) > 0 else 0
        print(f"    TP: {tp} ({tp_pct:.1f}%), FP: {fp} ({fp_pct:.1f}%), FN: {fn} ({fn_pct:.1f}%)")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")
        
        # AP at common IoU thresholds
        ap30 = calculate_ap_at_iou(predictions, ground_truths, 0.3)
        ap50 = calculate_ap_at_iou(predictions, ground_truths, 0.5)
        ap75 = calculate_ap_at_iou(predictions, ground_truths, 0.75)
        print(f"    AP30: {ap30:.3f}, AP50: {ap50:.3f}, AP75: {ap75:.3f}")
    
    # Print overall results summary
    print(f"\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Processed: {processed_count} images, {images_cropped} cropped")
    print(f"Predictions: {len(all_predictions)}, Ground Truth: {len(all_ground_truths)}")
    
    # Calipers information
    calipers_percentage = (images_with_calipers / processed_count * 100) if processed_count > 0 else 0
    print(f"Images with calipers: {images_with_calipers} ({calipers_percentage:.1f}%)")
    
    # Ground truth distribution
    print(f"\nGROUND TRUTH DISTRIBUTION:")
    total_images = images_with_zero_gt + images_with_one_gt + images_with_multi_gt
    zero_gt_percentage = (images_with_zero_gt / total_images * 100) if total_images > 0 else 0
    one_gt_percentage = (images_with_one_gt / total_images * 100) if total_images > 0 else 0
    multi_gt_percentage = (images_with_multi_gt / total_images * 100) if total_images > 0 else 0

    print(f"  Images with 0 annotations:  {images_with_zero_gt} ({zero_gt_percentage:.1f}%)")
    print(f"  Images with 1 annotation:   {images_with_one_gt} ({one_gt_percentage:.1f}%)")
    print(f"  Images with >1 annotations: {images_with_multi_gt} ({multi_gt_percentage:.1f}%)")
    print(f"  Total images: {total_images}")

    positive_images = images_with_one_gt + images_with_multi_gt
    avg_annotations_per_positive = len(all_ground_truths) / positive_images if positive_images > 0 else 0
    print(f"  Average annotations per positive image: {avg_annotations_per_positive:.1f}")

    # Calculate and display metrics for all three subsets
    calculate_and_print_metrics(all_predictions, all_ground_truths, "Overall Performance (Per lesion)")
    calculate_and_print_metrics(calipers_predictions, calipers_ground_truths, "Calipers Only Performance", len(calipers_images))
    calculate_and_print_metrics(multi_gt_predictions, multi_gt_ground_truths, "Multi-annotation Images Only Performance", len(multi_gt_images))
    calculate_and_print_metrics(single_lesion_predictions, single_lesion_ground_truths, "Single-Lesion Images Only Performance", len(single_lesion_images))
    calculate_and_print_metrics(single_node_predictions, single_node_ground_truths, "Single-Node Images Only Performance", len(single_node_images))
    
    # Generate confusion matrices and visual examples only for overall performance
    save_confusion_matrices(all_predictions, all_ground_truths, current_dir, iou_threshold=0.3)
    save_visual_examples(all_predictions, all_ground_truths, ground_truth_data, current_dir, max_examples=10)
    
    return all_predictions, all_ground_truths

def main():
    """Main function to run inference"""
    # Determine storage client
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    # ========== CONFIGURATION ==========
    CONFIDENCE_THRESHOLD = 0.3
    MAX_IMAGES = None  # Set to None to process all images, or specify a number
    # ===================================
    
    print("Loading model...")
    model = load_model()
    
    # Get class names from the model
    class_names = model.names  # This returns a dict like {0: 'lesion', 1: 'axillary node'}
    print(f"Model classes: {class_names}")
    
    # Load ground truth data from CSV with crop adjustments
    print("Loading ground truth data with crop adjustments...")
    ground_truth_data = load_ground_truth_from_csv()
    
    if not ground_truth_data:
        print("No ground truth data found! Check the CSV file paths.")
        return
    
    # Get all images (limited by MAX_IMAGES)
    image_paths = get_images_with_ground_truth(ground_truth_data, MAX_IMAGES)
    
    if not image_paths:
        print("No images found!")
        return
        
    # Calculate performance metrics with cropping
    all_predictions, all_ground_truths = calculate_metrics_only(
        model, image_paths, class_names, ground_truth_data, CONFIDENCE_THRESHOLD
    )

if __name__ == "__main__":
    main()