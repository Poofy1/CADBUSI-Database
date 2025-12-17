import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class YOLOEvaluator:
    def __init__(self, model_path, dataset_dir):
        """Initialize evaluator with model and dataset paths"""
        self.model = YOLO(model_path)
        self.dataset_dir = Path(dataset_dir)
        
        # Load validation split info
        self.val_split = pd.read_csv(self.dataset_dir / "val_split.csv")
        
        print(f"Loaded model from: {model_path}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Total validation images: {len(self.val_split)}")
        print(f"  - With labels (positive): {len(self.val_split[~self.val_split['is_birads1']])}")
        print(f"  - Without labels (BI-RADS 1): {len(self.val_split[self.val_split['is_birads1']])}")
    
    def get_image_subsets(self):
        """Get different subsets of validation images"""
        all_images = self.val_split['ImageName'].tolist()
        
        # Images with labels (positive examples)
        with_labels = self.val_split[~self.val_split['is_birads1']]['ImageName'].tolist()
        
        # Images without labels (BI-RADS 1 negative examples)
        without_labels = self.val_split[self.val_split['is_birads1']]['ImageName'].tolist()
        
        return {
            'all': all_images,
            'with_labels': with_labels,
            'without_labels': without_labels
        }
    
    def load_ground_truth_boxes(self, image_name):
        """Load ground truth boxes from YOLO label file"""
        label_name = Path(image_name).stem + '.txt'
        label_path = self.dataset_dir / "labels" / "val" / label_name

        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_center, y_center, width, height = map(float, parts)
                        boxes.append([cls, x_center, y_center, width, height])

        return boxes

    def preload_all_ground_truth(self, image_names):
        """Preload all ground truth boxes at once to avoid repeated file I/O"""
        print("Preloading ground truth labels...")
        gt_cache = {}
        for image_name in tqdm(image_names, desc="Loading GT labels"):
            gt_cache[image_name] = self.load_ground_truth_boxes(image_name)
        return gt_cache
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in [x_center, y_center, width, height] format"""
        # Convert to [x1, y1, x2, y2]
        def to_corners(box):
            # Handle both formats:
            # GT box: [cls, x_center, y_center, width, height] - 5 elements
            # Pred box: [cls, x_center, y_center, width, height, conf] - 6 elements
            # Take elements 1-4 (x_center, y_center, width, height)
            x_center, y_center, width, height = box[1:5]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_corners = to_corners(box1)
        box2_corners = to_corners(box2)
        
        # Calculate intersection
        x1 = max(box1_corners[0], box2_corners[0])
        y1 = max(box1_corners[1], box2_corners[1])
        x2 = min(box1_corners[2], box2_corners[2])
        y2 = min(box1_corners[3], box2_corners[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
        area2 = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_all_images_batched(self, image_names, conf_threshold=0.25, iou_threshold=0.5, batch_size=32):
        """Evaluate all images using batch inference - process each image only ONCE"""
        print(f"\n{'='*60}")
        print(f"Running BATCH EVALUATION on {len(image_names)} images")
        print(f"Batch size: {batch_size}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"{'='*60}\n")

        # Preload all ground truth boxes
        gt_cache = self.preload_all_ground_truth(image_names)

        # Prepare image paths
        image_paths = []
        valid_image_names = []
        for image_name in image_names:
            image_path = self.dataset_dir / "images" / "val" / image_name
            if image_path.exists():
                image_paths.append(str(image_path))
                valid_image_names.append(image_name)
            else:
                print(f"Warning: Image not found: {image_path}")

        # Run batch inference
        print(f"\nRunning inference on {len(image_paths)} images in batches of {batch_size}...")
        all_predictions = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch inference"):
            batch_paths = image_paths[i:i+batch_size]
            batch_names = valid_image_names[i:i+batch_size]

            # Run batch prediction
            batch_results = self.model(
                source=batch_paths,
                conf=conf_threshold,
                verbose=False
            )

            # Extract predictions for each image in the batch
            for j, (result, image_name) in enumerate(zip(batch_results, batch_names)):
                pred_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for k in range(len(boxes)):
                        box_xywh = boxes.xywhn[k].cpu().numpy()
                        conf = boxes.conf[k].cpu().numpy()
                        cls = boxes.cls[k].cpu().numpy()
                        pred_boxes.append([cls, box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3], conf])

                all_predictions[image_name] = pred_boxes

        # Match predictions to ground truth for all images
        print("\nMatching predictions to ground truth...")
        per_image_results = []

        for image_name in tqdm(valid_image_names, desc="Matching boxes"):
            gt_boxes = gt_cache[image_name]
            pred_boxes = all_predictions[image_name]

            # Match predictions to ground truth
            matched_gt = set()
            matched_pred = set()

            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue

                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)

            # Store per-image results
            per_image_results.append({
                'image_name': image_name,
                'gt_boxes': len(gt_boxes),
                'pred_boxes': len(pred_boxes),
                'tp': len(matched_pred),
                'fp': len(pred_boxes) - len(matched_pred),
                'fn': len(gt_boxes) - len(matched_gt),
                'tn': 1 if len(gt_boxes) == 0 and len(pred_boxes) == 0 else 0
            })

        return per_image_results

    def calculate_metrics_from_per_image(self, per_image_results):
        """Calculate metrics from per-image results"""
        results = {
            'true_positives': sum(r['tp'] for r in per_image_results),
            'false_positives': sum(r['fp'] for r in per_image_results),
            'false_negatives': sum(r['fn'] for r in per_image_results),
            'true_negatives': sum(r['tn'] for r in per_image_results),
            'total_gt_boxes': sum(r['gt_boxes'] for r in per_image_results),
            'total_pred_boxes': sum(r['pred_boxes'] for r in per_image_results),
            'images_processed': len(per_image_results),
            'images_with_gt': sum(1 for r in per_image_results if r['gt_boxes'] > 0),
            'images_with_pred': sum(1 for r in per_image_results if r['pred_boxes'] > 0),
            'per_image_results': per_image_results
        }

        metrics = self.calculate_metrics(results)
        return results, metrics
    
    def calculate_metrics(self, results):
        """Calculate precision, recall, F1, etc."""
        tp = results['true_positives']
        fp = results['false_positives']
        fn = results['false_negatives']
        tn = results['true_negatives']
        
        # Box-level metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Image-level metrics
        images_with_fp = sum(1 for r in results['per_image_results'] if r['fp'] > 0)
        images_with_fn = sum(1 for r in results['per_image_results'] if r['fn'] > 0)
        
        # False positive rate (images)
        images_without_gt = results['images_processed'] - results['images_with_gt']
        fpr_image = images_with_fp / images_without_gt if images_without_gt > 0 else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'images_processed': results['images_processed'],
            'images_with_gt': results['images_with_gt'],
            'images_with_pred': results['images_with_pred'],
            'images_with_fp': images_with_fp,
            'images_with_fn': images_with_fn,
            'fpr_image': fpr_image,
            'avg_gt_boxes_per_image': results['total_gt_boxes'] / results['images_processed'] if results['images_processed'] > 0 else 0,
            'avg_pred_boxes_per_image': results['total_pred_boxes'] / results['images_processed'] if results['images_processed'] > 0 else 0,
        }
        
        return metrics
    
    def print_metrics(self, metrics, subset_name):
        """Print metrics in a formatted way"""
        print(f"\n{'='*60}")
        print(f"RESULTS: {subset_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 BOX-LEVEL METRICS:")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1_score']:.4f}")
        
        print(f"\n📦 BOX COUNTS:")
        print(f"  True Positives:   {metrics['true_positives']}")
        print(f"  False Positives:  {metrics['false_positives']}")
        print(f"  False Negatives:  {metrics['false_negatives']}")
        print(f"  True Negatives:   {metrics['true_negatives']}")
        
        print(f"\n🖼️ IMAGE-LEVEL STATS:")
        print(f"  Images processed:         {metrics['images_processed']}")
        print(f"  Images with GT boxes:     {metrics['images_with_gt']}")
        print(f"  Images with predictions:  {metrics['images_with_pred']}")
        print(f"  Images with FP:           {metrics['images_with_fp']}")
        print(f"  Images with FN:           {metrics['images_with_fn']}")
        print(f"  FP rate (image-level):    {metrics['fpr_image']:.4f}")
        
        print(f"\n📈 AVERAGE BOXES PER IMAGE:")
        print(f"  Ground truth:     {metrics['avg_gt_boxes_per_image']:.2f}")
        print(f"  Predictions:      {metrics['avg_pred_boxes_per_image']:.2f}")
        
        print(f"{'='*60}\n")
    
    def save_results(self, all_results, output_dir):
        """Save results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics summary
        metrics_summary = {}
        for subset_name, (results, metrics) in all_results.items():
            metrics_summary[subset_name] = metrics
        
        with open(output_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"✅ Saved metrics summary to: {output_dir / 'metrics_summary.json'}")
        
        # Save per-image results for each subset
        for subset_name, (results, metrics) in all_results.items():
            df = pd.DataFrame(results['per_image_results'])
            csv_path = output_dir / f"{subset_name}_per_image_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved {subset_name} per-image results to: {csv_path}")
        
        # Create visualization
        self.plot_comparison(all_results, output_dir)
    
    def plot_comparison(self, all_results, output_dir):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        subset_names = list(all_results.keys())
        
        # Precision, Recall, F1
        metrics_to_plot = ['precision', 'recall', 'f1_score']
        for i, metric in enumerate(metrics_to_plot):
            values = [all_results[name][1][metric] for name in subset_names]
            axes[0, 0].bar([n.replace('_', '\n') for n in subset_names], values, alpha=0.7)
        
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision, Recall, F1')
        axes[0, 0].legend(metrics_to_plot)
        axes[0, 0].set_ylim([0, 1.0])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Box counts
        tp_values = [all_results[name][1]['true_positives'] for name in subset_names]
        fp_values = [all_results[name][1]['false_positives'] for name in subset_names]
        fn_values = [all_results[name][1]['false_negatives'] for name in subset_names]
        
        x = np.arange(len(subset_names))
        width = 0.25
        
        axes[0, 1].bar(x - width, tp_values, width, label='TP', alpha=0.7, color='green')
        axes[0, 1].bar(x, fp_values, width, label='FP', alpha=0.7, color='red')
        axes[0, 1].bar(x + width, fn_values, width, label='FN', alpha=0.7, color='orange')
        
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('True/False Positives/Negatives')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([n.replace('_', '\n') for n in subset_names])
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Average boxes per image
        avg_gt = [all_results[name][1]['avg_gt_boxes_per_image'] for name in subset_names]
        avg_pred = [all_results[name][1]['avg_pred_boxes_per_image'] for name in subset_names]
        
        x = np.arange(len(subset_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, avg_gt, width, label='Ground Truth', alpha=0.7)
        axes[1, 0].bar(x + width/2, avg_pred, width, label='Predictions', alpha=0.7)
        
        axes[1, 0].set_ylabel('Average Boxes')
        axes[1, 0].set_title('Average Boxes per Image')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([n.replace('_', '\n') for n in subset_names])
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Image-level stats
        imgs_with_gt = [all_results[name][1]['images_with_gt'] for name in subset_names]
        imgs_with_pred = [all_results[name][1]['images_with_pred'] for name in subset_names]
        imgs_with_fp = [all_results[name][1]['images_with_fp'] for name in subset_names]
        
        x = np.arange(len(subset_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, imgs_with_gt, width, label='With GT', alpha=0.7)
        axes[1, 1].bar(x, imgs_with_pred, width, label='With Pred', alpha=0.7)
        axes[1, 1].bar(x + width, imgs_with_fp, width, label='With FP', alpha=0.7, color='red')
        
        axes[1, 1].set_ylabel('Image Count')
        axes[1, 1].set_title('Image-Level Statistics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([n.replace('_', '\n') for n in subset_names])
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plots to: {plot_path}")
        plt.close()
    
    def run_all_evaluations(self, conf_threshold=0.25, iou_threshold=0.5, batch_size=32, output_dir="evaluation_results"):
        """Run all three evaluation scenarios - optimized to process images only ONCE"""
        subsets = self.get_image_subsets()

        # Process ALL validation images ONCE with batch inference
        print("\n" + "="*60)
        print("OPTIMIZED BATCH EVALUATION")
        print("Processing each image ONCE, then filtering by subset")
        print("="*60)

        all_per_image_results = self.evaluate_all_images_batched(
            subsets['all'],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size
        )

        # Create a lookup for quick filtering
        results_by_image = {r['image_name']: r for r in all_per_image_results}

        all_results = {}

        # 1. Calculate metrics for entire validation set
        print("\n" + "="*60)
        print("SUBSET 1: ENTIRE VALIDATION SET")
        print("="*60)
        results, metrics = self.calculate_metrics_from_per_image(all_per_image_results)
        self.print_metrics(metrics, 'ENTIRE VALIDATION SET')
        all_results['entire_val_set'] = (results, metrics)

        # 2. Filter and calculate metrics for images WITHOUT labels
        print("\n" + "="*60)
        print("SUBSET 2: IMAGES WITHOUT LABELS (BI-RADS 1 - NEGATIVE EXAMPLES)")
        print("="*60)
        without_labels_results = [results_by_image[img] for img in subsets['without_labels'] if img in results_by_image]
        results, metrics = self.calculate_metrics_from_per_image(without_labels_results)
        self.print_metrics(metrics, 'WITHOUT LABELS (BI-RADS 1)')
        all_results['without_labels'] = (results, metrics)

        # 3. Filter and calculate metrics for images WITH labels
        print("\n" + "="*60)
        print("SUBSET 3: IMAGES WITH LABELS (POSITIVE EXAMPLES)")
        print("="*60)
        with_labels_results = [results_by_image[img] for img in subsets['with_labels'] if img in results_by_image]
        results, metrics = self.calculate_metrics_from_per_image(with_labels_results)
        self.print_metrics(metrics, 'WITH LABELS (POSITIVE)')
        all_results['with_labels'] = (results, metrics)

        # Save all results
        self.save_results(all_results, output_dir)

        return all_results


def main():
    # Configuration
    MODEL_PATH = "F:/CODE/CADBUSI/CADBUSI-Database/src/ML_processing/models/yolo_lesion_detect.pt"
    DATASET_DIR = "D:/DATA/CADBUSI/training_sets/Yolo8/"
    OUTPUT_DIR = "evaluation_results"

    CONF_THRESHOLD = 0.3  # Confidence threshold for predictions (lower = more detections)
    IOU_THRESHOLD = 0.5    # IoU threshold for matching predictions to GT
    BATCH_SIZE = 32        # Batch size for inference (adjust based on GPU memory)

    # Create evaluator
    evaluator = YOLOEvaluator(MODEL_PATH, DATASET_DIR)

    # Run all evaluations (optimized - processes each image only once!)
    results = evaluator.run_all_evaluations(
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "="*60)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
