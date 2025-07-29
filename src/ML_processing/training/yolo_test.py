from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import yaml

def load_model_and_data(base_dir="C:/Users/Tristan/Desktop/Yolo2/", data_yaml_path="C:/Users/Tristan/Desktop/Yolo2/data.yaml"):
    """Load the trained model and dataset information"""
    
    base_path = Path(base_dir)
    
    # Look for the trained model in various possible locations
    possible_model_paths = [
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

def run_full_validation(model, data_yaml_path="C:/Users/Tristan/Desktop/Yolo2/data.yaml"):
    """Run validation on entire validation dataset and get mAP scores"""
    
    print("\n" + "="*60)
    print("RUNNING FULL VALIDATION ON ENTIRE DATASET")
    print("="*60)
    
    try:
        # Run validation on entire validation set
        results = model.val(
            data=data_yaml_path,
            save_json=True,  # Save detailed results
            plots=True,      # Generate validation plots
            verbose=True     # Show detailed output
        )
        
        # Extract key metrics
        metrics = results.results_dict
        
        print(f"\n" + "="*50)
        print("VALIDATION RESULTS - ENTIRE DATASET")
        print("="*50)
        
        # Main mAP metrics
        if 'metrics/mAP50(B)' in metrics:
            map50 = metrics['metrics/mAP50(B)']
            print(f"mAP@0.50         : {map50:.4f} ({map50*100:.2f}%)")
        
        if 'metrics/mAP50-95(B)' in metrics:
            map50_95 = metrics['metrics/mAP50-95(B)']
            print(f"mAP@0.50:0.95    : {map50_95:.4f} ({map50_95*100:.2f}%)")
        
        # Additional metrics
        if 'metrics/precision(B)' in metrics:
            precision = metrics['metrics/precision(B)']
            print(f"Precision        : {precision:.4f} ({precision*100:.2f}%)")
        
        if 'metrics/recall(B)' in metrics:
            recall = metrics['metrics/recall(B)']
            print(f"Recall           : {recall:.4f} ({recall*100:.2f}%)")
        
        # F1 Score calculation
        if 'metrics/precision(B)' in metrics and 'metrics/recall(B)' in metrics:
            precision = metrics['metrics/precision(B)']
            recall = metrics['metrics/recall(B)']
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"F1 Score         : {f1:.4f} ({f1*100:.2f}%)")
        
        # Speed metrics
        if hasattr(results, 'speed'):
            speed = results.speed
            print(f"\nInference Speed:")
            if 'preprocess' in speed:
                print(f"  Preprocess     : {speed['preprocess']:.2f}ms")
            if 'inference' in speed:
                print(f"  Inference      : {speed['inference']:.2f}ms") 
            if 'postprocess' in speed:
                print(f"  Postprocess    : {speed['postprocess']:.2f}ms")
        
        print("\n" + "="*50)
        
        return results
        
    except Exception as e:
        print(f"Error during validation: {e}")
        print("This might be due to:")
        print("1. Incorrect data.yaml path or format")
        print("2. Missing ground truth labels") 
        print("3. Mismatched image and label files")
        return None

def get_validation_images(data_config, base_dir="C:/Users/Tristan/Desktop/Yolo2/", num_images=20):
    """Get random validation images for visualization"""
    
    if data_config is None:
        # Fallback: look for images in common validation directories
        base_path = Path(base_dir)
        possible_val_dirs = [
            base_path / "valid" / "images",
            base_path / "val" / "images", 
            base_path / "validation" / "images",
            base_path / "images" / "val",
            base_path / "images" / "valid",
            base_path / "dataset" / "valid" / "images",
            base_path / "dataset" / "val" / "images"
        ]
        
        val_path = None
        for path in possible_val_dirs:
            if path.exists():
                val_path = path
                print(f"Found validation images at: {val_path}")
                break
        
        if val_path is None:
            print("Could not find validation directory. Looking for any images...")
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(base_dir).glob(f'**/*{ext}')))
                image_files.extend(list(Path(base_dir).glob(f'**/*{ext.upper()}')))
            
            if image_files:
                print(f"Found {len(image_files)} images in directory")
                selected_images = random.sample(image_files, min(num_images, len(image_files)))
                return selected_images
            else:
                print("No images found!")
                return []
    else:
        # Use the validation path from data.yaml
        val_path = Path(data_config['val'])
        if not val_path.is_absolute():
            val_path = Path(base_dir) / val_path
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(val_path.glob(f'*{ext}')))
        image_files.extend(list(val_path.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {val_path}")
        return []
    
    print(f"Total validation images available: {len(image_files)}")
    
    # Randomly select images for visualization
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    return selected_images

def visualize_predictions(model, image_paths, class_names, confidence_threshold=0.5, save_dir="C:/Users/Tristan/Desktop/Yolo/validation_results"):
    """Visualize model predictions on sample validation images"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    fig_size = (20, 25)
    cols = 4
    rows = (len(image_paths) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif len(image_paths) == 1:
        axes = [axes]
    
    for idx, image_path in enumerate(image_paths):
        if rows > 1:
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
        else:
            ax = axes[idx] if len(image_paths) > 1 else axes[0]
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(str(image_path), conf=confidence_threshold)
        
        # Draw predictions
        annotated_image = image_rgb.copy()
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[cls % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                if cls < len(class_names):
                    label = f"{class_names[cls]}: {conf:.2f}"
                else:
                    label = f"Class {cls}: {conf:.2f}"
                    
                # Calculate text size and background
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_image, (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - baseline - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Plot
        ax.imshow(annotated_image)
        ax.set_title(f"{image_path.name}\nDetections: {len(results[0].boxes)}", fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    total_subplots = rows * cols
    for idx in range(len(image_paths), total_subplots):
        if rows > 1:
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        elif len(axes) > len(image_paths):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/validation_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()

def create_confidence_histogram(model, image_paths, save_dir="C:/Users/Tristan/Desktop/Yolo/validation_results"):
    """Create histogram of prediction confidences on sample images"""
    
    all_confidences = []
    
    for image_path in image_paths:
        results = model(str(image_path))
        if len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            all_confidences.extend(confidences)
    
    if all_confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidences (Sample Images)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/confidence_histogram.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Sample Images Confidence Stats:")
        print(f"  Average confidence: {np.mean(all_confidences):.3f}")
        print(f"  Median confidence: {np.median(all_confidences):.3f}")
        print(f"  Min confidence: {np.min(all_confidences):.3f}")
        print(f"  Max confidence: {np.max(all_confidences):.3f}")
    else:
        print("No detections found in sample images to analyze confidence scores.")

def main():
    """Main function to run validation analysis"""
    
    print("Loading model and dataset configuration...")
    model, data_config = load_model_and_data()
    
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
        class_names = ["lesion"]  # Adjust this based on your classes
    
    print(f"Classes: {class_names}")
    
    # Create results directory
    save_dir = "C:/Users/Tristan/Desktop/Yolo/validation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. RUN FULL VALIDATION ON ENTIRE DATASET FOR mAP SCORES
    print("\n" + "="*60)
    print("STEP 1: FULL DATASET VALIDATION")
    print("="*60)
    
    validation_results = run_full_validation(model, "C:/Users/Tristan/Desktop/Yolo2/data.yaml")
    
    # 2. SAMPLE VISUALIZATION
    print("\n" + "="*60) 
    print("STEP 2: SAMPLE VISUALIZATION (20 images)")
    print("="*60)
    
    # Get sample validation images for visualization
    image_paths = get_validation_images(data_config, num_images=20)
    
    if not image_paths:
        print("No validation images found for visualization!")
        return
        
    print(f"Selected {len(image_paths)} sample images for visualization")
    
    # Visualize sample predictions
    print("\nGenerating sample visualizations...")
    visualize_predictions(model, image_paths, class_names, confidence_threshold=0.3, save_dir=save_dir)
    
    # Create confidence histogram from samples
    create_confidence_histogram(model, image_paths, save_dir)
    
    print(f"\nAll validation results saved to: {save_dir}")
    print("\n" + "="*60)
    print("VALIDATION ANALYSIS COMPLETE!")
    print("="*60)
    print("✓ Full dataset mAP scores calculated")
    print("✓ Sample images visualized")
    print("✓ Confidence analysis completed")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    main()