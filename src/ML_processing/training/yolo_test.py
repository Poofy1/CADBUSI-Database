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
        base_path / "yolo11m_lesions11" / "weights" / "best.pt",
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

def get_validation_images(data_config, base_dir="C:/Users/Tristan/Desktop/Yolo/", num_images=20):
    """Get random validation images"""
    
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
            # Look for any images in the base directory
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
    
    # Randomly select images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    return selected_images

def visualize_predictions(model, image_paths, class_names, confidence_threshold=0.5, save_dir="C:/Users/Tristan/Desktop/Yolo/validation_results"):
    """Visualize model predictions on validation images"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    fig_size = (20, 25)  # Adjust based on your needs
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
    """Create histogram of prediction confidences"""
    
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
        plt.title('Distribution of Prediction Confidences on Validation Set')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/confidence_histogram.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Average confidence: {np.mean(all_confidences):.3f}")
        print(f"Median confidence: {np.median(all_confidences):.3f}")
        print(f"Min confidence: {np.min(all_confidences):.3f}")
        print(f"Max confidence: {np.max(all_confidences):.3f}")
    else:
        print("No detections found to analyze confidence scores.")

def detailed_validation_analysis(model, image_paths, class_names, save_dir="C:/Users/Tristan/Desktop/Yolo/validation_results"):
    """Provide detailed analysis of validation performance"""
    
    total_detections = 0
    images_with_detections = 0
    class_counts = {name: 0 for name in class_names} if class_names else {}
    
    print("\n" + "="*60)
    print("DETAILED VALIDATION ANALYSIS")
    print("="*60)
    
    for i, image_path in enumerate(image_paths):
        results = model(str(image_path))
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            images_with_detections += 1
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            print(f"\nImage {i+1}: {image_path.name}")
            print(f"  Detections: {num_detections}")
            
            for cls, conf in zip(classes, confidences):
                if class_names and cls < len(class_names):
                    class_name = class_names[cls]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    print(f"    {class_name}: {conf:.3f}")
                else:
                    print(f"    Class {cls}: {conf:.3f}")
    
    print(f"\n" + "-"*40)
    print("SUMMARY STATISTICS:")
    print(f"  Total images analyzed: {len(image_paths)}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Images without detections: {len(image_paths) - images_with_detections}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections/len(image_paths):.2f}")
    
    if class_counts and total_detections > 0:
        print(f"\nCLASS DISTRIBUTION:")
        for class_name, count in class_counts.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

def save_individual_predictions(model, image_paths, class_names, save_dir="C:/Users/Tristan/Desktop/Yolo/validation_results/individual"):
    """Save individual prediction images"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        # Run prediction
        results = model(str(image_path))
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Save
        output_path = f"{save_dir}/prediction_{i+1:02d}_{image_path.stem}.jpg"
        cv2.imwrite(output_path, annotated_image)
    
    print(f"Individual prediction images saved to: {save_dir}")

def main():
    """Main function to run validation visualization"""
    
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
        # You can manually specify your class names here if needed
        class_names = ["lesion"]  # Adjust this based on your classes
    
    print(f"Classes: {class_names}")
    
    # Get validation images
    print("Selecting validation images...")
    image_paths = get_validation_images(data_config, num_images=20)
    
    if not image_paths:
        print("No validation images found!")
        return
        
    print(f"Selected {len(image_paths)} validation images")
    
    # Create results directory
    save_dir = "C:/Users/Tristan/Desktop/Yolo/validation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Run different visualizations
    print("\nGenerating validation visualizations...")
    
    # 1. Main prediction visualization
    visualize_predictions(model, image_paths, class_names, confidence_threshold=0.3)
    
    # 2. Confidence histogram
    create_confidence_histogram(model, image_paths, save_dir)
    
    # 3. Detailed analysis
    detailed_validation_analysis(model, image_paths, class_names, save_dir)
    
    # 4. Save individual predictions
    save_individual_predictions(model, image_paths, class_names)
    
    print(f"\nAll validation results saved to: {save_dir}")
    print("Validation analysis complete!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    main()