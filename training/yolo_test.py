from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import yaml

def load_model_and_data(base_dir="C:/Users/Tristan/Desktop/Yolo_ultrasound1/", data_yaml_path="C:/Users/Tristan/Desktop/Yolo_ultrasound1/data.yaml"):
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

def get_all_images_from_directory(image_dir):
    """Get all images from specified directory"""
    
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
    
    print(f"Total images found: {len(image_files)}")
    return image_files

def process_all_images(model, image_paths, class_names, confidence_threshold=0.3, output_dir="C:/Users/Tristan/Desktop/test_images_output"):
    """Process all images and save annotated outputs"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Output directory: {output_dir}")
    
    total_detections = 0
    processed_count = 0
    
    for idx, image_path in enumerate(image_paths):
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
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[cls % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                
                # Add label
                if cls < len(class_names):
                    label = f"{class_names[cls]}: {conf:.2f}"
                else:
                    label = f"Class {cls}: {conf:.2f}"
                    
                # Calculate text size and background
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_image, (x1, y1 - text_height - baseline - 10), 
                            (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - baseline - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Save annotated image
        output_filename = f"{image_path.stem}_annotated{image_path.suffix}"
        output_path = Path(output_dir) / output_filename
        
        # Convert RGB back to BGR for saving with cv2
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), annotated_bgr)
        
        processed_count += 1
        
        # Progress update every 10 images
        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f"Processed {idx + 1}/{len(image_paths)} images...")
    
    print(f"\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Images processed: {processed_count}")
    print(f"Total detections: {total_detections}")
    if processed_count > 0:
        print(f"Average detections per image: {total_detections/processed_count:.2f}")
    print(f"Output saved to: {output_dir}")

def main():
    """Main function to run inference"""
    
    # ========== CONFIGURATION ==========
    IMAGE_DIR = "C:/Users/Tristan/Desktop/test_images"
    OUTPUT_DIR = "C:/Users/Tristan/Desktop/test_images_output"
    BASE_DIR = "C:/Users/Tristan/Desktop/Yolo_ultrasound1/"
    DATA_YAML_PATH = "C:/Users/Tristan/Desktop/Yolo_ultrasound1/data.yaml"
    CONFIDENCE_THRESHOLD = 0.3
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
    
    # Get all images
    image_paths = get_all_images_from_directory(IMAGE_DIR)
    
    if not image_paths:
        print("No images found!")
        return
        
    # Process all images and save outputs
    process_all_images(model, image_paths, class_names, CONFIDENCE_THRESHOLD, OUTPUT_DIR)
    
    print("DONE!")

if __name__ == "__main__":
    main()