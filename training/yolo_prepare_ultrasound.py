import pandas as pd
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

def create_yolo_dataset(csv_path, image_dir, output_dir, train_split=0.8, random_seed=42, class_id=0, min_size=200, num_workers=8):
    """
    Convert CSV with bounding box data to YOLO dataset format
    
    Args:
        csv_path: Path to CSV file
        image_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (default 0.8)
        random_seed: Random seed for consistent splits
        class_id: Class ID for all annotations (default 0)
        min_size: Minimum crop width/height in pixels (default 200)
        num_workers: Number of threads for parallel processing (default 8)
    """
    
    # Set random seed for reproducible splits
    np.random.seed(random_seed)
    
    # Read CSV
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Analyze crop sizes before filtering
    small_crops = df[(df['crop_w'] < min_size) | (df['crop_h'] < min_size)]
    small_crop_percentage = (len(small_crops) / len(df)) * 100
    
    print(f"\nCrop size analysis:")
    print(f"Rows with crop width < {min_size}px: {len(df[df['crop_w'] < min_size])}")
    print(f"Rows with crop height < {min_size}px: {len(df[df['crop_h'] < min_size])}")
    print(f"Rows with either width OR height < {min_size}px: {len(small_crops)} ({small_crop_percentage:.1f}%)")
    
    # Filter out small crops
    df_filtered = df[(df['crop_w'] >= min_size) & (df['crop_h'] >= min_size)]
    print(f"Rows after filtering: {len(df_filtered)} ({((len(df_filtered)/len(df))*100):.1f}% retained)")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Get unique images for train/val split (from filtered data)
    unique_images = df_filtered['ImageName'].unique()
    train_images, val_images = train_test_split(
        unique_images, 
        train_size=train_split, 
        random_state=random_seed
    )

    def process_single_image(image_name, split_name, annotations_list, image_dir, output_dir, class_id):
        """Ultra-optimized version with pre-converted annotation data"""
        try:
            src_image_path = os.path.join(image_dir, image_name)
            dst_image_path = os.path.join(output_dir, 'images', split_name, image_name)
            
            if not os.path.exists(src_image_path):
                return {'status': 'missing', 'annotations': 0}
            
            # Open image once, get dimensions and save
            with Image.open(src_image_path) as img:
                img_width, img_height = img.size
                img.save(dst_image_path)
            
            # Create label file with pre-processed annotations
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(output_dir, 'labels', split_name, label_name)
            
            annotation_count = len(annotations_list)
            with open(label_path, 'w') as f:
                for crop_x, crop_y, crop_w, crop_h in annotations_list:
                    # Convert to YOLO format
                    center_x = (crop_x + crop_w / 2) / img_width
                    center_y = (crop_y + crop_h / 2) / img_height
                    width = crop_w / img_width
                    height = crop_h / img_height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            return {'status': 'success', 'annotations': annotation_count}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'annotations': 0}

    def process_split(image_list, split_name, df_data, image_dir, output_dir, class_id, num_workers):
        """Ultra-optimized processing with minimal pandas operations"""
        print(f"Pre-processing annotations for {split_name}...")
        
        # Convert to dictionary of lists for maximum efficiency
        annotations_dict = {}
        split_data = df_data[df_data['ImageName'].isin(image_list)]
        
        for _, row in split_data.iterrows():
            image_name = row['ImageName']
            if image_name not in annotations_dict:
                annotations_dict[image_name] = []
            annotations_dict[image_name].append((
                row['crop_x'], row['crop_y'], row['crop_w'], row['crop_h']
            ))
        
        processed_count = 0
        missing_count = 0
        error_count = 0
        annotation_count = 0
        
        def process_func(image_name):
            return process_single_image(
                image_name=image_name,
                split_name=split_name,
                annotations_list=annotations_dict.get(image_name, []),
                image_dir=image_dir,
                output_dir=output_dir,
                class_id=class_id
            )
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_image = {executor.submit(process_func, image_name): image_name 
                            for image_name in image_list}
            
            with tqdm(total=len(image_list), desc=f"Processing {split_name}", unit="images") as pbar:
                for future in as_completed(future_to_image):
                    result = future.result()
                    
                    if result['status'] == 'success':
                        processed_count += 1
                        annotation_count += result['annotations']
                    elif result['status'] == 'missing':
                        missing_count += 1
                    else:
                        error_count += 1
                        if error_count <= 5:
                            image_name = future_to_image[future]
                            print(f"Error processing {image_name}: {result.get('error', 'Unknown error')}")
                    
                    pbar.update(1)
        
        return processed_count, missing_count, error_count, annotation_count
    
    # Process train and validation splits - FIXED FUNCTION CALLS
    print(f"\nProcessing splits with {num_workers} workers...")
    train_processed, train_missing, train_errors, train_annotations = process_split(
        train_images, 'train', df_filtered, image_dir, output_dir, class_id, num_workers
    )
    val_processed, val_missing, val_errors, val_annotations = process_split(
        val_images, 'val', df_filtered, image_dir, output_dir, class_id, num_workers
    )
    
    # Create dataset.yaml file for YOLO
    yaml_content = f"""# Dataset configuration for YOLO
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names: ['object']  # Change this to your actual class name
"""
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # Print comprehensive summary
    total_expected = len(unique_images)
    total_processed = train_processed + val_processed
    total_missing = train_missing + val_missing
    total_errors = train_errors + val_errors
    total_annotations = train_annotations + val_annotations
    
    print(f"\n{'='*60}")
    print(f"Dataset creation completed!")
    print(f"{'='*60}")
    print(f"Original CSV rows: {len(df)}")
    print(f"Small crops filtered out: {len(small_crops)} ({small_crop_percentage:.1f}%)")
    print(f"Rows after filtering: {len(df_filtered)}")
    print(f"")
    print(f"Expected images: {total_expected}")
    print(f"Successfully processed: {total_processed}")
    print(f"Missing images: {total_missing}")
    print(f"Error images: {total_errors}")
    print(f"Success rate: {(total_processed/total_expected)*100:.1f}%")
    print(f"")
    print(f"Training set: {train_processed} images, {train_annotations} annotations ({train_missing} missing, {train_errors} errors)")
    print(f"Validation set: {val_processed} images, {val_annotations} annotations ({val_missing} missing, {val_errors} errors)")
    print(f"Total annotations: {total_annotations}")
    print(f"")
    print(f"Output directory: {output_dir}")
    
    # Additional statistics
    if total_processed > 0:
        avg_annotations_per_image = total_annotations / total_processed
        print(f"Average annotations per image: {avg_annotations_per_image:.2f}")

# Example usage
if __name__ == "__main__":
    # Set your paths here
    csv_path = "F:/Train_data/Calipers2/ImageData.csv"  # Path to your CSV file
    image_dir = "F:/Train_data/Calipers2/caliper_images/"  # Directory containing your images
    output_dir = "C:/Users/Tristan/Desktop/Yolo_ultrasound1/"  # Output directory for YOLO format
    
    # Create the dataset
    create_yolo_dataset(
        csv_path=csv_path,
        image_dir=image_dir,
        output_dir=output_dir,
        train_split=0.8,  # 80% train, 20% validation
        random_seed=42,   # For reproducible splits
        class_id=0,       # Class ID (0 for single class)
        min_size=200,     # Minimum crop width/height in pixels
        num_workers=16     # Number of threads (adjust based on your CPU)
    )