import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import yaml
import ast
import re
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def parse_caliper_boxes(box_string):
    """Parse caliper_boxes string into list of bounding boxes"""
    if pd.isna(box_string) or box_string == '' or box_string == '[]':
        return []
    
    # Handle multiple boxes separated by semicolon
    boxes = []
    try:
        # Split by semicolon if multiple boxes
        box_parts = box_string.split(';')
        for part in box_parts:
            # Extract numbers from string like "[354, 300, 504, 411]"
            numbers = re.findall(r'\d+', part)
            if len(numbers) == 4:
                box = [int(n) for n in numbers]
                boxes.append(box)
    except:
        return []
    
    return boxes

def convert_to_yolo_format(box, img_width, img_height):
    """Convert [x1, y1, x2, y2] to YOLO format [class, x_center, y_center, width, height]"""
    x1, y1, x2, y2 = box
    
    # Calculate center coordinates and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Class 0 for lesion (single class detection)
    return [0, x_center, y_center, width, height]

def adjust_boxes_for_crop(boxes, crop_x, crop_y):
    """Adjust bounding box coordinates relative to crop - THIS WAS THE KEY FIX"""
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Adjust coordinates relative to crop
        adj_x1 = x1 - crop_x
        adj_y1 = y1 - crop_y
        adj_x2 = x2 - crop_x
        adj_y2 = y2 - crop_y
        adjusted_boxes.append([adj_x1, adj_y1, adj_x2, adj_y2])
    return adjusted_boxes

def process_single_image(args):
    """Process a single image - used for multiprocessing"""
    row_data, source_images_dir, output_dir, split_type = args
    
    image_name = row_data['ImageName']
    boxes = row_data['parsed_boxes']
    crop_x = int(row_data['crop_x'])  # Convert to int for consistency
    crop_y = int(row_data['crop_y'])
    crop_w = int(row_data['crop_w'])
    crop_h = int(row_data['crop_h'])
    
    # Source image path
    source_path = os.path.join(source_images_dir, image_name)
    
    # Destination path for cropped image
    dest_path = f"{output_dir}/images/{split_type}/{image_name}"
    
    # Try to crop and save image
    try:
        # Open the image
        img = Image.open(source_path)
        
        # Define crop box (left, upper, right, lower)
        crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        
        # Crop the image
        cropped_img = img.crop(crop_box)
        
        # Save the cropped image
        cropped_img.save(dest_path)
        
        # FIXED: Adjust boxes for crop position before converting to YOLO format
        adjusted_boxes = adjust_boxes_for_crop(boxes, crop_x, crop_y)
        
        # Create label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/{split_type}/{label_name}"
        
        with open(label_path, 'w') as f:
            for adj_box in adjusted_boxes:
                x1, y1, x2, y2 = adj_box
                
                # Only include boxes that are within crop bounds and have positive dimensions
                if (x1 >= 0 and y1 >= 0 and x2 <= crop_w and y2 <= crop_h and 
                    x2 > x1 and y2 > y1):
                    # Use crop dimensions for normalization (boxes are now relative to crop)
                    yolo_box = convert_to_yolo_format(adj_box, crop_w, crop_h)
                    f.write(' '.join(map(str, yolo_box)) + '\n')
        
        return True
    except Exception as e:
        print(f"\nError processing image {source_path}: {e}")
        return False

def prepare_yolo_dataset(csv_path, source_images_dir, output_dir='yolo_dataset', train_ratio=0.8, num_workers=None):
    """Prepare YOLO dataset from CSV file with image cropping using multiprocessing"""
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count() - 1  # Leave one CPU free
    
    print(f"Using {num_workers} workers for processing")
    
    # Load CSV
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid caliper_boxes
    df['parsed_boxes'] = df['caliper_boxes'].apply(parse_caliper_boxes)
    df_valid = df[df['parsed_boxes'].apply(len) > 0].copy()
    
    # Also filter rows with valid crop coordinates
    df_valid = df_valid.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'])
    
    print(f"Total images: {len(df)}")
    print(f"Images with valid bounding boxes and crop coordinates: {len(df_valid)}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df_valid, 
        test_size=1-train_ratio, 
        random_state=RANDOM_SEED
    )
    
    print(f"\nTrain set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Create directory structure
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # Prepare arguments for multiprocessing
    train_args = [(row.to_dict(), source_images_dir, output_dir, 'train') 
                  for _, row in train_df.iterrows()]
    val_args = [(row.to_dict(), source_images_dir, output_dir, 'val') 
                for _, row in val_df.iterrows()]
    
    # Process training data with multiprocessing
    print("\nProcessing training images...")
    with mp.Pool(processes=num_workers) as pool:
        train_results = list(tqdm(
            pool.imap(process_single_image, train_args),
            total=len(train_args),
            desc="Training set"
        ))
    
    train_success = sum(train_results)
    
    # Process validation data with multiprocessing
    print("\nProcessing validation images...")
    with mp.Pool(processes=num_workers) as pool:
        val_results = list(tqdm(
            pool.imap(process_single_image, val_args),
            total=len(val_args),
            desc="Validation set"
        ))
    
    val_success = sum(val_results)
    
    print(f"\nSuccessfully cropped {train_success}/{len(train_df)} training images")
    print(f"Successfully cropped {val_success}/{len(val_df)} validation images")
    
    # Create data.yaml file for YOLO
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # number of classes
        'names': ['lesion']  # class names
    }
    
    with open(f"{output_dir}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Save train/val splits for reference with crop info
    train_df[['ImageName', 'caliper_boxes', 'crop_x', 'crop_y', 'crop_w', 'crop_h']].to_csv(
        f"{output_dir}/train_split.csv", index=False
    )
    val_df[['ImageName', 'caliper_boxes', 'crop_x', 'crop_y', 'crop_w', 'crop_h']].to_csv(
        f"{output_dir}/val_split.csv", index=False
    )
    
    print(f"\nDataset prepared in: {output_dir}/")
    print(f"Data configuration saved to: {output_dir}/data.yaml")
    
    return train_df, val_df

# Usage
if __name__ == "__main__":
    # Replace with your paths
    csv_path = "C:/Users/Tristan/Desktop/Calipers/ImageData.csv"
    source_images_dir = "C:/Users/Tristan/Desktop/Calipers/caliper_images/"  # Directory containing original images
    
    # Prepare dataset
    train_df, val_df = prepare_yolo_dataset(
        csv_path=csv_path,
        source_images_dir=source_images_dir,
        output_dir="C:/Users/Tristan/Desktop/Yolo2/",
        train_ratio=0.8,  # 80% train, 20% validation
        num_workers=None  # Set to None to use all available CPUs - 1
    )