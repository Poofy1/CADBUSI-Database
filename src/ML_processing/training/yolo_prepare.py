import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import yaml
import ast
import re
from PIL import Image, ImageDraw, ImageFont
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

def create_visualization(cropped_img, adjusted_boxes, image_name):
    """
    Create visualization of bounding boxes on cropped image
    
    Args:
        cropped_img: PIL Image object (already cropped)
        adjusted_boxes: List of bounding boxes [x1, y1, x2, y2] relative to cropped image
        image_name: Name of the image for title/reference
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy of the image to draw on
    vis_img = cropped_img.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Define colors for different boxes (cycling through if multiple boxes)
    colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange']
    
    # Draw each bounding box
    for i, box in enumerate(adjusted_boxes):
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds
        img_width, img_height = cropped_img.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Select color (cycle through colors if more boxes than colors)
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add box number label
        label = f"Box {i+1}"
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw label background
        if font:
            bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-20), label, fill='white', font=font)
        else:
            # Fallback without font
            draw.rectangle([x1, y1-15, x1+50, y1], fill=color)
            draw.text((x1+2, y1-12), label, fill='white')
    
    # Add image info at the bottom
    img_width, img_height = cropped_img.size
    info_text = f"{image_name} | {len(adjusted_boxes)} boxes | {img_width}x{img_height}"
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw info background at bottom
    if font:
        text_bbox = draw.textbbox((5, img_height-25), info_text, font=font)
        draw.rectangle([0, img_height-30, img_width, img_height], fill='black', outline='white')
        draw.text((5, img_height-25), info_text, fill='white', font=font)
    else:
        draw.rectangle([0, img_height-20, img_width, img_height], fill='black', outline='white')
        draw.text((5, img_height-15), info_text, fill='white')
    
    return vis_img

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
    
    # Destination paths
    dest_path = f"{output_dir}/images/{split_type}/{image_name}"
    vis_path = f"{output_dir}/visualized/{split_type}/{image_name}"
    
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
        
        # Create and save visualization
        debug = False
        if debug:
            vis_img = create_visualization(cropped_img, adjusted_boxes, image_name)
            vis_img.save(vis_path)
        
        # Create label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/{split_type}/{label_name}"
        
        valid_boxes = 0
        with open(label_path, 'w') as f:
            for adj_box in adjusted_boxes:
                x1, y1, x2, y2 = adj_box
                
                # Only include boxes that are within crop bounds and have positive dimensions
                if (x1 >= 0 and y1 >= 0 and x2 <= crop_w and y2 <= crop_h and 
                    x2 > x1 and y2 > y1):
                    # Use crop dimensions for normalization (boxes are now relative to crop)
                    yolo_box = convert_to_yolo_format(adj_box, crop_w, crop_h)
                    f.write(' '.join(map(str, yolo_box)) + '\n')
                    valid_boxes += 1
        
        return True, valid_boxes
    except Exception as e:
        print(f"\nError processing image {source_path}: {e}")
        return False, 0

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
    
    # Create visualization directories
    os.makedirs(f"{output_dir}/visualized/train", exist_ok=True)
    os.makedirs(f"{output_dir}/visualized/val", exist_ok=True)
    
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
    
    train_success = sum(result[0] for result in train_results)
    train_total_boxes = sum(result[1] for result in train_results)
    
    # Process validation data with multiprocessing
    print("\nProcessing validation images...")
    with mp.Pool(processes=num_workers) as pool:
        val_results = list(tqdm(
            pool.imap(process_single_image, val_args),
            total=len(val_args),
            desc="Validation set"
        ))
    
    val_success = sum(result[0] for result in val_results)
    val_total_boxes = sum(result[1] for result in val_results)
    
    print(f"\nSuccessfully processed {train_success}/{len(train_df)} training images")
    print(f"Total training boxes: {train_total_boxes}")
    print(f"Successfully processed {val_success}/{len(val_df)} validation images")
    print(f"Total validation boxes: {val_total_boxes}")
    
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
    print(f"🎯 CHECK THE VISUALIZATIONS IN: {output_dir}/visualized/")
    print(f"   - Training examples: {output_dir}/visualized/train/")
    print(f"   - Validation examples: {output_dir}/visualized/val/")
    
    return train_df, val_df

# Usage
if __name__ == "__main__":
    # Replace with your paths
    csv_path = "F:/Train_data/Calipers2/ImageData.csv"
    source_images_dir = "F:/Train_data/Calipers2/caliper_images/"  # Directory containing original images
    
    # Prepare dataset
    train_df, val_df = prepare_yolo_dataset(
        csv_path=csv_path,
        source_images_dir=source_images_dir,
        output_dir="C:/Users/Tristan/Desktop/Yolo5/",
        train_ratio=0.8,  # 80% train, 20% validation
        num_workers=None  # Set to None to use all available CPUs - 1
    )