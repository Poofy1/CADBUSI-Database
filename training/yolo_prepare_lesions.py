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

def load_caliper_mapping(caliper_csv_path):
    """Load the caliper CSV and create mapping from raw images to caliper images"""
    print(f"Loading caliper mapping from: {caliper_csv_path}")
    caliper_df = pd.read_csv(caliper_csv_path)
    
    print(f"Caliper CSV columns: {caliper_df.columns.tolist()}")
    print(f"Caliper CSV shape: {caliper_df.shape}")
    
    # Create mapping from Raw_Image to Caliper_Image
    caliper_mapping = {}
    for _, row in caliper_df.iterrows():
        raw_image = row['Raw_Image']
        caliper_image = row['Caliper_Image']
        if pd.notna(raw_image) and pd.notna(caliper_image):
            caliper_mapping[raw_image] = caliper_image
    
    print(f"Loaded {len(caliper_mapping)} raw->caliper image mappings")
    return caliper_mapping

def load_birads1_data(birads1_csv_path):
    """Load BI-RADS 1 images (negative examples with no lesions)"""
    print(f"\nLoading BI-RADS 1 images from: {birads1_csv_path}")
    birads1_df = pd.read_csv(birads1_csv_path)
    
    print(f"BI-RADS 1 CSV columns: {birads1_df.columns.tolist()}")
    print(f"BI-RADS 1 CSV shape: {birads1_df.shape}")
    
    # Rename image_name to ImageName for consistency
    if 'image_name' in birads1_df.columns:
        birads1_df = birads1_df.rename(columns={'image_name': 'ImageName'})
    
    # Filter to only breast area
    birads1_df = birads1_df[birads1_df['area'] != 'axilla'].copy()

    # Filter rows with valid crop coordinates
    birads1_df = birads1_df.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'])

    # Filter out images with dimensions < 200px in either width or height
    print(f"BI-RADS 1 images before dimension filtering: {len(birads1_df)}")
    birads1_df = birads1_df[(birads1_df['crop_w'] >= 200) & (birads1_df['crop_h'] >= 200)].copy()
    print(f"BI-RADS 1 images after filtering out <200px dimensions: {len(birads1_df)}")

    # Add empty parsed_boxes column (no lesions in BI-RADS 1)
    birads1_df['parsed_boxes'] = [[] for _ in range(len(birads1_df))]
    birads1_df['caliper_boxes'] = ''
    birads1_df['is_birads1'] = True

    print(f"Loaded {len(birads1_df)} BI-RADS 1 images (negative examples)")
    
    return birads1_df

def select_caliper_images_from_split(image_names, caliper_mapping, caliper_percentage=0.15, split_name=""):
    """Randomly select a percentage of images to use caliper versions from a specific split"""
    # Only select from images that actually have caliper versions
    available_for_caliper = [img for img in image_names if img in caliper_mapping]
    
    print(f"\n{split_name} - Images available for caliper versions: {len(available_for_caliper)} out of {len(image_names)}")
    
    if len(available_for_caliper) == 0:
        print(f"{split_name} - No images available for caliper versions!")
        return set()
    
    # Calculate how many to select (but don't exceed available)
    requested_count = int(len(available_for_caliper) * caliper_percentage)
    actual_count = min(requested_count, len(available_for_caliper))
    
    print(f"{split_name} - Requested {requested_count} caliper images ({caliper_percentage*100:.1f}% of {len(available_for_caliper)} available)")
    print(f"{split_name} - Will select {actual_count} caliper images")
    
    if actual_count == 0:
        return set()
    
    # Use different random state for each split to avoid correlation
    split_random_state = np.random.RandomState(RANDOM_SEED + hash(split_name) % 1000)
    caliper_selected = split_random_state.choice(
        available_for_caliper, 
        size=actual_count, 
        replace=False
    )
    
    caliper_set = set(caliper_selected)
    print(f"{split_name} - ✅ Selected {len(caliper_set)} images to use caliper versions")
    
    # Show some examples
    if caliper_set:
        print(f"{split_name} - Sample selections:")
        for i, img in enumerate(list(caliper_set)[:3]):
            print(f"  {img} -> {caliper_mapping[img]}")
    
    return caliper_set

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
    """Adjust bounding box coordinates relative to crop"""
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

def create_visualization(cropped_img, adjusted_boxes, image_name, is_caliper=False, is_birads1=False):
    """Create visualization of bounding boxes on cropped image"""
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
    
    # Add image info at the bottom with caliper and BI-RADS indicator
    img_width, img_height = cropped_img.size
    caliper_indicator = " [CALIPER]" if is_caliper else ""
    birads1_indicator = " [BIRADS-1 NEG]" if is_birads1 else ""
    info_text = f"{image_name}{caliper_indicator}{birads1_indicator} | {len(adjusted_boxes)} boxes | {img_width}x{img_height}"
    
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

# Global counters for tracking caliper usage (for multiprocessing visibility)
train_caliper_counter = mp.Value('i', 0)
val_caliper_counter = mp.Value('i', 0)

def process_single_image(args):
    """Process a single image - used for multiprocessing"""
    row_data, source_images_dir, birads1_dir, output_dir, split_type, caliper_mapping, caliper_selected = args
    
    image_name = row_data['ImageName']
    boxes = row_data['parsed_boxes']
    crop_x = int(row_data['crop_x'])
    crop_y = int(row_data['crop_y'])
    crop_w = int(row_data['crop_w'])
    crop_h = int(row_data['crop_h'])
    is_birads1 = row_data.get('is_birads1', False)
    
    # Determine if we should use caliper version (only for non-BI-RADS-1 images)
    use_caliper = False
    actual_image_name = image_name
    
    if not is_birads1 and image_name in caliper_selected:
        use_caliper = True
        if image_name in caliper_mapping:
            actual_image_name = caliper_mapping[image_name]
            # Increment appropriate counter
            if split_type == 'train':
                with train_caliper_counter.get_lock():
                    train_caliper_counter.value += 1
            else:
                with val_caliper_counter.get_lock():
                    val_caliper_counter.value += 1

    # Source image path - check BI-RADS 1 directory first if it's a BI-RADS 1 image
    if is_birads1:
        source_path = os.path.join(birads1_dir, image_name)
    else:
        source_path = os.path.join(source_images_dir, actual_image_name)
    
    # Destination paths (keep original image name for consistency in dataset)
    dest_path = f"{output_dir}/images/{split_type}/{image_name}"
    vis_path = f"{output_dir}/visualized/{split_type}/{image_name}"
    
    # Try to crop and save image
    try:
        # Check if source file exists
        if not os.path.exists(source_path):
            if use_caliper:
                # Try falling back to original image
                fallback_path = os.path.join(source_images_dir, image_name)
                if os.path.exists(fallback_path):
                    source_path = fallback_path
                    use_caliper = False
                else:
                    return False, 0, use_caliper, is_birads1
            else:
                return False, 0, use_caliper, is_birads1
        
        # Open the image
        img = Image.open(source_path)
        
        # Define crop box (left, upper, right, lower)
        crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        
        # Crop the image
        cropped_img = img.crop(crop_box)
        
        # Save the cropped image
        cropped_img.save(dest_path)
        
        # Adjust boxes for crop position before converting to YOLO format
        adjusted_boxes = adjust_boxes_for_crop(boxes, crop_x, crop_y)
        
        # Create and save visualization
        debug = True
        if debug:
            vis_img = create_visualization(cropped_img, adjusted_boxes, image_name, 
                                         is_caliper=use_caliper, is_birads1=is_birads1)
            vis_img.save(vis_path)
        
        # Create label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/{split_type}/{label_name}"
        
        valid_boxes = 0
        with open(label_path, 'w') as f:
            # For BI-RADS 1 images, create empty label file (no boxes)
            if is_birads1:
                pass  # Empty file
            else:
                for adj_box in adjusted_boxes:
                    x1, y1, x2, y2 = adj_box
                    
                    # Only include boxes that are within crop bounds and have positive dimensions
                    if (x1 >= 0 and y1 >= 0 and x2 <= crop_w and y2 <= crop_h and 
                        x2 > x1 and y2 > y1):
                        # Use crop dimensions for normalization (boxes are now relative to crop)
                        yolo_box = convert_to_yolo_format(adj_box, crop_w, crop_h)
                        f.write(' '.join(map(str, yolo_box)) + '\n')
                        valid_boxes += 1
        
        return True, valid_boxes, use_caliper, is_birads1
    except Exception as e:
        print(f"\nError processing image {source_path}: {e}")
        return False, 0, use_caliper, is_birads1

def prepare_yolo_dataset(csv_path, source_images_dir, caliper_csv_path, 
                        birads1_csv_path, birads1_dir,
                        output_dir='yolo_dataset', 
                        train_ratio=0.8, caliper_percentage=0.15, num_workers=None):
    """Prepare YOLO dataset from CSV files including BI-RADS 1 negative examples"""
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count() - 1  # Leave one CPU free
    
    print(f"Using {num_workers} workers for processing")
    
    # Load caliper mapping
    caliper_mapping = load_caliper_mapping(caliper_csv_path)
    
    # Load main CSV with lesion boxes
    print("\nLoading main CSV file with lesion boxes...")
    df = pd.read_csv(csv_path)
    
    # Filter to only include breast area
    print(f"Total images before area filtering: {len(df)}")
    df = df[df['area'] != 'axilla'].copy()
    print(f"Images after filtering for area='breast': {len(df)}")

    # Filter rows with valid caliper_boxes
    df['parsed_boxes'] = df['caliper_boxes'].apply(parse_caliper_boxes)
    df_valid = df[df['parsed_boxes'].apply(len) > 0].copy()

    # Also filter rows with valid crop coordinates
    df_valid = df_valid.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'])

    # Filter out images with dimensions < 200px in either width or height
    print(f"Total images before dimension filtering: {len(df_valid)}")
    df_valid = df_valid[(df_valid['crop_w'] >= 200) & (df_valid['crop_h'] >= 200)].copy()
    print(f"Images after filtering out <200px dimensions: {len(df_valid)}")

    df_valid['is_birads1'] = False

    print(f"Total images: {len(df)}")
    print(f"Images with valid bounding boxes and crop coordinates: {len(df_valid)}")
    
    # Load BI-RADS 1 images (negative examples)
    birads1_df = load_birads1_data(birads1_csv_path)
    
    # Combine datasets
    print(f"\n=== COMBINING DATASETS ===")
    print(f"Images with lesions (positive): {len(df_valid)}")
    print(f"BI-RADS 1 images (negative): {len(birads1_df)}")
    
    combined_df = pd.concat([df_valid, birads1_df], ignore_index=True)
    print(f"Total combined images: {len(combined_df)}")
    
    # Calculate class distribution
    positive_images = len(df_valid)
    negative_images = len(birads1_df)
    total_images = len(combined_df)
    print(f"\nClass distribution:")
    print(f"  Positive (with lesions): {positive_images} ({positive_images/total_images*100:.1f}%)")
    print(f"  Negative (BI-RADS 1): {negative_images} ({negative_images/total_images*100:.1f}%)")
    
    # Split into train and validation (DETERMINISTIC - uses fixed random_state)
    print(f"\nSplitting dataset into train ({train_ratio*100:.0f}%) and validation ({(1-train_ratio)*100:.0f}%)...")
    print(f"Using DETERMINISTIC split with random_state={RANDOM_SEED}")
    train_df, val_df = train_test_split(
        combined_df,
        test_size=1-train_ratio,
        random_state=RANDOM_SEED,  # Fixed seed ensures consistent split across runs
        stratify=combined_df['is_birads1'],  # Stratify to maintain positive/negative ratio
        shuffle=True  # Shuffle before split (but deterministic due to random_state)
    )
    
    print(f"Train set: {len(train_df)} images")
    print(f"  - With lesions: {len(train_df[~train_df['is_birads1']])}")
    print(f"  - BI-RADS 1: {len(train_df[train_df['is_birads1']])}")
    print(f"Validation set: {len(val_df)} images")
    print(f"  - With lesions: {len(val_df[~val_df['is_birads1']])}")
    print(f"  - BI-RADS 1: {len(val_df[val_df['is_birads1']])}")
    
    # Select caliper images from each split separately (only for non-BI-RADS-1 images)
    train_non_birads1 = train_df[~train_df['is_birads1']]['ImageName'].tolist()
    val_non_birads1 = val_df[~val_df['is_birads1']]['ImageName'].tolist()
    
    train_caliper_selected = select_caliper_images_from_split(
        train_non_birads1, 
        caliper_mapping,
        caliper_percentage=caliper_percentage,
        split_name="TRAIN"
    )
    
    val_caliper_selected = select_caliper_images_from_split(
        val_non_birads1, 
        caliper_mapping,
        caliper_percentage=caliper_percentage,
        split_name="VAL"
    )
    
    # Combine the selections for processing
    all_caliper_selected = train_caliper_selected.union(val_caliper_selected)
    
    print(f"\n=== CALIPER SELECTION SUMMARY ===")
    print(f"Train caliper images selected: {len(train_caliper_selected)}")
    print(f"Val caliper images selected: {len(val_caliper_selected)}")
    print(f"Total caliper images selected: {len(all_caliper_selected)}")
    
    # Create directory structure
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # Create visualization directories
    os.makedirs(f"{output_dir}/visualized/train", exist_ok=True)
    os.makedirs(f"{output_dir}/visualized/val", exist_ok=True)
    
    # Reset counters
    global train_caliper_counter, val_caliper_counter
    with train_caliper_counter.get_lock():
        train_caliper_counter.value = 0
    with val_caliper_counter.get_lock():
        val_caliper_counter.value = 0
    
    # Prepare arguments for multiprocessing
    train_args = [(row.to_dict(), source_images_dir, birads1_dir, output_dir, 'train', 
                   caliper_mapping, all_caliper_selected) 
                  for _, row in train_df.iterrows()]
    val_args = [(row.to_dict(), source_images_dir, birads1_dir, output_dir, 'val', 
                 caliper_mapping, all_caliper_selected) 
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
    train_caliper_used = sum(result[2] for result in train_results if len(result) > 2)
    train_birads1_processed = sum(result[3] for result in train_results if len(result) > 3)
    
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
    val_caliper_used = sum(result[2] for result in val_results if len(result) > 2)
    val_birads1_processed = sum(result[3] for result in val_results if len(result) > 3)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Successfully processed {train_success}/{len(train_df)} training images")
    print(f"  - Total training boxes: {train_total_boxes}")
    print(f"  - Training caliper images used: {train_caliper_used}/{len(train_caliper_selected)}")
    print(f"  - Training BI-RADS 1 images: {train_birads1_processed}")
    print(f"Successfully processed {val_success}/{len(val_df)} validation images")
    print(f"  - Total validation boxes: {val_total_boxes}")
    print(f"  - Validation caliper images used: {val_caliper_used}/{len(val_caliper_selected)}")
    print(f"  - Validation BI-RADS 1 images: {val_birads1_processed}")
    print(f"🎯 TOTAL CALIPER IMAGES USED: {train_caliper_used + val_caliper_used}")
    print(f"🎯 TOTAL BI-RADS 1 IMAGES: {train_birads1_processed + val_birads1_processed}")
    
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
    
    # Save train/val splits for reference with crop info and caliper info
    train_df_with_info = train_df.copy()
    train_df_with_info['uses_caliper'] = train_df_with_info['ImageName'].isin(all_caliper_selected)
    train_df_with_info[['ImageName', 'caliper_boxes', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 
                        'uses_caliper', 'is_birads1']].to_csv(
        f"{output_dir}/train_split.csv", index=False
    )
    
    val_df_with_info = val_df.copy()
    val_df_with_info['uses_caliper'] = val_df_with_info['ImageName'].isin(all_caliper_selected)
    val_df_with_info[['ImageName', 'caliper_boxes', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 
                      'uses_caliper', 'is_birads1']].to_csv(
        f"{output_dir}/val_split.csv", index=False
    )
    
    print(f"\nDataset prepared in: {output_dir}/")
    print(f"Data configuration saved to: {output_dir}/data.yaml")
    print(f"🎯 CHECK THE VISUALIZATIONS IN: {output_dir}/visualized/")
    print(f"   - Training examples: {output_dir}/visualized/train/")
    print(f"   - Validation examples: {output_dir}/visualized/val/")
    print(f"   - Look for '[CALIPER]' to identify caliper images")
    print(f"   - Look for '[BIRADS-1 NEG]' to identify negative examples")
    
    return train_df, val_df

# Usage
if __name__ == "__main__":
    # Replace with your paths
    csv_path = "F:/Train_data/Calipers2/ImageData.csv"
    source_images_dir = "F:/Train_data/Calipers2/caliper_images/"
    caliper_csv_path = "F:/Train_data/Calipers2/CaliperData.csv"
    
    # BI-RADS 1 negative examples
    birads1_csv_path = "F:/Train_data/Calipers2/birads_1_images.csv"
    birads1_dir = "F:/Train_data/Calipers2/birads1/"
    
    # Prepare dataset
    train_df, val_df = prepare_yolo_dataset(
        csv_path=csv_path,
        source_images_dir=source_images_dir,
        caliper_csv_path=caliper_csv_path,
        birads1_csv_path=birads1_csv_path,
        birads1_dir=birads1_dir,
        output_dir="D:/DATA/CADBUSI/training_sets/Yolo8/",
        train_ratio=0.8,  # 80% train, 20% validation
        caliper_percentage=0.00,  # 10% of images will use caliper versions
        num_workers=None  # Set to None to use all available CPUs - 1
    )