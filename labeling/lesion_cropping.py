import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import ast
from pathlib import Path
from collections import defaultdict
from tools.storage_adapter import * 
from src.DB_processing.database import DatabaseManager

env = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(env))
from config import CONFIG
import warnings
warnings.filterwarnings('ignore')


def parse_caliper_boxes(caliper_box_str):
    """
    Parse caliper_boxes string into list of bounding boxes.
    
    Args:
        caliper_box_str: String containing caliper box data
        
    Returns:
        list: List of [x_min, y_min, x_max, y_max] boxes
    """
    if pd.isna(caliper_box_str) or caliper_box_str == 'null' or caliper_box_str == '':
        return []
    
    try:
        # Handle semicolon-separated multiple boxes
        if ';' in str(caliper_box_str):
            box_strings = str(caliper_box_str).split(';')
            boxes = []
            for box_str in box_strings:
                box_str = box_str.strip()
                if box_str:
                    box = ast.literal_eval(box_str)
                    if isinstance(box, list) and len(box) == 4:
                        boxes.append(box)
            return boxes
        
        # Handle single box or list
        else:
            box = ast.literal_eval(str(caliper_box_str))
            if isinstance(box, list):
                if len(box) == 4:  # Single box
                    return [box]
                elif len(box) == 0:  # Empty list
                    return []
            return []
            
    except (ValueError, SyntaxError):
        return []

def filter_non_inpainted_images(image_df):
    """
    Filter out images that have data in the inpainted_from column and RGB PhotometricInterpretation.
    
    Args:
        image_df: DataFrame with image data
        
    Returns:
        DataFrame: Filtered dataframe excluding inpainted images and RGB images
    """
    initial_count = len(image_df)
    
    # Filter out rows where inpainted_from is not null/empty
    filtered_df = image_df[
        (pd.isna(image_df['inpainted_from'])) | 
        (image_df['inpainted_from'] == '') |
        (image_df['inpainted_from'] == 'null')
    ].copy()
    
    inpainted_excluded = initial_count - len(filtered_df)
    
    # Filter out RGB images
    pre_rgb_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['PhotometricInterpretation'] != 'RGB'].copy()
    rgb_excluded = pre_rgb_count - len(filtered_df)
    
    filtered_count = len(filtered_df)
    total_excluded = initial_count - filtered_count
    
    print(f"Image filtering results:")
    print(f"  Initial images: {initial_count}")
    print(f"  Excluded inpainted images: {inpainted_excluded}")
    print(f"  Excluded RGB images: {rgb_excluded}")
    print(f"  Remaining images: {filtered_count}")
    print(f"  Total exclusion rate: {total_excluded/initial_count*100:.1f}%")
    
    return filtered_df

def create_balanced_cancer_dataset(lesion_df, image_df, target_count=25, max_images_per_accession=25):
    """
    Create a balanced dataset of accession numbers with target_count examples per cancer type.
    
    Args:
        lesion_df: DataFrame with lesion data
        image_df: DataFrame with image data
        target_count: Target number of accession numbers per cancer type (default: 25)
        max_images_per_accession: Maximum number of images per accession to allow (default: 25)
    
    Returns:
        tuple: (set of selected accession numbers, dict of counts per cancer type)
    """
    print(f"Creating balanced dataset with {target_count} examples per cancer type...")
    print(f"Filtering out accessions with more than {max_images_per_accession} images...")
    
    # Get available accession numbers from image data
    available_accessions = set(image_df['Accession_Number'].unique())
    
    # Count images per accession
    accession_image_counts = image_df.groupby('Accession_Number').size()
    
    # Filter out accessions with more than max_images_per_accession images
    accessions_within_limit = set(accession_image_counts[accession_image_counts <= max_images_per_accession].index)
    
    # Show filtering statistics
    original_count = len(available_accessions)
    filtered_count = len(accessions_within_limit)
    excluded_count = original_count - filtered_count
    
    print(f"Accession filtering results:")
    print(f"  Original accessions: {original_count}")
    print(f"  Excluded accessions (>{max_images_per_accession} images): {excluded_count}")
    print(f"  Remaining accessions: {filtered_count}")
    print(f"  Exclusion rate: {excluded_count/original_count*100:.1f}%")
    
    # Update available accessions to only include those within the image limit
    available_accessions = accessions_within_limit
    
    # Filter lesion data to only include accession numbers that exist in filtered image data
    lesion_df_filtered = lesion_df[lesion_df['ACCESSION_NUMBER'].isin(available_accessions)]
    
    # Group by cancer type and collect accession numbers
    cancer_type_accessions = defaultdict(set)
    
    for _, row in lesion_df_filtered.iterrows():
        cancer_type = row['cancer_type']
        accession_num = row['ACCESSION_NUMBER']
        
        # Skip if cancer_type is NaN or empty
        if pd.isna(cancer_type) or cancer_type == '':
            continue
            
        cancer_type_accessions[cancer_type].add(accession_num)
    
    # Select up to target_count accession numbers per cancer type
    selected_accessions = set()
    cancer_type_counts = {}
    
    print(f"\nSelecting up to {target_count} accession numbers per cancer type:")
    print("-" * 60)
    
    for cancer_type, accessions in cancer_type_accessions.items():
        available_count = len(accessions)
        selected_count = min(target_count, available_count)
        
        # Randomly sample the required number
        if selected_count > 0:
            selected_for_type = np.random.choice(
                list(accessions), 
                size=selected_count, 
                replace=False
            )
            selected_accessions.update(selected_for_type)
            cancer_type_counts[cancer_type] = selected_count
            
            print(f"{cancer_type:<30}: {selected_count:>3} selected (from {available_count} available)")
        else:
            cancer_type_counts[cancer_type] = 0
            print(f"{cancer_type:<30}: {0:>3} selected (from {available_count} available)")
    
    print("-" * 60)
    print(f"Total unique accession numbers selected: {len(selected_accessions)}")
    
    return selected_accessions, cancer_type_counts

def save_image_with_caliper_boxes(image_path, boxes, output_path):
    """
    Save image with caliper boxes drawn on it.
    
    Args:
        image_path: Path to the original image file
        boxes: List of bounding boxes [x_min, y_min, x_max, y_max]
        output_path: Path where to save the annotated image
    """
    try:
        # Load image
        img = read_image(image_path, use_pil=True)
        
        # Convert to RGB to ensure red color is visible
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Define colors for different boxes
        color = 'red'
        
        # Draw each box
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            
            # Draw rectangle with thick border
            line_width = 3
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

        # Save the annotated image as RGB
        img.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def process_and_save_annotated_images(image_df, selected_accessions, image_base_path, output_dir=f"{env}/output_images"):
    """
    Process images for selected accessions and save with caliper boxes drawn on them.
    
    Args:
        image_df: DataFrame with image data
        selected_accessions: Set of selected accession numbers
        image_base_path: Base path where images are stored
        output_dir: Output directory for annotated images
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to selected accessions
    filtered_df = image_df[image_df['Accession_Number'].isin(selected_accessions)].copy()
    
    print(f"Processing {len(filtered_df)} images from {len(selected_accessions)} accessions...")
    
    processed_count = 0
    saved_count = 0
    
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        image_name = row['ImageName']
        caliper_boxes_str = row['caliper_boxes']
        
        # Parse caliper boxes
        boxes = parse_caliper_boxes(caliper_boxes_str)
        
        # Construct image path
        image_path = os.path.join(image_base_path, image_name)
        
        # Use exact same filename as original
        output_filename = image_name
        output_path = os.path.join(output_dir, output_filename)
        
        # If there are boxes, save annotated image; otherwise, save original image
        if len(boxes) > 0:
            # Save annotated image with boxes
            if save_image_with_caliper_boxes(image_path, boxes, output_path):
                saved_count += 1
        else:
            # Save original image without annotation
            try:
                img = read_image(image_path, use_pil=True)
                img.save(output_path)
                saved_count += 1
            except Exception as e:
                print(f"Error copying image {image_path}: {e}")
            
        processed_count += 1
    
    print(f"\nCompleted! Processed {processed_count} images, saved {saved_count} images to {output_dir}")

def analyze_caliper_boxes(image_df, selected_accessions):
    """
    Analyze the caliper_boxes data for selected accessions only.
    
    Args:
        image_df: DataFrame with image data
        selected_accessions: Set of selected accession numbers
    """
    print("Analyzing caliper_boxes data for selected accessions...")
    print("-" * 60)
    
    # Filter to selected accessions only
    filtered_df = image_df[image_df['Accession_Number'].isin(selected_accessions)]
    total_rows = len(filtered_df)
    
    # Count different types
    null_count = 0
    empty_count = 0
    single_box_count = 0
    multi_box_count = 0
    
    for idx, row in filtered_df.iterrows():
        caliper_boxes_str = row['caliper_boxes']
        boxes = parse_caliper_boxes(caliper_boxes_str)
        
        if pd.isna(caliper_boxes_str) or caliper_boxes_str == 'null':
            null_count += 1
        elif len(boxes) == 0:
            empty_count += 1
        elif len(boxes) == 1:
            single_box_count += 1
        elif len(boxes) > 1:
            multi_box_count += 1
    
    images_with_boxes = single_box_count + multi_box_count
    
    print(f"Selected accessions: {len(selected_accessions)}")
    print(f"Total image rows: {total_rows}")
    print(f"Null/missing: {null_count} ({null_count/total_rows*100:.1f}%)")
    print(f"Empty boxes: {empty_count} ({empty_count/total_rows*100:.1f}%)")
    print(f"Single box: {single_box_count} ({single_box_count/total_rows*100:.1f}%)")
    print(f"Multiple boxes: {multi_box_count} ({multi_box_count/total_rows*100:.1f}%)")
    print(f"Images with boxes: {images_with_boxes} ({images_with_boxes/total_rows*100:.1f}%)")

def save_results(selected_accessions, cancer_type_counts, image_df):

    # Filter image data to selected accessions and get unique records
    selected_image_data = image_df[image_df['Accession_Number'].isin(selected_accessions)].copy()
    
    # Get unique combinations of Accession_Number, ImageName, and DicomHash
    unique_selected = selected_image_data[['Accession_Number', 'ImageName', 'DicomHash']].drop_duplicates()
    
    # Save selected accession numbers with image details
    accession_file = f"yolo_accessions.csv"
    unique_selected.to_csv(accession_file, index=False)

    # Save cancer type summary
    summary_df = pd.DataFrame([
        {'cancer_type': cancer_type, 'count': count} 
        for cancer_type, count in cancer_type_counts.items()
    ])
    summary_df = summary_df.sort_values('count', ascending=False)
    summary_file = f"yolo_cancer_type.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Cancer type summary saved to: {summary_file}")

def main():
    
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    database_path = "Databases/database_2025_8_11_main"
    image_base_path = f"{database_path}/images"
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Loading data from database...")
    with DatabaseManager() as db:
        lesion_df = db.get_lesions_dataframe()
        image_df = db.get_images_dataframe()
        
        # Filter out inpainted images
        print("\nFiltering out inpainted images...")
        image_df = filter_non_inpainted_images(image_df)
        
        # Create balanced dataset with examples per cancer type, avoiding accessions with >25 images
        selected_accessions, cancer_type_counts = create_balanced_cancer_dataset(
            lesion_df, image_df, target_count=4, max_images_per_accession=25
        )
        
        # Save the balanced selection results
        save_results(selected_accessions, cancer_type_counts, image_df)
        
        # Analyze caliper boxes for selected accessions only
        analyze_caliper_boxes(image_df, selected_accessions)
        
        # Save annotated images with caliper boxes drawn on them
        process_and_save_annotated_images(image_df, selected_accessions, image_base_path)

if __name__ == "__main__":
    main()