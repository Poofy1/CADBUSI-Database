import os, cv2, ast, datetime, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))


import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from storage_adapter import *
from src.DB_processing.tools import append_audit
from src.DB_export.mask_processing import Mask_Lesions
from src.DB_export.audit_report import generate_audit_report


# Paths
labeled_data_dir = f'{env}/labeled_data_archive/'

def process_single_image(row, image_folder_path, image_output, mask_folder_input, mask_folder_output):
    try:
        image_name = row['ImageName']
        image_path = os.path.join(image_folder_path, image_name)
        mask_path = os.path.join(mask_folder_input, 'mask_' + image_name)
        
        if not file_exists(image_path):
            return f"Error: Image file not found - {image_path}"
            
        image = read_image(image_path)
        if image is None:
            return f"Error: Failed to read image - {image_path}"
            
        mask = None
        if file_exists(mask_path):
            mask = read_image(mask_path)
            if mask is None:
                return f"Warning: Failed to read mask - {mask_path}"
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        try:
            x = int(row['crop_x'])
            y = int(row['crop_y'])
            w = int(row['crop_w'])
            h = int(row['crop_h'])
        except (ValueError, KeyError) as e:
            return f"Error: Invalid crop coordinates for {image_name} - {str(e)}"
            
        if x < 0 or y < 0 or w <= 0 or h <= 0 or \
           x + w > image.shape[1] or y + h > image.shape[0]:
            return f"Error: Invalid crop dimensions for {image_name} - x:{x} y:{y} w:{w} h:{h} image_size:{image.shape}"
        
        try:
            cropped_image = image[y:y+h, x:x+w]
            image_output_path = os.path.join(image_output, image_name)
            save_data(cropped_image, image_output_path)
        except Exception as e:
            return f"Error: Failed to crop/save image {image_name} - {str(e)}"
        
        if mask is not None:
            try:
                cropped_mask = mask[y:y+h, x:x+w]
                mask_output_path = os.path.join(mask_folder_output, 'mask_' + image_name)
                save_data(cropped_mask, mask_output_path)
            except Exception as e:
                return f"Error: Failed to crop/save mask {image_name} - {str(e)}"
        
        return "Success"
        
    except Exception as e:
        return f"Error: Unexpected error processing {row.get('ImageName', 'unknown')} - {str(e)}"

def Crop_Images(df, input_dir, output_dir):
    image_output = f"{output_dir}/images/"
    mask_folder_output = f"{output_dir}/masks/"
    make_dirs(image_output)
    make_dirs(mask_folder_output)
    
    image_folder_path = f"{input_dir}/images/"
    mask_folder_input = f"{labeled_data_dir}/masks/"
    
    results = {'success': 0, 'failed': 0}
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(
                process_single_image, 
                row, 
                image_folder_path, 
                image_output, 
                mask_folder_input, 
                mask_folder_output
            ): index for index, row in df.iterrows()
        }
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "Success":
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    failed_images.append(result)
                pbar.update()
    
    # Only print errors and final statistics
    if failed_images:
        print("\nFailed images and errors:")
        for error in failed_images:
            print(error)
            
    print(f"\nProcessing Complete: Success={results['success']}, Failed={results['failed']}")
    append_audit("export.failed_crops_removed", results['failed'])
    append_audit("export.exported_images", results['success'])
    
                
                
                
                
                
def process_single_video(row, video_folder_path, output_dir):
    # Get the folder name and crop data
    folder_name = row['ImagesPath']
    crop_y = int(row['crop_y'])
    crop_x = int(row['crop_x'])
    crop_w = int(row['crop_w'])
    crop_h = int(row['crop_h'])

    # Get all PNG files in the folder
    input_folder = os.path.join(video_folder_path, folder_name)
    all_images = list_files(input_folder, '.png')
    
    if not all_images:
        return 0  # Return 0 if no images were processed
        
    # Prepare output folder path
    output_folder = os.path.join(output_dir, folder_name)
    make_dirs(output_folder)

    # Process each image
    for image_path in all_images:
        # Get just the filename for the output
        image_name = os.path.basename(image_path)
        
        # Read, crop and save
        image = read_image(image_path)
        if image is not None:
            cropped_image = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            output_path = os.path.join(output_folder, image_name)
            save_data(cropped_image, output_path)
    
    return 1  # Return 1 to indicate a successfully processed video


def Crop_Videos(df, input_dir, output_dir):
    
    video_output = f"{output_dir}/videos/"
    make_dirs(video_output)
    
    video_folder_path = f"{input_dir}/videos/"
    
    processed_videos = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_video, row, video_folder_path, video_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()
                try:
                    result = future.result()
                    processed_videos += result
                except Exception as e:
                    print(f"Error processing video: {e}")
                
    append_audit("export.exported_videos", processed_videos)


def PerformSplit(CONFIG, df):
    val_split = CONFIG["VAL_SPLIT"]
    test_split = CONFIG["TEST_SPLIT"]
    
    if 'Valid' not in df.columns:
        df['Valid'] = None
    
    # Get unique patient IDs
    unique_patients = df['Patient_ID'].unique()
    total_patients = len(unique_patients)
    
    # Calculate how many patients should be in each set
    num_test_patients = int(total_patients * test_split)
    num_val_patients = int(total_patients * val_split)
    num_train_patients = total_patients - num_val_patients - num_test_patients
    
    # Randomly shuffle the patient IDs
    np.random.shuffle(unique_patients)
    
    # Split into three groups
    test_patients = unique_patients[:num_test_patients]
    val_patients = unique_patients[num_test_patients:num_test_patients + num_val_patients]
    
    # Assign split status based on patient ID (0=train, 1=val, 2=test)
    def assign_split(patient_id):
        if patient_id in test_patients:
            return 2  # Test
        elif patient_id in val_patients:
            return 1  # Validation
        else:
            return 0  # Training
    
    df['Valid'] = df['Patient_ID'].apply(assign_split)
    
    # Count samples in each split
    train_samples = (df['Valid'] == 0).sum()
    val_samples = (df['Valid'] == 1).sum()
    test_samples = (df['Valid'] == 2).sum()
    
    # Print split statistics
    print(f"Split completed: {train_samples} training, {val_samples} validation, {test_samples} test samples")
    print(f"Patient split: {num_train_patients} training, {num_val_patients} validation, {num_test_patients} test patients")
    
    # Log statistics to audit file
    append_audit("export.train_patients", num_train_patients)
    append_audit("export.val_patients", num_val_patients)
    append_audit("export.test_patients", num_test_patients)
    
    return df

def create_train_set(breast_data, image_data, lesion_df=None):
    # Join breast_data and image_data
    data = pd.merge(breast_data, image_data, 
                    on=['Accession_Number'], 
                    suffixes=('', '_image_data'))

    # Remove duplicate columns
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)
    
    columns_to_keep = ['Patient_ID', 'Accession_Number', 'Study_Laterality', 
                       'Has_Malignant', 'Has_Benign', 'Valid', 'AGE_AT_EVENT', 
                       'ImageName']
    
    data = data[columns_to_keep]
    
    # Aggregation dictionary
    agg_dict = {
        'Patient_ID': 'first',
        'Study_Laterality': 'first',
        'Has_Malignant': 'first', 
        'Has_Benign': 'first',
        'Valid': 'first',
        'AGE_AT_EVENT': 'first',
        'ImageName': lambda x: list(x)
    }
    
    data = data.reset_index(drop=True)
    data = data.groupby('Accession_Number').agg(agg_dict).reset_index()
    
    # Clean original images
    def clean_list(img_list):
        if not isinstance(img_list, list):
            return []
        return [str(img).strip() for img in img_list if img and str(img).strip()]
    
    data['Images'] = data['ImageName'].apply(clean_list)
    data.drop(['ImageName'], axis=1, inplace=True)
    
    # Add lesion images if lesion_df is provided
    if lesion_df is not None and not lesion_df.empty and 'ImageName' in lesion_df.columns:
        # Group lesions by Accession_Number
        lesion_grouped = lesion_df.groupby('Accession_Number')['ImageName'].apply(list).reset_index()
        lesion_grouped.columns = ['Accession_Number', 'LesionImages']
        
        # Merge lesion data with main data
        data = data.merge(lesion_grouped, on='Accession_Number', how='left')
        
        # Clean lesion images list
        data['LesionImages'] = data['LesionImages'].apply(
            lambda x: clean_list(x) if pd.notna(x) else []
        )
    else:
        # No lesion data available
        data['LesionImages'] = [[] for _ in range(len(data))]
    
    # Filter out rows with empty Images
    data = data.reset_index(drop=True)
    initial_count = len(data)
    has_images_mask = data['Images'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    data = data[has_images_mask].reset_index(drop=True)
    
    removed_count = initial_count - len(data)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with no valid images")
    
    data.drop(['Patient_ID'], axis=1, inplace=True)
    data['ID'] = range(len(data))
    columns = ['ID'] + [col for col in data.columns if col != 'ID']
    data = data[columns]

    return data


def generate_video_images_csv(video_df, root_dir):
    """
    Creates a CSV containing all video image paths.
    """
    video_image_data = []
    
    for _, row in tqdm(video_df.iterrows(), total=len(video_df), desc="Processing video folders"):
        video_folder = row['ImagesPath']
        video_dir = os.path.join(root_dir, 'videos', video_folder).replace('\\', '/')
        
        video_files = list_files(video_dir) 
        
        if video_files:
            video_image_data.append({
                'accession_number': row['Accession_Number'],
                'video_name': video_folder,
                'images': video_files
            })
    
    # Create DataFrame and save
    video_images_df = pd.DataFrame(video_image_data)
    video_images_df['images'] = video_images_df['images'].apply(str)  # Convert lists to string
    return video_images_df

def map_breast_data_to_instances(instance_data, image_df, breast_df, column_mapping):
    """
    Maps data from breast_df to instance_data through image_df using a generic merge approach.
    
    Args:
        instance_data: DataFrame to add columns to
        image_df: DataFrame containing image data with linking keys
        breast_df: DataFrame containing breast data to map from
        column_mapping: dict where key is the column name in breast_df and value is the desired column name in instance_data
    
    Returns:
        Modified instance_data DataFrame
    """
    # Get the columns that actually exist in breast_df
    available_columns = [col for col in column_mapping.keys() if col in breast_df.columns]
    
    if not available_columns:
        return instance_data  # No columns to map
    
    # Prepare the merge columns
    merge_columns = ['Patient_ID', 'Accession_Number', 'Study_Laterality'] + available_columns
    
    # Perform the merge
    merged_data = image_df[['ImageName', 'Patient_ID', 'Accession_Number', 'laterality']].merge(
        breast_df[merge_columns],
        left_on=['Patient_ID', 'Accession_Number', 'laterality'],
        right_on=['Patient_ID', 'Accession_Number', 'Study_Laterality'],
        how='left'
    )
    
    # Create mappings and add to instance_data
    for breast_col, instance_col in column_mapping.items():
        if breast_col in available_columns:
            image_to_value_map = dict(zip(merged_data['ImageName'], merged_data[breast_col]))
            instance_data[instance_col] = instance_data['ImageName'].map(image_to_value_map)
    
    return instance_data
        
def apply_filters(image_df, video_df, breast_df, CONFIG):
    """
    Apply all quality and relevance filters to the image and video data.
    
    Returns:
        tuple: (filtered_image_df, filtered_video_df, filtered_breast_df, audit_stats)
    """
    audit_stats = {}
    
    # Track initial counts
    audit_stats['init_images'] = len(image_df)
    audit_stats['init_videos'] = len(video_df)
    audit_stats['init_breasts'] = len(breast_df)
    
    # Remove bilateral cases
    before_bilateral_count = len(breast_df)
    breast_df = breast_df[breast_df['Study_Laterality'] != 'BILATERAL']
    audit_stats['bilateral_removed'] = before_bilateral_count - len(breast_df)
    
    # Only labeled images
    image_df = image_df[image_df['label'] == True]
    image_df = image_df.drop(['label', 'area'], axis=1)
    
    # Only images/videos with valid patient IDs (from filtered breast_df)
    valid_patient_ids = breast_df['Patient_ID'].unique()
    image_df = image_df[image_df['Patient_ID'].isin(valid_patient_ids)]
    video_df = video_df[video_df['Patient_ID'].isin(valid_patient_ids)]
    
    # Remove bad aspect ratios
    min_aspect_ratio = CONFIG.get('MIN_ASPECT_RATIO', 0.5)
    max_aspect_ratio = CONFIG.get('MAX_ASPECT_RATIO', 4.0)
    
    before_aspect_count = len(image_df)
    image_df = image_df[
        (image_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
        (image_df['crop_aspect_ratio'] <= max_aspect_ratio)
    ]
    audit_stats['bad_aspect_removed'] = before_aspect_count - len(image_df)
    
    # Remove images that are too small
    min_dimension = CONFIG.get('MIN_DIMENSION', 200)
    
    before_dimension_count = len(image_df)
    image_df = image_df[
        (image_df['crop_w'] >= min_dimension) & 
        (image_df['crop_h'] >= min_dimension)
    ]
    audit_stats['too_small_removed'] = before_dimension_count - len(image_df)
    
    # Log audit statistics
    for key, value in audit_stats.items():
        append_audit(f"export.{key}", value)
    
    return image_df, video_df, breast_df


def apply_reject_system(image_df, instance_labels_csv_file, use_reject_system):
    # Merge labelbox data if available
    if file_exists(instance_labels_csv_file):
        labelbox_instance_data = read_csv(instance_labels_csv_file)
        
        instance_data = instance_data.merge(
            labelbox_instance_data, 
            on='DicomHash', 
            how='left',
            suffixes=('', '_labelbox')
        )
        
        if 'ImageName_labelbox' in instance_data.columns:
            instance_data.drop(columns=['ImageName_labelbox'], inplace=True)
        
        instance_data = instance_data[instance_data['DicomHash'].isin(image_df['DicomHash'])]
        
        # Handle reject system
        if 'Reject Image' in instance_data.columns:
            if use_reject_system:
                before_count = len(image_df)
                rejected_images = instance_data[instance_data['Reject Image'] == True][['DicomHash', 'ImageName']]
                instance_data = instance_data[instance_data['Reject Image'] != True]
                image_df = image_df[~image_df['DicomHash'].isin(rejected_images['DicomHash'])]
                
                removed_count = before_count - len(image_df)
                append_audit("export.labeled_reject_removed", removed_count)
                instance_data.drop(columns=['Reject Image'], inplace=True)
            else:
                instance_data['Reject Image'] = instance_data['Reject Image'].fillna(False)
                
                
    return image_df
    
def build_instance_data(image_df, breast_df):
    """
    Build and enrich instance_data with all necessary fields from image_df, breast_df, and labelbox.
    
    Returns:
        tuple: (instance_data, image_df) - image_df may be filtered if reject system is used
    """
    # Create base instance_data
    instance_data = image_df[['DicomHash', 'ImageName']].copy()
    
    # Map breast data columns
    column_mappings = {
        'SYNOPTIC_REPORT': 'SYNOPTIC_REPORT',
        'FINDINGS': 'FINDINGS', 
        'AGE_AT_EVENT': 'Age'
    }
    instance_data = map_breast_data_to_instances(instance_data, image_df, breast_df, column_mappings)
    
    # Add PhysicalDeltaX
    if 'PhysicalDeltaX' in image_df.columns:
        image_to_physicaldelta_map = dict(zip(image_df['ImageName'], image_df['PhysicalDeltaX']))
        instance_data['PhysicalDeltaX'] = instance_data['ImageName'].map(image_to_physicaldelta_map)
    
    # Add image dimensions
    if 'crop_w' in image_df.columns:
        image_to_width_map = dict(zip(image_df['ImageName'], image_df['crop_w'].fillna(0).astype(int)))
        instance_data['image_w'] = instance_data['ImageName'].map(image_to_width_map)
    
    if 'crop_h' in image_df.columns:
        image_to_height_map = dict(zip(image_df['ImageName'], image_df['crop_h'].fillna(0).astype(int)))
        instance_data['image_h'] = instance_data['ImageName'].map(image_to_height_map)
    
    return instance_data

def normalize_dataframes(image_df, video_df, breast_df):
    """
    Standardize data types and formats across all dataframes.
    
    Returns:
        tuple: (image_df, video_df, breast_df)
    """
    # Normalize laterality to uppercase (do once for each df)
    image_df['laterality'] = image_df['laterality'].str.upper()
    video_df['laterality'] = video_df['laterality'].str.upper()
    
    # Standardize Patient_ID as string
    image_df['Patient_ID'] = image_df['Patient_ID'].astype(int).astype(str)
    breast_df['Patient_ID'] = breast_df['Patient_ID'].astype(int).astype(str)
    
    # Standardize Accession_Number as string
    image_df['Accession_Number'] = image_df['Accession_Number'].astype(str)
    breast_df['Accession_Number'] = breast_df['Accession_Number'].astype(str)
    
    return image_df, video_df, breast_df


def add_lesions_to_instance_data(instance_data, lesion_df, image_df, breast_df):
    """
    Add lesion images to instance_data with their specific dimensions.
    Lesions inherit metadata from their source images.
    
    Args:
        instance_data: Existing instance_data DataFrame
        lesion_df: DataFrame with lesion images (has ImageSource, ImageName, Patient_ID, Accession_Number)
        image_df: Original image_df to get source image metadata
        breast_df: Breast data for mapping
    
    Returns:
        Updated instance_data with lesion rows added
    """
    if lesion_df.empty:
        return instance_data
    
    # Create base lesion instance data
    lesion_instances = lesion_df[['ImageName']].copy()
    
    # Add a placeholder DicomHash (lesions don't have their own DicomHash)
    lesion_instances['DicomHash'] = lesion_df['ImageSource']  # Use source image as reference
    
    # Map metadata from source images via ImageSource
    source_to_metadata = {}
    for _, row in image_df.iterrows():
        source_name = row['ImageName']
        source_to_metadata[source_name] = {
            'Patient_ID': row['Patient_ID'],
            'Accession_Number': row['Accession_Number'],
            'laterality': row['laterality'],
            'PhysicalDeltaX': row.get('PhysicalDeltaX', None)
        }
    
    # Get metadata from source images
    for col in ['Patient_ID', 'Accession_Number', 'laterality', 'PhysicalDeltaX']:
        lesion_instances[col] = lesion_df['ImageSource'].map(
            lambda x: source_to_metadata.get(x, {}).get(col)
        )
    
    # Map breast data columns (same as regular images)
    column_mappings = {
        'SYNOPTIC_REPORT': 'SYNOPTIC_REPORT',
        'FINDINGS': 'FINDINGS', 
        'AGE_AT_EVENT': 'Age'
    }
    
    # Get columns that exist in breast_df
    available_columns = [col for col in column_mappings.keys() if col in breast_df.columns]
    
    if available_columns:
        merge_columns = ['Patient_ID', 'Accession_Number', 'Study_Laterality'] + available_columns
        
        merged_data = lesion_instances[['ImageName', 'Patient_ID', 'Accession_Number', 'laterality']].merge(
            breast_df[merge_columns],
            left_on=['Patient_ID', 'Accession_Number', 'laterality'],
            right_on=['Patient_ID', 'Accession_Number', 'Study_Laterality'],
            how='left'
        )
        
        for breast_col, instance_col in column_mappings.items():
            if breast_col in available_columns:
                image_to_value_map = dict(zip(merged_data['ImageName'], merged_data[breast_col]))
                lesion_instances[instance_col] = lesion_instances['ImageName'].map(image_to_value_map)
    
    # Get lesion-specific dimensions from lesion_df
    lesion_name_to_dims = dict(zip(lesion_df['ImageName'], 
                                    zip(lesion_df['crop_w'], lesion_df['crop_h'])))
    
    lesion_instances['image_w'] = lesion_instances['ImageName'].map(
        lambda x: lesion_name_to_dims.get(x, (0, 0))[0]
    )
    lesion_instances['image_h'] = lesion_instances['ImageName'].map(
        lambda x: lesion_name_to_dims.get(x, (0, 0))[1]
    )
    
    # Drop temporary columns
    lesion_instances = lesion_instances.drop(['Patient_ID', 'Accession_Number', 'laterality'], axis=1, errors='ignore')
    
    # Combine with original instance_data
    combined = pd.concat([instance_data, lesion_instances], ignore_index=True)
    
    return combined


def Export_Database(CONFIG, limit = None, reparse_images = True):

    use_reject_system = False # True = removes rejects from training
    output_dir = CONFIG["EXPORT_DIR"]
    parsed_database = CONFIG["DATABASE_DIR"]
    labelbox_path = CONFIG["LABELBOX_LABELS"]
    
    date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_dir = f'{output_dir}/export_{date}/'
    print(f"Exporting dataset to {output_dir}")
    make_dirs(output_dir)
    
    # Save the config to the export location
    save_data(json.dumps(CONFIG, indent=4), os.path.normpath(os.path.join(output_dir, 'export_config.json'))) # Convert CONFIG to a JSON string

    #Dirs
    image_csv_file = os.path.join(parsed_database, 'ImageData.csv')
    breast_csv_file = os.path.join(parsed_database, 'BreastData.csv') 
    video_csv_file = os.path.join(parsed_database, 'VideoData.csv')
    instance_labels_csv_file = os.path.join(labelbox_path, 'InstanceLabels.csv')
    
    # Read data
    video_df = read_csv(video_csv_file)
    image_df = read_csv(image_csv_file)
    breast_df = read_csv(breast_csv_file)
    
    # Apply test subset early if specified
    if limit:
        # Limit breast_df first
        breast_df = breast_df.head(limit)
        
        # Filter image_df and video_df to only include patients from the subset breast_df
        subset_patient_ids = breast_df['Patient_ID'].unique()
        image_df = image_df[image_df['Patient_ID'].isin(subset_patient_ids)]
        video_df = video_df[video_df['Patient_ID'].isin(subset_patient_ids)]
        print(f"Subset data sizes - Breast: {len(breast_df)}, Image: {len(image_df)}, Video: {len(video_df)}")
    
    
        
    # Normalize image_df BEFORE building instance_data
    image_df, video_df, breast_df = normalize_dataframes(image_df, video_df, breast_df)
    image_df, video_df, breast_df = apply_filters(image_df, video_df, breast_df, CONFIG)
    image_df = apply_reject_system(image_df, instance_labels_csv_file, use_reject_system)
    
    # Merge lesion data into image_df using Patient_ID, Accession_Number, and ImageSource
    lesion_df = Mask_Lesions(image_df, parsed_database, output_dir)
    image_df = image_df.copy()
    if not lesion_df.empty:
        image_df_with_lesions = pd.concat([image_df, lesion_df], ignore_index=True)
    
    instance_data = build_instance_data(image_df_with_lesions, breast_df)
    instance_data = add_lesions_to_instance_data(instance_data, lesion_df, image_df_with_lesions, breast_df)

    if reparse_images:   
        # Crop the images for the relevant studies
        Crop_Images(image_df, parsed_database, output_dir)
        Crop_Videos(video_df, parsed_database, output_dir)
    
    #Find Image Counts (Breast Data)
    image_counts = image_df.groupby(['Patient_ID', 'laterality']).size().reset_index(name='Image_Count')
    breast_df = pd.merge(breast_df, image_counts, how='left', left_on=['Patient_ID', 'Study_Laterality'], right_on=['Patient_ID', 'laterality'])
    breast_df = breast_df.drop(['laterality'], axis=1)
    breast_df['Image_Count'] = breast_df['Image_Count'].fillna(0).astype(int)
    
    # Filter out case and breast data that isn't relevant
    initial_breast_count = len(breast_df)
    image_patient_ids = image_df['Patient_ID'].unique()
    breast_df = breast_df[breast_df['Patient_ID'].isin(image_patient_ids)]
    remaining_breast_count = len(breast_df)
    removed_breast_count = initial_breast_count - remaining_breast_count
    append_audit("export.breasts_no_data_removed", removed_breast_count)
    append_audit("export.final_breasts", remaining_breast_count)
        
    # Val split for case data
    breast_df = PerformSplit(CONFIG, breast_df)
    
    # Create trainable csv data
    train_data = create_train_set(breast_df, image_df, lesion_df)
    
    # Create a mapping of (Accession_Number, laterality) to list of ImagesPath
    if not video_df.empty and 'ImagesPath' in video_df.columns:
        video_paths = video_df.groupby(['Accession_Number', 'laterality'])['ImagesPath'].agg(list).to_dict()
        train_data['VideoPaths'] = train_data.apply(lambda row: video_paths.get((row['Accession_Number'], row['Study_Laterality']), []), axis=1)
        
        if reparse_images:  
            video_images_df = generate_video_images_csv(video_df, output_dir)
            save_data(video_images_df, os.path.join(output_dir, 'VideoImages.csv'))
        else:
            video_images_df = None
    else:
        # No video data available
        train_data['VideoPaths'] = [[] for _ in range(len(train_data))]  # Empty lists for all rows
        video_images_df = None
        print("No video data found - VideoPaths set to empty lists")

    # Write the filtered dataframes to CSV files in the output directory
    save_data(breast_df, os.path.join(output_dir, 'BreastData.csv'))
    save_data(video_df, os.path.join(output_dir, 'VideoData.csv'))
    save_data(image_df, os.path.join(output_dir, 'ImageData.csv'))
    save_data(train_data, os.path.join(output_dir, 'TrainData.csv'))
    save_data(lesion_df, os.path.join(output_dir, 'LesionLink.csv'))
    if instance_data is not None:
        save_data(instance_data, os.path.join(output_dir, 'InstanceData.csv'))
    
    # Generate and save audit report
    generate_audit_report(image_df, breast_df, video_df, video_images_df if reparse_images else None)