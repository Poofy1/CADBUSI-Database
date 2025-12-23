import os, cv2, ast, datetime, glob
import pandas as pd
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
env = os.path.dirname(os.path.abspath(__file__))
from tools.storage_adapter import *
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager
from src.DB_export.mask_processing import Mask_Lesions
from src.DB_export.audit_report import generate_audit_report
from src.DB_export.instance_data import *
from src.DB_export.filter_images import apply_filters


# Paths
labeled_data_dir = f'{env}/labeled_data_archive/'


def combine_labels(row):
    """Combine all present labels from margin, shape, orientation, echo, posterior, boundary columns into a single string"""
    label_columns = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']
    labels = []
    
    for col in label_columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            # Split by comma in case multiple labels in one column
            values = str(row[col]).split(',')
            for val in values:
                val = val.strip()
                if val:
                    labels.append(val)
    
    return ','.join(labels) if labels else ''

def process_single_image(row, image_folder_path, image_output, mask_folder_input, mask_folder_output):
    try:
        image_name = row['image_name']
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
        return f"Error: Unexpected error processing {row.get('image_name', 'unknown')} - {str(e)}"

def Crop_images(df, input_dir, output_dir):
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
        
        with tqdm(total=len(futures), desc="Cropping US images") as pbar:
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
    folder_name = row['images_path']
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
        with tqdm(total=len(futures), desc="Cropping US videos") as pbar:
            for future in as_completed(futures):
                pbar.update()
                try:
                    result = future.result()
                    processed_videos += result
                except Exception as e:
                    print(f"Error processing video: {e}")
                
    append_audit("export.exported_videos", processed_videos)



def PerformSplit(df):
    val_split = 0.15
    test_split = 0.15
    
    if 'valid' not in df.columns:
        df['valid'] = None
    
    def get_split_from_hash(patient_id, val_split, test_split):
        """Deterministically assign split based on patient_id hash."""
        # Create a stable hash from patient_id
        hash_bytes = hashlib.md5(str(patient_id).encode()).digest()
        # Convert first 8 bytes to a float between 0 and 1
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        hash_float = hash_int / (2**64)
        
        # Assign split based on hash value
        if hash_float < test_split:
            return 2  # Test
        elif hash_float < test_split + val_split:
            return 1  # Validation
        else:
            return 0  # Training
    
    # Apply deterministic split
    df['valid'] = df['patient_id'].apply(
        lambda pid: get_split_from_hash(pid, val_split, test_split)
    )
    
    # Count samples and patients in each split
    train_samples = (df['valid'] == 0).sum()
    val_samples = (df['valid'] == 1).sum()
    test_samples = (df['valid'] == 2).sum()
    
    train_patients = df[df['valid'] == 0]['patient_id'].nunique()
    val_patients = df[df['valid'] == 1]['patient_id'].nunique()
    test_patients = df[df['valid'] == 2]['patient_id'].nunique()
    
    print(f"Split completed: {train_samples} training, {val_samples} validation, {test_samples} test samples")
    print(f"Patient split: {train_patients} training, {val_patients} validation, {test_patients} test patients")
    
    append_audit("export.train_patients", train_patients)
    append_audit("export.val_patients", val_patients)
    append_audit("export.test_patients", test_patients)
    
    return df

def create_train_set(breast_data, image_data, lesion_df=None):
    # Join breast_data and image_data on BOTH accession_number and laterality
    # This ensures split bilateral cases only get their respective images
    # Use LEFT JOIN to keep ALL breast cases, even those without images (e.g., US-only studies)
    data = pd.merge(breast_data, image_data,
                    left_on=['accession_number', 'study_laterality'],
                    right_on=['accession_number', 'laterality'],
                    how='left',
                    suffixes=('', '_image_data'))

    # Remove duplicate columns
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)

    # Drop the redundant laterality column from image_data
    if 'laterality' in data.columns:
        data.drop('laterality', axis=1, inplace=True)

    # Build aggregation dictionary dynamically from breast_data columns only
    # Note: accession_number and study_laterality are groupby keys, so don't include them here
    agg_dict = {}

    # Special handling for image_name - convert to list
    if 'image_name' in data.columns:
        agg_dict['image_name'] = lambda x: list(x)

    # Only include columns that were originally in breast_data (StudyCases table)
    # This excludes image-specific columns like distance, closest_fn, crop_x, etc.
    # Also exclude diagnosis columns that are not needed in final export
    excluded_columns = ['accession_number', 'study_laterality', 'image_name',
                       'has_benign', 'left_diagnosis', 'right_diagnosis']
    breast_columns = set(breast_data.columns)
    for col in data.columns:
        if col in breast_columns and col not in excluded_columns:
            agg_dict[col] = 'first'

    data = data.reset_index(drop=True)
    # Group by BOTH accession_number and study_laterality to keep split bilateral cases separate
    data = data.groupby(['accession_number', 'study_laterality'], as_index=False).agg(agg_dict)
    
    # Combine labels into description column
    data['description'] = data.apply(combine_labels, axis=1)
    
    # Drop individual label columns after combining
    data.drop(['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary'], 
              axis=1, inplace=True)
    
    # Clean original images
    def clean_list(img_list):
        if not isinstance(img_list, list):
            return []
        return [str(img).strip() for img in img_list if pd.notna(img) and str(img).strip()]

    data['images'] = data['image_name'].apply(clean_list)
    data.drop(['image_name'], axis=1, inplace=True)
    
    # Add lesion images if lesion_df is provided
    if lesion_df is not None and not lesion_df.empty and 'lesion_name' in lesion_df.columns:
        # Check if laterality column exists in lesion_df
        if 'laterality' in lesion_df.columns:
            # Handle missing laterality values - drop them as they can't be matched
            lesion_df_with_lat = lesion_df[lesion_df['laterality'].notna()].copy()
            lesion_df_with_lat['laterality'] = lesion_df_with_lat['laterality'].str.upper()

            if not lesion_df_with_lat.empty:
                # Group lesions by BOTH accession_number and laterality for split bilateral cases
                lesion_grouped = lesion_df_with_lat.groupby(['accession_number', 'laterality'])['lesion_name'].apply(list).reset_index()
                lesion_grouped.columns = ['accession_number', 'laterality', 'lesion_images']

                # Merge lesion data with main data on both accession_number and laterality
                data = data.merge(lesion_grouped,
                                left_on=['accession_number', 'study_laterality'],
                                right_on=['accession_number', 'laterality'],
                                how='left')

                # Drop redundant laterality column
                if 'laterality' in data.columns:
                    data.drop('laterality', axis=1, inplace=True)
            else:
                print("Warning: No lesions with valid laterality found")
                data['lesion_images'] = [[] for _ in range(len(data))]
        else:
            # Fallback to old behavior if laterality not present
            print("Warning: laterality column not found in lesion_df, using old grouping")
            lesion_grouped = lesion_df.groupby('accession_number')['lesion_name'].apply(list).reset_index()
            lesion_grouped.columns = ['accession_number', 'lesion_images']
            data = data.merge(lesion_grouped, on='accession_number', how='left')

        # Clean lesion images list - handle NaN values properly
        if 'lesion_images' in data.columns:
            data['lesion_images'] = data['lesion_images'].apply(
                lambda x: clean_list(x) if isinstance(x, list) else []
            )
        else:
            data['lesion_images'] = [[] for _ in range(len(data))]
    else:
        # No lesion data available
        data['lesion_images'] = [[] for _ in range(len(data))]
    
    # Keep ALL rows, including those without images
    data = data.reset_index(drop=True)

    # Report how many rows have no images
    no_images_count = data['images'].apply(lambda x: not isinstance(x, list) or len(x) == 0).sum()
    if no_images_count > 0:
        print(f"Note: {no_images_count} rows have no images (kept in export)")

    data['id'] = range(len(data))
    columns = ['id'] + [col for col in data.columns if col != 'id']
    data = data[columns]

    return data


def generate_video_images_csv(video_df, root_dir):
    """
    Creates a CSV containing all video image paths.
    """
    video_image_data = []
    
    for _, row in tqdm(video_df.iterrows(), total=len(video_df), desc="Processing video folders"):
        video_folder = row['images_path']
        video_dir = os.path.join(root_dir, 'videos', video_folder).replace('\\', '/')
        
        video_files = list_files(video_dir) 
        
        if video_files:
            video_image_data.append({
                'accession_number': row['accession_number'],
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
    merge_columns = ['patient_id', 'accession_number', 'study_laterality'] + available_columns
    
    # Perform the merge
    merged_data = image_df[['image_name', 'patient_id', 'accession_number', 'laterality']].merge(
        breast_df[merge_columns],
        left_on=['patient_id', 'accession_number', 'laterality'],
        right_on=['patient_id', 'accession_number', 'study_laterality'],
        how='left'
    )
    
    # Create mappings and add to instance_data
    for breast_col, instance_col in column_mapping.items():
        if breast_col in available_columns:
            image_to_value_map = dict(zip(merged_data['image_name'], merged_data[breast_col]))
            instance_data[instance_col] = instance_data['image_name'].map(image_to_value_map)
    
    return instance_data
        


def generate_pathology_data(pathology_df):
    """
    Generate pathology data CSV from pathology_df.

    Args:
        pathology_df: DataFrame from Pathology table

    Returns:
        DataFrame with columns: accession_number, patient_id, date, cancer_type
    """
    if pathology_df.empty:
        return pd.DataFrame(columns=['accession_number', 'patient_id', 'date', 'cancer_type'])

    # Select only needed columns
    pathology_data = pathology_df[['accession_number', 'patient_id', 'date', 'cancer_type']].copy()

    # Normalize data types
    pathology_data['patient_id'] = pathology_data['patient_id'].astype(str)
    pathology_data['accession_number'] = pathology_data['accession_number'].astype(str)

    return pathology_data

def generate_caliper_data(image_df):
    """
    Generate caliper data CSV from image_df containing caliper_coordinates column.
    Adjusts coordinates to account for image cropping.

    Args:
        image_df: DataFrame with columns including caliper_coordinates, accession_number, patient_id, image_name, crop_x, crop_y

    Returns:
        DataFrame with columns: id, accession_number, patient_id, image_name, x, y
    """
    caliper_records = []
    caliper_id = 1

    for _, row in image_df.iterrows():
        caliper_coords_str = row.get('caliper_coordinates', '')

        # Skip if no caliper coordinates
        if pd.isna(caliper_coords_str) or caliper_coords_str == '':
            continue

        accession_number = row['accession_number']
        patient_id = row['patient_id']
        image_name = row['image_name']

        # Get crop offsets to adjust coordinates
        crop_x = int(row.get('crop_x', 0))
        crop_y = int(row.get('crop_y', 0))

        # Parse caliper coordinates
        # Format: "999,462;610,309;" - semicolons separate calipers, commas separate x,y
        caliper_pairs = caliper_coords_str.strip().rstrip(';').split(';')

        for pair in caliper_pairs:
            pair = pair.strip()
            if pair:
                try:
                    parts = pair.split(',')
                    if len(parts) == 2:
                        # Original coordinates from DICOM
                        original_x = int(parts[0].strip())
                        original_y = int(parts[1].strip())

                        # Adjust for crop offset
                        x = original_x - crop_x
                        y = original_y - crop_y

                        caliper_records.append({
                            'id': caliper_id,
                            'accession_number': accession_number,
                            'patient_id': patient_id,
                            'image_name': image_name,
                            'x': x,
                            'y': y
                        })
                        caliper_id += 1
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse caliper coordinate '{pair}' for image {image_name}: {e}")

    caliper_df = pd.DataFrame(caliper_records)
    return caliper_df



def normalize_dataframes(image_df, video_df, breast_df):
    """
    Standardize data types and formats across all dataframes.
    
    Returns:
        tuple: (image_df, video_df, breast_df)
    """
    # Normalize laterality to uppercase (do once for each df)
    image_df['laterality'] = image_df['laterality'].str.upper()
    video_df['laterality'] = video_df['laterality'].str.upper()
    
    # Standardize patient_id as string
    image_df['patient_id'] = image_df['patient_id'].astype(int).astype(str)
    breast_df['patient_id'] = breast_df['patient_id'].astype(int).astype(str)
    
    # Standardize accession_number as string
    image_df['accession_number'] = image_df['accession_number'].astype(str)
    breast_df['accession_number'] = breast_df['accession_number'].astype(str)
    
    return image_df, video_df, breast_df

def build_instance_data(image_df, breast_df):
    """
    Build and enrich instance_data with all necessary fields from image_df, breast_df, and labelbox.
    
    Returns:
        tuple: (instance_data, image_df) - image_df may be filtered if reject system is used
    """
    # Create base instance_data
    instance_data = image_df[['dicom_hash', 'image_name']].copy()

    # Add is_lesion column (0 for regular images)
    instance_data['is_lesion'] = 0

    # Map breast data columns
    column_mappings = {
        'age_at_event': 'Age'
    }
    instance_data = map_breast_data_to_instances(instance_data, image_df, breast_df, column_mappings)

    # Add columns from Images table
    image_columns_to_add = ['area', 'orientation', 'photometric_interpretation',
                             'has_calipers', 'has_calipers_prediction', 'inpainted_from',
                             'laterality', 'description', 'crop_aspect_ratio', 'darkness',
                             'software_versions', 'manufacturer_model_name']
    for col in image_columns_to_add:
        if col in image_df.columns:
            image_to_col_map = dict(zip(image_df['image_name'], image_df[col]))
            instance_data[col] = instance_data['image_name'].map(image_to_col_map)

    # Add closest_fn as pair_image
    if 'closest_fn' in image_df.columns:
        image_to_pair_map = dict(zip(image_df['image_name'], image_df['closest_fn']))
        instance_data['pair_image'] = instance_data['image_name'].map(image_to_pair_map)

    # Add physical_delta_x
    if 'physical_delta_x' in image_df.columns:
        image_to_physicaldelta_map = dict(zip(image_df['image_name'], image_df['physical_delta_x']))
        instance_data['physical_delta_x'] = instance_data['image_name'].map(image_to_physicaldelta_map)
    
    # Add image dimensions
    if 'crop_w' in image_df.columns:
        image_to_width_map = dict(zip(image_df['image_name'], image_df['crop_w'].fillna(0).astype(int)))
        instance_data['image_w'] = instance_data['image_name'].map(image_to_width_map)
    
    if 'crop_h' in image_df.columns:
        image_to_height_map = dict(zip(image_df['image_name'], image_df['crop_h'].fillna(0).astype(int)))
        instance_data['image_h'] = instance_data['image_name'].map(image_to_height_map)
    
    return instance_data


def Export_Database(CONFIG, limit = None, reparse_images = True):

    use_reject_system = False # True = removes rejects from training
    database_dir = CONFIG["DATABASE_DIR"]
    instance_labels_csv_file = os.path.join(CONFIG["LABELBOX_LABELS"], 'InstanceLabels.csv')

    date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_dir = os.path.join(database_dir, 'exports', f'export_{date}')
    print(f"Exporting dataset to {output_dir}")
    make_dirs(output_dir)

    # Save the config to the export location
    save_data(json.dumps(CONFIG, indent=4), os.path.normpath(os.path.join(output_dir, 'export_config.json'))) # Convert CONFIG to a JSON string

    # Read data from SQLite database
    with DatabaseManager() as db:
        video_df = db.get_videos_dataframe()
        image_df = db.get_images_dataframe()
        breast_df = db.get_study_cases_dataframe()
        pathology_df = db.get_pathology_dataframe()
        print('Loaded database')

    # Apply test subset early if specified
    if limit:
        # Limit breast_df first
        breast_df = breast_df.head(limit)

        # Filter image_df and video_df to only include patients from the subset breast_df
        subset_patient_ids = breast_df['patient_id'].unique()
        image_df = image_df[image_df['patient_id'].isin(subset_patient_ids)]
        video_df = video_df[video_df['patient_id'].isin(subset_patient_ids)]
        print(f"Subset data sizes - Breast: {len(breast_df)}, Image: {len(image_df)}, Video: {len(video_df)}")
        
    # Track which studies contained axilla images BEFORE filtering removes them
    axilla_accessions = set(image_df[image_df['area'] == 'axilla']['accession_number'].unique())
    breast_df['contained_axilla'] = breast_df['accession_number'].isin(axilla_accessions).astype(int)

    # Normalize image_df BEFORE building instance_data
    image_df, video_df, breast_df = normalize_dataframes(image_df, video_df, breast_df)
    image_df, video_df, breast_df = apply_filters(image_df, video_df, breast_df, CONFIG, output_dir)
    image_df = apply_reject_system(image_df, instance_labels_csv_file, use_reject_system)
    
    # Process lesion masks and get lesion data from database
    lesion_df = Mask_Lesions(database_dir, output_dir, filtered_image_df=image_df) # Set filtered_image_df to None to avoid filtering
    print(f"Processed {len(lesion_df)} lesion images from {lesion_df['image_source'].nunique() if not lesion_df.empty else 0} source images")

    
    instance_data = build_instance_data(image_df, breast_df)
    instance_data = merge_labelbox_labels(instance_data, instance_labels_csv_file)

    if reparse_images:
        # Crop the images for the relevant studies
        Crop_images(image_df, database_dir, output_dir)
        Crop_Videos(video_df, database_dir, output_dir)

    # Val split for case data
    breast_df = PerformSplit(breast_df)
    
    # Create trainable csv data
    train_data = create_train_set(breast_df, image_df, lesion_df)
    
    # Create a mapping of (accession_number, laterality) to list of images_path
    if not video_df.empty and 'images_path' in video_df.columns:
        video_paths = video_df.groupby(['accession_number', 'laterality'])['images_path'].agg(list).to_dict()
        train_data['video_paths'] = train_data.apply(lambda row: video_paths.get((row['accession_number'], row['study_laterality']), []), axis=1)
        
        if reparse_images:  
            video_images_df = generate_video_images_csv(video_df, output_dir)
        else:
            video_images_df = None
    else:
        # No video data available
        train_data['video_paths'] = [[] for _ in range(len(train_data))]  # Empty lists for all rows
        video_images_df = None
        print("No video data found - video_paths set to empty lists")

    # Generate caliper data CSV
    caliper_df = generate_caliper_data(image_df)
    print(f"Generated {len(caliper_df)} caliper coordinate records from {caliper_df['image_name'].nunique() if not caliper_df.empty else 0} images")

    # Generate pathology data CSV
    pathology_data = generate_pathology_data(pathology_df)
    print(f"Generated {len(pathology_data)} pathology records from {pathology_data['accession_number'].nunique() if not pathology_data.empty else 0} accessions")

    # Write the filtered dataframes to CSV files in the output directory
    save_data(video_df, os.path.join(output_dir, 'VideoData.csv'))
    save_data(train_data, os.path.join(output_dir, 'BreastData.csv'))
    save_data(lesion_df, os.path.join(output_dir, 'LesionData.csv'))
    save_data(caliper_df, os.path.join(output_dir, 'CaliperData.csv'))
    save_data(pathology_data, os.path.join(output_dir, 'PathologyData.csv'))
    if instance_data is not None:
        save_data(instance_data, os.path.join(output_dir, 'ImageData.csv'))
    
    # Generate and save audit report
    generate_audit_report(image_df, breast_df, video_df, video_images_df if reparse_images else None)