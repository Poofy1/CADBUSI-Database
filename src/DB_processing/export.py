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
    
    if 'valid' not in df.columns:
        df['valid'] = None
    
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
    
    df['valid'] = df['Patient_ID'].apply(assign_split)
    
    # Count samples in each split
    train_samples = (df['valid'] == 0).sum()
    val_samples = (df['valid'] == 1).sum()
    test_samples = (df['valid'] == 2).sum()
    
    # Print split statistics
    print(f"Split completed: {train_samples} training, {val_samples} validation, {test_samples} test samples")
    print(f"Patient split: {num_train_patients} training, {num_val_patients} validation, {num_test_patients} test patients")
    
    # Log statistics to audit file
    append_audit("export.train_patients", num_train_patients)
    append_audit("export.val_patients", num_val_patients)
    append_audit("export.test_patients", num_test_patients)
    
    return df
    

def Fix_CM_Data(df):
    df['nipple_dist'] = df['nipple_dist'].str.replace('cm', '').str.replace(' ', '')

    # Handle range values
    df['nipple_dist'] = df['nipple_dist'].apply(lambda x: round(np.mean([int(i) for i in x.split('-')])) if isinstance(x, str) and '-' in x else x)

    # Convert to numeric and handle errors
    df['nipple_dist'] = pd.to_numeric(df['nipple_dist'], errors='coerce')

    df.loc[df['nipple_dist'] > 25, 'nipple_dist'] = np.nan

    # Replace 0 with NaN
    df.loc[df['nipple_dist'] == 0, 'nipple_dist'] = np.nan
    
    return df




def format_data(breast_data, image_data):
    # Join breast_data and image_data on Accession_Number and Breast/laterality
    data = pd.merge(breast_data, image_data, left_on=['Accession_Number', 'Study_Laterality'], 
                    right_on=['Accession_Number', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)

    # Keep only the specified columns
    columns_to_keep = ['Patient_ID', 'Accession_Number', 'Study_Laterality', 'ImageName', 'Has_Malignant', 'Has_Benign', 'valid']
    data = data[columns_to_keep]
    
    
    
    # Group by Accession_Number and Breast, and aggregate
    data = data.groupby(['Accession_Number', 'Study_Laterality']).agg({
        'Patient_ID': 'first',
        'ImageName': lambda x: list(x),
        'Has_Malignant': 'first',
        'Has_Benign': 'first',
        'valid': 'first',
    }).reset_index()
    
    #data.to_csv('D:\DATA\CASBUSI\exports\export_01_30_2024/test.csv', index=False)

    # Remove the Patient_ID column
    data.drop('Patient_ID', axis=1, inplace=True)

    # Rename columns
    data.rename(columns={'ImageName': 'Images', 'valid': 'Valid'}, inplace=True)
    
    # Add a new column 'ID' that counts up from 0
    data['ID'] = range(len(data))

    # Make 'ID' the first column
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

def ExportAuditReport(image_df, breast_df, video_df, video_images_df):
    # -- Basic counts --
    # Number of patients
    unique_patients = breast_df['Patient_ID'].nunique()
    append_audit("export.num_patients", unique_patients)
    
    # -- Year range --
    # Convert DATE column to datetime if not already
    breast_df['DATE'] = pd.to_datetime(breast_df['DATE'], errors='coerce')
    year_min = breast_df['DATE'].dt.year.min()
    year_max = breast_df['DATE'].dt.year.max()
    append_audit("export.year_range_start", int(year_min))
    append_audit("export.year_range_end", int(year_max))
    
    # -- Image statistics per exam --
    # Group by Accession_Number to get image counts per exam
    exam_image_counts = image_df.groupby('Accession_Number').size().reset_index(name='image_count')
    append_audit("export.min_images_per_exam", int(exam_image_counts['image_count'].min()))
    append_audit("export.max_images_per_exam", int(exam_image_counts['image_count'].max()))
    append_audit("export.avg_images_per_exam", float(exam_image_counts['image_count'].mean()))
    
    # -- Video statistics per exam --
    if not video_df.empty:
        exam_video_counts = video_df.groupby('Accession_Number').size().reset_index(name='video_count')
        min_videos = int(exam_video_counts['video_count'].min()) if not exam_video_counts.empty else 0
        max_videos = int(exam_video_counts['video_count'].max()) if not exam_video_counts.empty else 0
        avg_videos = float(exam_video_counts['video_count'].mean()) if not exam_video_counts.empty else 0
        append_audit("export.min_videos_per_exam", min_videos)
        append_audit("export.max_videos_per_exam", max_videos)
        append_audit("export.avg_videos_per_exam", avg_videos)
    else:
        append_audit("export.min_videos_per_exam", 0)
        append_audit("export.max_videos_per_exam", 0)
        append_audit("export.avg_videos_per_exam", 0)
    
    # -- Patient age statistics --
    # Filter out any non-numeric ages
    valid_ages = pd.to_numeric(breast_df['AGE_AT_EVENT'], errors='coerce').dropna()
    append_audit("export.min_patient_age", float(valid_ages.min()))
    append_audit("export.max_patient_age", float(valid_ages.max()))
    append_audit("export.avg_patient_age", float(valid_ages.mean()))

    # -- Image dimensions --
    # Calculate average image dimensions if crop_w and crop_h are available
    append_audit("export.avg_image_width", float(image_df['crop_w'].mean()))
    append_audit("export.avg_image_height", float(image_df['crop_h'].mean()))
    
    # -- Video dimensions and frames --
    append_audit("export.avg_video_width", float(video_df['crop_w'].mean()))
    append_audit("export.avg_video_height", float(video_df['crop_h'].mean()))
    
    # -- Video frames calculation --
    # If video_images_df is provided and not empty
    if video_images_df is not None and not video_images_df.empty:
        # Calculate the frame count for each video
        video_images_df['frame_count'] = video_images_df['images'].apply(lambda x: len(x) if isinstance(x, list) else len(eval(x)))
        append_audit("export.avg_video_frames", float(video_images_df['frame_count'].mean()))
        
        # Also add min and max frame counts
        append_audit("export.min_video_frames", int(video_images_df['frame_count'].min()))
        append_audit("export.max_video_frames", int(video_images_df['frame_count'].max()))

    # -- Laterality counts --
    # Count left and right breasts
    laterality_counts = breast_df['Study_Laterality'].value_counts()
    append_audit("export.num_left_breasts", int(laterality_counts.get('LEFT', 0)))
    append_audit("export.num_right_breasts", int(laterality_counts.get('RIGHT', 0)))
    append_audit("export.num_bilateral_breasts", int(laterality_counts.get('BILATERAL', 0)))
    
    # -- Training counts by laterality and diagnosis --
    # Initialize a dictionary to store all counts
    breast_counts = {
        f"{lat.lower()}_{diag.lower()}": [0, 0, 0]  # [train, val, test]
        for lat in ['RIGHT', 'LEFT']
        for diag in ['MALIGNANT', 'BENIGN']
    }
    
    # Process each split
    for split_num in [0, 1, 2]:  # 0->train, 1->val, 2->test
        # Filter data by split
        split_data = breast_df[breast_df['valid'] == split_num]
        
        # Count for each combination
        for laterality in ['RIGHT', 'LEFT']:
            for diagnosis in ['MALIGNANT', 'BENIGN']:
                # Create diagnosis condition
                diagnosis_condition = split_data['final_interpretation'].isin([diagnosis])
                
                # Count and store in the appropriate array position
                key = f"{laterality.lower()}_{diagnosis.lower()}"
                breast_counts[key][split_num] = len(
                    split_data[(split_data['Study_Laterality'] == laterality) & diagnosis_condition]
                )
    
    # Save all arrays to audit
    for key, counts in breast_counts.items():
        append_audit(f'export.{key}_breasts', counts)
        
    
    # -- Machine model distribution counts by split --
    if 'ManufacturerModelName' in image_df.columns:
        # Merge dataframes to get valid split information with the machine models
        model_df = image_df.merge(breast_df[['Patient_ID', 'Accession_Number', 'valid']], 
                              on=['Patient_ID', 'Accession_Number'], how='left')
        
        # Get the unique machine models
        unique_models = model_df['ManufacturerModelName'].unique().tolist()
        
        # Create three dictionaries for train, val, test splits
        train_models = {}
        val_models = {}
        test_models = {}
        
        # Process each split and model
        for model in unique_models:
            # Create a safe key by replacing any characters that might cause issues
            safe_model = str(model).replace(' ', '_').replace('-', '_').replace('.', '_').replace("'", "")
            
            # Count each model in each split
            train_count = len(model_df[(model_df['valid'] == 0) & (model_df['ManufacturerModelName'] == model)])
            val_count = len(model_df[(model_df['valid'] == 1) & (model_df['ManufacturerModelName'] == model)])
            test_count = len(model_df[(model_df['valid'] == 2) & (model_df['ManufacturerModelName'] == model)])
            
            # Only add to dictionaries if count > 0
            if train_count > 0:
                train_models[safe_model] = train_count
            if val_count > 0:
                val_models[safe_model] = val_count
            if test_count > 0:
                test_models[safe_model] = test_count
        
        # Save the dictionaries to audit
        append_audit("export.train_machine_models", train_models)
        append_audit("export.val_machine_models", val_models)
        append_audit("export.test_machine_models", test_models)
    else:
        # Log that the ManufacturerModelName column wasn't found
        append_audit("export.machine_models", "Column 'ManufacturerModelName' not found in image_df")
    
    # -- Breast density distribution counts by split --
    if 'Density_Desc' in breast_df.columns:
        # Define the density categories and their keywords
        density_categories = {
            'entirely_fatty': ['entirely fatty'],
            'fibroglandular': ['fibroglandular'],
            'heterogeneously': ['heterogeneously'],
            'extremely_dense': ['extremely dense'],
            'unknown': []  # Default category if no match found
        }
        
        # Create three dictionaries for train, val, test splits
        train_densities = {cat: 0 for cat in density_categories.keys()}
        val_densities = {cat: 0 for cat in density_categories.keys()}
        test_densities = {cat: 0 for cat in density_categories.keys()}
        
        # Function to classify a density description
        def classify_density(desc):
            if pd.isna(desc):
                return 'unknown'
                
            desc = str(desc).lower()
            
            for category, keywords in density_categories.items():
                for keyword in keywords:
                    if keyword in desc:
                        return category
                        
            return 'unknown'  # Default if no match found
        
        # Add a new column with the classified density
        breast_df['density_category'] = breast_df['Density_Desc'].apply(classify_density)
        
        # Count densities by split
        for split_num, density_dict in [(0, train_densities), (1, val_densities), (2, test_densities)]:
            # Filter by split
            split_data = breast_df[breast_df['valid'] == split_num]
            
            # Count occurrences of each density category
            density_counts = split_data['density_category'].value_counts().to_dict()
            
            # Update the dictionary with counts
            for category in density_categories.keys():
                density_dict[category] = density_counts.get(category, 0)
        
        # Save the dictionaries to audit
        append_audit("export.train_breast_densities", train_densities)
        append_audit("export.val_breast_densities", val_densities)
        append_audit("export.test_breast_densities", test_densities)
    else:
        # Log that the Density_Desc column wasn't found
        append_audit("export.breast_densities", "Column 'Density_Desc' not found in breast_df")
        
    
    # -- BI-RADS distribution counts by split --
    # Define all possible BI-RADS values to check
    birad_values = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']
    
    # Initialize a dictionary to store counts for each BI-RADS value
    birad_counts = {
        birad: [0, 0, 0]  # [train, val, test]
        for birad in birad_values
    }
    
    # Process each split
    for split_num in [0, 1, 2]:  # 0->train, 1->val, 2->test
        # Filter data by split
        split_data = breast_df[breast_df['valid'] == split_num]
        
        # Count each BI-RADS value
        for birad in birad_values:
            birad_counts[birad][split_num] = len(split_data[split_data['BI-RADS'] == birad])
    
    # Save all arrays to audit
    for birad, counts in birad_counts.items():
        # Use a safe key name by replacing characters that might cause issues
        safe_birad = birad.replace('-', '_').replace('/', '_')
        append_audit(f'export.birad_{safe_birad}', counts)
        
        
    # -- Images per split --
    # First, merge image_df with breast_df to get split information for each image
    merged_df = image_df.merge(breast_df[['Patient_ID', 'Accession_Number', 'Study_Laterality', 'valid']], 
                            on=['Patient_ID', 'Accession_Number'], how='left')

    # Create mapping for numeric codes to split names
    split_mapping = {
        0: 'train',
        1: 'val',
        2: 'test'
    }

    # Count images by valid value (0, 1, 2)
    valid_counts = merged_df.groupby('valid').size()

    # Map the counts to the correct split names and log them
    for valid_code, split_name in split_mapping.items():
        count = int(valid_counts.get(valid_code, 0))
        append_audit(f'export.images_in_{split_name}', count)
        
    # -- Detailed counts per case --
    # Get image counts for every case (just the counts, not the IDs)
    case_image_counts = image_df.groupby('Accession_Number').size().tolist()
    append_audit("export.img_per_case", case_image_counts)
    
    # Get video counts for every case (just the counts, not the IDs)
    if not video_df.empty:
        case_video_counts = video_df.groupby('Accession_Number').size().tolist()
        append_audit("export.vid_per_case", case_video_counts)
    else:
        # If no videos, return empty list
        append_audit("export.vid_per_case", [])
        
        
def Export_Database(CONFIG, reparse_images = True):
    #Debug Tools
    use_reject_system = True # True = removes rejects from trianing
    
    output_dir = CONFIG["EXPORT_DIR"]
    parsed_database = CONFIG["DATABASE_DIR"]
    labelbox_path = CONFIG["LABELBOX_LABELS"]
    
    
    date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_dir = f'{output_dir}/export_{date}/'
    
    print("Exporting Data:")
    make_dirs(output_dir)
    
    # Save the config to the export location
    export_config_path = os.path.join(output_dir, 'export_config.json')
    export_config_path = os.path.normpath(export_config_path)
    # Convert CONFIG to a JSON string
    config_json_str = json.dumps(CONFIG, indent=4)
    save_data(config_json_str, export_config_path)
    
    #Dirs
    image_csv_file = os.path.join(parsed_database, 'ImageData.csv')
    breast_csv_file = os.path.join(parsed_database, 'BreastData.csv') 
    video_csv_file = os.path.join(parsed_database, 'VideoData.csv')
    instance_labels_csv_file = os.path.join(labelbox_path, 'InstanceLabels.csv')
    
    # Read data
    video_df = read_csv(video_csv_file)
    image_df = read_csv(image_csv_file)
    breast_df = read_csv(breast_csv_file)
    if file_exists(instance_labels_csv_file):
        instance_data = read_csv(instance_labels_csv_file)
        
        # Format Instance Data
        file_to_image_name_map = dict(zip(image_df['FileName'], image_df['ImageName']))
        instance_data['ImageName'] = instance_data['FileName'].map(file_to_image_name_map)
        instance_data = instance_data[instance_data['Accession_Number'].isin(image_df['Accession_Number'])]

        if 'Reject Image' in instance_data.columns:
            if use_reject_system:
                # Count before filtering
                before_count = len(image_df)
                
                # Create a new DataFrame with rejected instances
                rejected_images = instance_data[instance_data['Reject Image'] == True][['FileName', 'ImageName']]
                
                # Remove rows where 'Reject Image' is True from instance_data
                instance_data = instance_data[instance_data['Reject Image'] != True]
                
                # Remove rows from image_df based on rejected FileNames
                image_df = image_df[~image_df['FileName'].isin(rejected_images['FileName'])]
                
                # Calculate how many were removed
                removed_count = before_count - len(image_df)
                
                append_audit("export.labeled_reject_removed", removed_count)
            
            # If not using reject system, keep 'Reject Image' as a column
            if not use_reject_system:
                instance_data['Reject Image'] = instance_data['Reject Image'].fillna(False)
            else:
                instance_data.drop(columns=['Reject Image'], inplace=True)
    else:
        instance_data = None
        

    if os.path.exists(labeled_data_dir):
        all_files = glob.glob(f'{labeled_data_dir}/*.csv')
        all_dfs = (read_csv(f) for f in all_files)
        labeled_df = pd.concat(all_dfs, ignore_index=True)
    else:
        labeled_df = pd.DataFrame(columns=['Patient_ID'])
    

    # Filter the image data based on the filtered case study data and the 'label' column
    image_df = image_df[image_df['label'] == True]
    image_df = image_df[(image_df['Patient_ID'].isin(breast_df['Patient_ID']))]
    image_df = image_df.drop(['label', 'area'], axis=1)
    
    video_df = video_df[video_df['laterality'] != 'unknown']
    video_df = video_df[(video_df['Patient_ID'].isin(breast_df['Patient_ID']))]
    
    initial_image_count = len(image_df)
    initial_video_count = len(video_df)
    append_audit("export.init_images", initial_image_count)
    append_audit("export.init_videos", initial_video_count)
    
    #Remove bad aspect ratios
    min_aspect_ratio = 0.5
    max_aspect_ratio = 4.0
    image_df_after_aspect = image_df[(image_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
                        (image_df['crop_aspect_ratio'] <= max_aspect_ratio)]
    video_df_after_aspect = video_df[(video_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
                        (video_df['crop_aspect_ratio'] <= max_aspect_ratio)]
    
    intermediate_image_count = len(image_df_after_aspect)
    intermediate_video_count = len(video_df_after_aspect)
    append_audit("export.bad_aspect_image_removed", initial_image_count - intermediate_image_count)
    append_audit("export.bad_aspect_video_removed", initial_video_count - intermediate_video_count)

    # Remove images with crop width or height less than 200 pixels
    min_dimension = 200
    image_df = image_df_after_aspect[(image_df_after_aspect['crop_w'] >= min_dimension) & 
                    (image_df_after_aspect['crop_h'] >= min_dimension)]
    video_df = video_df_after_aspect[(video_df_after_aspect['crop_w'] >= min_dimension) & 
                        (video_df_after_aspect['crop_h'] >= min_dimension)]
    
    append_audit("export.too_small_image_removed", intermediate_image_count - len(image_df))
    append_audit("export.too_small_video_removed", intermediate_video_count - len(video_df))

    if reparse_images:   
        # Crop the images for the relevant studies
        Crop_Images(image_df, parsed_database, output_dir)
        Crop_Videos(video_df, parsed_database, output_dir)
    
    # Convert 'Patient_ID' columns to integers
    labeled_df['Patient_ID'] = labeled_df['Patient_ID'].astype(str)
    image_df['Accession_Number'] = image_df['Accession_Number'].astype(str)
    image_df['Patient_ID'] = image_df['Patient_ID'].astype(str)
    breast_df['Accession_Number'] = breast_df['Accession_Number'].astype(str)
    breast_df['Patient_ID'] = breast_df['Patient_ID'].astype(str)

    # Set 'Labeled' to True for rows with a 'Patient_ID' in labeled_df
    image_df.loc[image_df['Patient_ID'].isin(labeled_df['Patient_ID']), 'labeled'] = True
    
    #Find Image Counts (Breast Data)
    image_df['laterality'] = image_df['laterality'].str.upper()
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
        
    # Fix cm data
    image_df = Fix_CM_Data(image_df)
    video_df = Fix_CM_Data(video_df)
    

    # Val split for case data
    breast_df = PerformSplit(CONFIG, breast_df)
    
    # Create trainable csv data
    train_data = format_data(breast_df, image_df)
    
    # Create a mapping of (Accession_Number, laterality) to list of ImagesPath
    video_df['laterality'] = video_df['laterality'].str.upper()
    video_paths = video_df.groupby(['Accession_Number', 'laterality'])['ImagesPath'].agg(list).to_dict()
    train_data['VideoPaths'] = train_data.apply(lambda row: video_paths.get((row['Accession_Number'], row['Study_Laterality']), []), axis=1)

    if reparse_images:  
        video_images_df = generate_video_images_csv(video_df, output_dir)
        save_data(video_images_df, os.path.join(output_dir, 'VideoImages.csv'))

    # Write the filtered dataframes to CSV files in the output directory
    save_data(breast_df, os.path.join(output_dir, 'BreastData.csv'))
    save_data(labeled_df, os.path.join(output_dir, 'LabeledData.csv'))
    save_data(video_df, os.path.join(output_dir, 'VideoData.csv'))
    save_data(image_df, os.path.join(output_dir, 'ImageData.csv'))
    save_data(train_data, os.path.join(output_dir, 'TrainData.csv'))
    if instance_data is not None:
        save_data(instance_data, os.path.join(output_dir, 'InstanceData.csv'))
    
    # Generate and save audit report
    ExportAuditReport(image_df, breast_df, video_df, video_images_df if reparse_images else None)