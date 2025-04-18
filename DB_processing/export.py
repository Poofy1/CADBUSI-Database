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

# Paths
labeled_data_dir = f'{env}/labeled_data_archive/'

biopsy_mapping = {
        'Pathology Malignant': 'malignant',
        'Known Biopsy-Proven Malignancy': 'malignant',
        'Malignant': 'malignant',
        
        'Pathology Benign': 'benign',
        'Probably Benign': 'benign',
        'Pathology Elevated Risk': 'benign',
        'Benign': 'benign',
        
        'Waiting for Pathology': 'unknown',
        'Low Suspicion for Malignancy': 'unknown',
        'Suspicious': 'unknown',
        'Need Additional Imaging Evaluation': 'unknown',
        'Post Procedure Mammogram for Marker Placement': 'unknown',
        'High Suspicion for Malignancy': 'unknown',
        'Highly Suggestive of Malignancy': 'unknown',
        'Moderate Suspicion for Malignancy': 'unknown',
        'Negative': 'unknown',
    }

def transform_biopsy_list(biopsy_list):
    return [biopsy_mapping.get(biopsy, 'unknown') for biopsy in biopsy_list]

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
                
                
                
                
                
def process_single_video(row, video_folder_path, output_dir):
    storage = StorageClient.get_instance()
    
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
        return
        
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


def Crop_Videos(df, input_dir, output_dir):
    
    video_output = f"{output_dir}/videos/"
    make_dirs(video_output)
    
    video_folder_path = f"{input_dir}/videos/"

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_video, row, video_folder_path, video_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()


def merge_and_fillna(df, breast_df):
    
    
    df['laterality'] = df['laterality'].str.upper()
    # Merge df with breast_df on 'Patient_ID' and 'laterality'/'Breast'
    # Before merging, convert Patient_ID to the same type in both dataframes
    breast_df['Patient_ID'] = breast_df['Patient_ID'].astype(str)
    df['Patient_ID'] = df['Patient_ID'].astype(str)

    # Now perform the merge
    df = pd.merge(df, 
                breast_df[['Patient_ID', 'Study_Laterality', 'Has_Malignant', 'Has_Benign', 'Has_Unknown']], 
                left_on=['Patient_ID', 'laterality'], 
                right_on=['Patient_ID', 'Study_Laterality'], 
                how='left')
    # Drop 'Breast' column as it's no longer needed
    df.drop('Study_Laterality', axis=1, inplace=True)
    # Replace NaN values in new columns with appropriate values
    df[['Has_Malignant', 'Has_Benign', 'Has_Unknown']].fillna(0, inplace=True)
    return df


def safe_literal_eval(val, idx):
    val = val.replace("nan,", "'unknown',")
    val = val.replace("nan]", "'unknown']")

    try:
        return ast.literal_eval(val)
    except ValueError:
        print(f"Error parsing value at index {idx}: {val}")
        return val  # or some other default value



def PerformVal(val_split, df):
    if 'valid' not in df.columns:
        df['valid'] = None
    
    # Get unique patient IDs
    unique_patients = df['Patient_ID'].unique()
    
    # Calculate how many patients should be in validation set
    num_val_patients = int(len(unique_patients) * val_split)
    
    # Randomly select patients for validation
    val_patients = np.random.choice(unique_patients, size=num_val_patients, replace=False)
    
    # Assign validation status based on patient ID
    df['valid'] = df['Patient_ID'].apply(lambda x: 1 if x in val_patients else 0)
    
    # Print split statistics
    train_count = (df['valid'] == 0).sum()
    val_count = (df['valid'] == 1).sum()
    print(f"Split completed: {train_count} training samples, {val_count} validation samples")
    print(f"Patient split: {len(unique_patients) - num_val_patients} training patients, {num_val_patients} validation patients")
    
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

    # Convert NaN values to -1 and convert to int
    df['nipple_dist'].fillna(-1, inplace=True)

    # Convert -1 back to NaN
    df.loc[df['nipple_dist'] == -1, 'nipple_dist'] = np.nan
    
    return df




def format_data(breast_data, image_data, num_of_tests):
    # Join breast_data and image_data on Accession_Number and Breast/laterality
    data = pd.merge(breast_data, image_data, left_on=['Accession_Number', 'Study_Laterality'], 
                    right_on=['Accession_Number', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)

    # Filter out rows where Has_Unknown is False
    data = data[data['Has_Unknown'] == False]

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

    # Randomly select a specified number of rows and change their 'Valid' status to '2'
    valid_indices = data.index[data['Valid'].isin([0, 1])].tolist()
    if num_of_tests > 0:
        selected_indices = np.random.choice(valid_indices, num_of_tests, replace=False)
        data.loc[selected_indices, 'Valid'] = 2

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
    
def Export_Database(CONFIG, reparse_images = True, trust_max = 2, num_of_tests = 10):
    #Debug Tools
    KnownInstancesOnly = False # When true it only exports images that have a instance label
    use_reject_system = True # True = removes rejects from trianing
    
    output_dir = CONFIG["EXPORT_DIR"]
    val_split = CONFIG["VAL_SPLIT"]
    parsed_database = CONFIG["DATABASE_DIR"]
    labelbox_path = CONFIG["LABELBOX_LABELS"]
    
    
    date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_dir = f'{output_dir}/export_{date}/'
    
    print("Exporting Data:")
    
    make_dirs(output_dir)
    
    # Save the config to the export location
    export_config_path = os.path.join(output_dir, 'export_config.json')
    with open(export_config_path, 'w') as export_config_file:
        json.dump(CONFIG, export_config_file, indent=4)
    
    #Dirs
    image_csv_file = f'{parsed_database}ImageData.csv'
    breast_csv_file = f'{parsed_database}BreastData.csv' 
    video_csv_file =  f'{parsed_database}VideoData.csv'
    instance_labels_csv_file = f'{labelbox_path}InstanceLabels.csv'

    # Read data
    video_df = read_csv(video_csv_file)
    image_df = read_csv(image_csv_file)
    breast_df = read_csv(breast_csv_file)
    instance_data = read_csv(instance_labels_csv_file)
    
    
    ##Format Instance Data
    file_to_image_name_map = dict(zip(image_df['FileName'], image_df['ImageName']))
    instance_data['ImageName'] = instance_data['FileName'].map(file_to_image_name_map)
    instance_data.drop(columns=['FileName'], inplace=True)

    if 'Reject Image' in instance_data.columns:
        if use_reject_system:
            # Create a new DataFrame with rejected instances
            rejected_images = instance_data[instance_data['Reject Image'] == True][['ImageName']]
            rejected_images['FileName'] = rejected_images['ImageName'].map({v: k for k, v in file_to_image_name_map.items()})
            
            # Remove rows where 'Reject Image' is True from instance_data
            instance_data = instance_data[instance_data['Reject Image'] != True]
            
            # Remove rows from image_df based on rejected_images['FileName']
            image_df = image_df[~image_df['FileName'].isin(rejected_images['FileName'])]
        
        # If not using reject system, keep 'Reject Image' as a column
        if not use_reject_system:
            instance_data['Reject Image'] = instance_data['Reject Image'].fillna(False)
        else:
            instance_data.drop(columns=['Reject Image'], inplace=True)

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
    image_df = image_df[image_df['laterality'].notna()]
    video_df = video_df[(video_df['Patient_ID'].isin(breast_df['Patient_ID']))]
    video_df = video_df[video_df['laterality'] != 'unknown']
    video_df = video_df[video_df['laterality'].notna()]
    
    if KnownInstancesOnly:
        # Filter image_df to only include instances present in instance_data
        image_df = image_df[image_df['ImageName'].isin(instance_data['ImageName'])]
        video_df = video_df[video_df['ImageName'].isin(instance_data['ImageName'])]
    
    #Remove bad aspect ratios
    min_aspect_ratio = 0.5
    max_aspect_ratio = 4.0
    image_df = image_df[(image_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
                        (image_df['crop_aspect_ratio'] <= max_aspect_ratio)]
    video_df = video_df[(video_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
                        (video_df['crop_aspect_ratio'] <= max_aspect_ratio)]
    
    # Remove images with crop width or height less than 200 pixels
    min_dimension = 200
    image_df = image_df[(image_df['crop_w'] >= min_dimension) & 
                        (image_df['crop_h'] >= min_dimension)]
    video_df = video_df[(video_df['crop_w'] >= min_dimension) & 
                        (video_df['crop_h'] >= min_dimension)]
    

    if reparse_images:   
        # Crop the images for the relevant studies
        Crop_Images(image_df, parsed_database, output_dir)
        Crop_Videos(video_df, parsed_database, output_dir)
    
    # Filter DFs
    image_columns = ['Patient_ID', 
                          'Accession_Number', 
                          'ImageName',
                          'FileName',
                          'PhotometricInterpretation',
                          'labeled',
                          'nipple_dist',
                          'orientation',
                          'laterality',
                          'reparsed_orientation',
                          'label_cat',
                          'Inpainted',
                          'crop_aspect_ratio']
    image_df = image_df[image_columns]
    video_columns = ['Patient_ID', 
                          'Accession_Number', 
                          'ImagesPath',
                          'FileName',
                          'area',
                          'nipple_dist',
                          'orientation',
                          'laterality',
                          'crop_aspect_ratio']
    video_df = video_df[video_columns]
    
    
    # Round 'crop_aspect_ratio' to 2 decimal places
    image_df['crop_aspect_ratio'] = image_df['crop_aspect_ratio'].round(2)
    video_df['crop_aspect_ratio'] = video_df['crop_aspect_ratio'].round(2)
    
    # Convert 'Patient_ID' columns to integers
    labeled_df['Patient_ID'] = labeled_df['Patient_ID'].astype(str)
    image_df['Accession_Number'] = image_df['Accession_Number'].astype(str)
    image_df['Patient_ID'] = image_df['Patient_ID'].astype(str)
    breast_df = breast_df.fillna(0).astype({'Accession_Number': 'str'})
    
    # Set 'Labeled' to True for rows with a 'Patient_ID' in labeled_df
    image_df.loc[image_df['Patient_ID'].isin(labeled_df['Patient_ID']), 'labeled'] = True
    
    # Transfer Biopsy data
    image_df = merge_and_fillna(image_df, breast_df)
    video_df = merge_and_fillna(video_df, breast_df)
    
    
    #Find Image Counts (Breast Data)
    image_df['laterality'] = image_df['laterality'].str.upper()
    image_counts = image_df.groupby(['Patient_ID', 'laterality']).size().reset_index(name='Image_Count')
    breast_df = pd.merge(breast_df, image_counts, how='left', left_on=['Patient_ID', 'Study_Laterality'], right_on=['Patient_ID', 'laterality'])
    breast_df = breast_df.drop(['laterality'], axis=1)
    breast_df['Image_Count'] = breast_df['Image_Count'].fillna(0).astype(int)
    
    # Filter out case and breast data that isnt relavent 
    image_patient_ids = image_df['Patient_ID'].unique()
    breast_df = breast_df[breast_df['Patient_ID'].isin(image_patient_ids)]
        
    # Fix cm data
    image_df = Fix_CM_Data(image_df)
    video_df = Fix_CM_Data(video_df)
    

    # Val split for case data
    breast_df = PerformVal(val_split, breast_df)
    
    
    # Create trainable csv data
    train_data = format_data(breast_df, image_df, num_of_tests)
    
    # Create a mapping of (Accession_Number, laterality) to list of ImagesPath
    video_paths = video_df.groupby(['Accession_Number', 'laterality'])['ImagesPath'].agg(list).to_dict()
    train_data['VideoPaths'] = train_data.apply(lambda row: video_paths.get((row['Accession_Number'], row['Study_Laterality']), []), axis=1)

    video_images_df = generate_video_images_csv(video_df, output_dir)

    # Write the filtered dataframes to CSV files in the output directory
    #save_data(breast_df, os.path.join(output_dir, 'BreastData.csv'))
    #save_data(labeled_df, os.path.join(output_dir, 'LabeledData.csv'))
    #save_data(video_df, os.path.join(output_dir, 'VideoData.csv'))
    #save_data(image_df, os.path.join(output_dir, 'ImageData.csv'))
    save_data(train_data, os.path.join(output_dir, 'TrainData.csv'))
    save_data(instance_data, os.path.join(output_dir, 'InstanceData.csv'))
    
    save_data(video_images_df, os.path.join(output_dir, 'VideoImages.csv'))