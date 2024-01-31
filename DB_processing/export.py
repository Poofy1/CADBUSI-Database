import os, cv2, ast, datetime, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))

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
    image_path = os.path.join(image_folder_path, row['ImageName'])
    mask_path = os.path.join(mask_folder_input, 'mask_' + row['ImageName'])
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    
    # Check if the image and mask were loaded properly
    if image is None:
        return
    
    # Get box coordinates
    x = int(row['crop_x'])
    y = int(row['crop_y'])
    w = int(row['crop_w'])
    h = int(row['crop_h'])
    
    # Crop Image
    cropped_image = image[y:y+h, x:x+w]
    image_output_path = os.path.join(image_output, row['ImageName'])
    cv2.imwrite(image_output_path, cropped_image)
    
    
    # Crop Mask
    if mask is None:
        return
    
    cropped_mask = mask[y:y+h, x:x+w]
    mask_output_path = os.path.join(mask_folder_output, 'mask_' + row['ImageName'])
    cv2.imwrite(mask_output_path, cropped_mask)

def Crop_Images(df, input_dir, output_dir):
    
    image_output = f"{output_dir}/images/"
    mask_folder_output = f"{output_dir}/masks/"
    os.makedirs(image_output, exist_ok=True)
    os.makedirs(mask_folder_output, exist_ok=True)
    
    image_folder_path = f"{input_dir}/images/"
    mask_folder_input = f"{labeled_data_dir}/masks/"
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, row, image_folder_path, image_output, mask_folder_input, mask_folder_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()
                
                
                
                
                
def process_single_video(row, video_folder_path, output_dir):
    # Get the folder name and crop data
    folder_name = row['ImagesPath']
    crop_y = row['crop_y']
    crop_w = row['crop_w']
    crop_x = row['crop_x']
    crop_h = row['crop_h']

    # Get the path to the folder
    folder_path = os.path.join(video_folder_path, folder_name)

    # Check if the folder exists
    if os.path.isdir(folder_path):
        # Get a list of all the images in the folder
        all_images = [file for file in os.listdir(folder_path) if file.endswith('.png')]

        # Create a new directory for the video in the output directory
        video_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Iterate over all the images and crop them
        for image_name in all_images:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            # Check if the image was loaded properly
            if image is not None:
                # Crop the image
                cropped_image = image[int(crop_y):int(crop_y)+int(crop_h), int(crop_x):int(crop_x)+int(crop_w)]

                # Save the cropped image
                output_path = os.path.join(video_output_dir, image_name)
                cv2.imwrite(output_path, cropped_image)


def Crop_Videos(df, input_dir, output_dir):
    
    video_output = f"{output_dir}/videos/"
    os.makedirs(video_output, exist_ok=True)
    
    video_folder_path = f"{input_dir}/videos/"

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_video, row, video_folder_path, video_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()


def merge_and_fillna(df, breast_df):
    
    breast_df = breast_df.drop_duplicates(subset=['Patient_ID', 'Breast'])
    
    df['laterality'] = df['laterality'].str.upper()
    # Merge df with breast_df on 'Patient_ID' and 'laterality'/'Breast'
    df = pd.merge(df, 
                  breast_df[['Patient_ID', 'Breast', 'Has_Malignant', 'Has_Benign', 'Has_Unknown']], 
                  left_on=['Patient_ID', 'laterality'], 
                  right_on=['Patient_ID', 'Breast'], 
                  how='left')
    # Drop 'Breast' column as it's no longer needed
    df.drop('Breast', axis=1, inplace=True)
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
    # Check if the 'valid' column exists, if not, create it
    if 'valid' not in df.columns:
        df['valid'] = None  # You can replace None with any default value

    # Get unique anonymized_accession_num values
    unique_accession_nums = df['Accession_Number'].unique()

    # Shuffle the unique accession numbers
    np.random.shuffle(unique_accession_nums)

    # Calculate the split index
    split_index = int(len(unique_accession_nums) * val_split)

    # Assign '1' to a portion of the unique accession numbers and '0' to the remaining
    valid_accession_nums = set(unique_accession_nums[:split_index])
    df['valid'] = df['Accession_Number'].apply(lambda x: 1 if x in valid_accession_nums else 0)

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




def format_data(breast_data, image_data, case_data, num_of_tests):
    # Join breast_data and image_data on Accession_Number and Breast/laterality
    data = pd.merge(breast_data, image_data, left_on=['Accession_Number', 'Breast'], 
                    right_on=['Accession_Number', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)

    # Filter out rows where Has_Unknown is False
    data = data[data['Has_Unknown'] == False]

    # Keep only the specified columns
    columns_to_keep = ['Patient_ID', 'Accession_Number', 'Breast', 'ImageName', 'Has_Malignant', 'Has_Benign']
    data = data[columns_to_keep]
    
    #data.to_csv('D:\DATA\CASBUSI\exports\export_01_30_2024/test.csv', index=False)
    
    # Group by Accession_Number and Breast, and aggregate
    data = data.groupby(['Accession_Number', 'Breast']).agg({
        'Patient_ID': 'first',
        'ImageName': lambda x: list(x),
        'Has_Malignant': 'first',
        'Has_Benign': 'first',
    }).reset_index()
    
    
    
    # Drop duplicates in case_data based on Patient_ID and keep the first occurrence
    unique_case_data = case_data.drop_duplicates(subset='Patient_ID')

    # Merge with the 'valid' column from unique_case_data on Patient_ID
    data = pd.merge(data, unique_case_data[['Patient_ID', 'valid']], on='Patient_ID', how='left')


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



    
def Export_Database(output_dir, val_split, parsed_database, labelbox_path, reparse_images = True, trust_max = 2, num_of_tests = 10):
    
    date = datetime.datetime.now().strftime("%m_%d_%Y")
    output_dir = f'{output_dir}/export_{date}/'
    
    print("Exporting Data:")
    
    os.makedirs(output_dir, exist_ok = True)
    
    #Dirs
    image_csv_file = f'{parsed_database}ImageData.csv'
    breast_csv_file = f'{parsed_database}BreastData.csv' 
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    video_csv_file =  f'{parsed_database}VideoData.csv'
    instance_labels_csv_file = f'{labelbox_path}InstanceLabels.csv'

    # Read data
    case_study_df = pd.read_csv(case_study_csv_file)
    video_df = pd.read_csv(video_csv_file)
    image_df = pd.read_csv(image_csv_file)
    breast_df = pd.read_csv(breast_csv_file)
    instance_data = pd.read_csv(instance_labels_csv_file)
    
    
    #Format Instance Data
    file_to_image_name_map = dict(zip(image_df['FileName'], image_df['ImageName']))
    instance_data['ImageName'] = instance_data['FileName'].map(file_to_image_name_map)
    instance_data.drop(columns=['FileName'], inplace=True)

    # Reformat biopsy
    case_study_df['Biopsy'] = case_study_df.apply(lambda row: safe_literal_eval(row['Biopsy'], row.name), axis=1)
    case_study_df['Biopsy'] = case_study_df['Biopsy'].apply(transform_biopsy_list)
    
    # Trustworthiness
    case_study_df = case_study_df[case_study_df['trustworthiness'] <= trust_max]
    #case_study_df.drop(['trustworthiness'], axis=1, inplace=True)
    

    if os.path.exists(labeled_data_dir):
        all_files = glob.glob(f'{labeled_data_dir}/*.csv')
        all_dfs = (pd.read_csv(f) for f in all_files)
        labeled_df = pd.concat(all_dfs, ignore_index=True)
    else:
        labeled_df = pd.DataFrame(columns=['Patient_ID'])
    

    # Filter the image data based on the filtered case study data and the 'label' column
    image_df = image_df[image_df['label'] == True]
    image_df = image_df[(image_df['Patient_ID'].isin(case_study_df['Patient_ID']))]
    image_df = image_df.drop(['label', 'area'], axis=1)
    image_df = image_df[image_df['laterality'].notna()]
    breast_df = breast_df[(breast_df['Patient_ID'].isin(case_study_df['Patient_ID']))]
    video_df = video_df[(video_df['Patient_ID'].isin(case_study_df['Patient_ID']))]
    video_df = video_df[video_df['laterality'] != 'unknown']
    video_df = video_df[video_df['laterality'].notna()]
    
    #Remove bad aspect ratios
    min_aspect_ratio = 0.6
    max_aspect_ratio = 3.0
    image_df = image_df[(image_df['crop_aspect_ratio'] >= min_aspect_ratio) & 
                        (image_df['crop_aspect_ratio'] <= max_aspect_ratio)]
    
    
    # Remove images with crop width or height less than 200 pixels
    min_dimension = 225
    image_df = image_df[(image_df['crop_w'] >= min_dimension) & 
                        (image_df['crop_h'] >= min_dimension)]
    

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
    labeled_df['Patient_ID'] = labeled_df['Patient_ID'].astype(int)
    image_df = image_df.astype({'Accession_Number': 'int', 'Patient_ID': 'int'})
    breast_df = breast_df.fillna(0).astype({'Accession_Number': 'int'})
    
    # Set 'Labeled' to True for rows with a 'Patient_ID' in labeled_df
    image_df.loc[image_df['Patient_ID'].isin(labeled_df['Patient_ID']), 'labeled'] = True
    
    # Transfer Biopsy data
    image_df = merge_and_fillna(image_df, breast_df)
    video_df = merge_and_fillna(video_df, breast_df)
    
    
    #Find Image Counts (Breast Data)
    image_df['laterality'] = image_df['laterality'].str.upper()
    image_counts = image_df.groupby(['Patient_ID', 'laterality']).size().reset_index(name='Image_Count')
    breast_df = pd.merge(breast_df, image_counts, how='left', left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'])
    breast_df = breast_df.drop(['laterality'], axis=1)
    breast_df['Image_Count'] = breast_df['Image_Count'].fillna(0).astype(int)
    
    # Filter out case and breast data that isnt relavent 
    image_patient_ids = image_df['Patient_ID'].unique()
    case_study_df = case_study_df[case_study_df['Patient_ID'].isin(image_patient_ids)]
    breast_df = breast_df[breast_df['Patient_ID'].isin(image_patient_ids)]
    
    
    # Fix cm data
    image_df = Fix_CM_Data(image_df)
    video_df = Fix_CM_Data(video_df)
    

    # Val split for case data
    case_study_df = PerformVal(val_split, case_study_df)
    
    
    # Create trainable csv data
    train_data = format_data(breast_df, image_df, case_study_df, num_of_tests)
    
    
    # Write the filtered dataframes to CSV files in the output directory
    breast_df.to_csv(os.path.join(output_dir, 'BreastData.csv'), index=False)
    case_study_df.to_csv(os.path.join(output_dir, 'CaseStudyData.csv'), index=False)
    labeled_df.to_csv(os.path.join(output_dir, 'LabeledData.csv'), index=False)
    video_df.to_csv(os.path.join(output_dir, 'VideoData.csv'), index=False)
    image_df.to_csv(os.path.join(output_dir, 'ImageData.csv'), index=False)
    train_data.to_csv(os.path.join(output_dir, 'TrainData.csv'), index=False)
    instance_data.to_csv(os.path.join(output_dir, 'InstanceData.csv'), index=False)