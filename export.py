import os, cv2
import pandas as pd
import ast
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))

# Paths
parsed_database = f'{env}/database/'
output_dir = f'{env}/export/'

biopsy_mapping = {
        'Pathology Malignant': 'malignant',
        'Known Biopsy-Proven Malignancy': 'malignant',
        
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

def Crop_Images(df):
    
    image_output = f"{output_dir}/images/"
    mask_folder_output = f"{output_dir}/masks/"
    os.makedirs(image_output, exist_ok=True)
    os.makedirs(mask_folder_output, exist_ok=True)
    
    image_folder_path = f"{env}/database/images/"
    mask_folder_input = f"{env}/database/masks/"
    
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


def Crop_Videos(df):
    
    video_output = f"{output_dir}/videos/"
    os.makedirs(video_output, exist_ok=True)
    
    video_folder_path = f"{env}/database/videos/"

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_video, row, video_folder_path, video_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()


def merge_and_fillna(df, breast_df):
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


def Export_Database(trust_threshold):
    
    print("Exporting Data:")
    
    os.makedirs(output_dir, exist_ok = True)
    
    #Dirs
    image_csv_file = f'{parsed_database}ImageData.csv'
    breast_csv_file = f'{parsed_database}BreastData.csv' 
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    labeled_csv_file = f'{parsed_database}LabeledData.csv'
    video_csv_file =  f'{parsed_database}VideoData.csv'

    # Read the case study data and filter it
    case_study_df = pd.read_csv(case_study_csv_file)
    filtered_case_study_df = case_study_df[case_study_df['trustworthiness'] <= trust_threshold]
    # Reformat biopsy
    filtered_case_study_df['Biopsy'] = filtered_case_study_df['Biopsy'].apply(ast.literal_eval)
    filtered_case_study_df['Biopsy'] = filtered_case_study_df['Biopsy'].apply(transform_biopsy_list)
    
    
    # Read Data Files
    video_df = pd.read_csv(video_csv_file)
    image_df = pd.read_csv(image_csv_file)
    breast_df = pd.read_csv(breast_csv_file)
    try:
        labeled_df = pd.read_csv(labeled_csv_file)
    except FileNotFoundError:
        labeled_df = pd.DataFrame(columns=['Patient_ID'])
    

    # Filter the image data based on the filtered case study data and the 'label' column
    image_df = image_df[(image_df['Patient_ID'].isin(filtered_case_study_df['Patient_ID'])) & (image_df['label'] == True)]
    video_df = video_df[(video_df['Patient_ID'].isin(filtered_case_study_df['Patient_ID'])) & (image_df['label'] == True)]
    
    # Crop the images for the relevant studies
    Crop_Images(image_df)
    Crop_Videos(video_df)
    
    # Filter DFs
    image_columns = ['Patient_ID', 
                          'Accession_Number', 
                          'ImageName',
                          'PhotometricInterpretation',
                          'area',
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
                          'area',
                          'nipple_dist',
                          'orientation',
                          'laterality',
                          'crop_aspect_ratio']
    video_df = video_df[video_columns]
    
    
    # Convert 'Patient_ID' columns to integers
    image_df['Patient_ID'] = image_df['Patient_ID'].astype(int)
    labeled_df['Patient_ID'] = labeled_df['Patient_ID'].astype(int)
    # Set 'Labeled' to True for rows with a 'Patient_ID' in labeled_df
    image_df.loc[image_df['Patient_ID'].isin(labeled_df['Patient_ID']), 'labeled'] = True
    
    # Transfer Biopsy data
    image_df = merge_and_fillna(image_df, breast_df)
    video_df = merge_and_fillna(video_df, breast_df)
    
    
    # Write the filtered dataframes to CSV files in the output directory
    breast_df.to_csv(os.path.join(output_dir, 'BreastData.csv'), index=False)
    filtered_case_study_df.to_csv(os.path.join(output_dir, 'CaseStudyData.csv'), index=False)
    labeled_df.to_csv(os.path.join(output_dir, 'LabeledData.csv'), index=False)
    video_df.to_csv(os.path.join(output_dir, 'VideoData.csv'), index=False)
    image_df.to_csv(os.path.join(output_dir, 'ImageData.csv'), index=False)