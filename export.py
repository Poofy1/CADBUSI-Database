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

def process_single_image(row, image_folder_path, image_output):
    image_path = os.path.join(image_folder_path, row['ImageName'])
    
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return
    
    # Get box coordinates
    x = int(row['crop_x'])
    y = int(row['crop_y'])
    w = int(row['crop_w'])
    h = int(row['crop_h'])
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    output_path = os.path.join(image_output, row['ImageName'])
    cv2.imwrite(output_path, cropped_image)

def Crop_Images(df):
    
    image_output = f"{output_dir}/images/"
    os.makedirs(image_output, exist_ok=True)
    
    image_folder_path = f"{env}/database/images/"
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, row, image_folder_path, image_output): index for index, row in df.iterrows()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update()



def Export_Database(trust_threshold):
    
    print("Exporting Data:")
    
    os.makedirs(output_dir, exist_ok = True)
    
    #Dirs
    image_csv_file = f'{parsed_database}ImageData.csv'
    breast_csv_file = f'{parsed_database}BreastData.csv' 
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    mask_folder_input = f'{parsed_database}masks/'
    mask_folder_output = f'{output_dir}masks/'
    labeled_csv_file = f'{parsed_database}LabeledData.csv'

    # Read the case study data and filter it
    case_study_df = pd.read_csv(case_study_csv_file)
    filtered_case_study_df = case_study_df[case_study_df['trustworthiness'] <= trust_threshold]
    
    # Reformat biopsy
    filtered_case_study_df['Biopsy'] = filtered_case_study_df['Biopsy'].apply(ast.literal_eval)
    filtered_case_study_df['Biopsy'] = filtered_case_study_df['Biopsy'].apply(transform_biopsy_list)
    
    
    
    # Read the image data
    image_df = pd.read_csv(image_csv_file)
    
    # Filter the image data based on the filtered case study data and the 'label' column
    filtered_image_df = image_df[(image_df['Patient_ID'].isin(filtered_case_study_df['Patient_ID'])) & (image_df['label'] == True)]
    
    # Crop the images for the relevant studies
    Crop_Images(filtered_image_df)
    
    
    
    # Read the breast data
    breast_df = pd.read_csv(breast_csv_file)
    
    # Reformat biopsy
    breast_df['Biopsy'] = breast_df['Biopsy'].map(biopsy_mapping).fillna('unknown')
    


    # Create the destination directory if it doesn't exist
    os.makedirs(mask_folder_output, exist_ok=True)

    for filename in os.listdir(mask_folder_input):
        source_file_path = os.path.join(mask_folder_input, filename)
        destination_file_path = os.path.join(mask_folder_output, filename)
        
        if os.path.isfile(source_file_path):
            shutil.copy2(source_file_path, destination_file_path)
            
    labeled_df = pd.read_csv(labeled_csv_file)
    
    # Write the filtered dataframes to CSV files in the output directory
    filtered_image_df.to_csv(os.path.join(output_dir, 'ImageData.csv'), index=False)
    breast_df.to_csv(os.path.join(output_dir, 'BreastData.csv'), index=False)
    filtered_case_study_df.to_csv(os.path.join(output_dir, 'CaseStudyData.csv'), index=False)
    labeled_df.to_csv(os.path.join(output_dir, 'LabeledData.csv'), index=False)