import os, cv2
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))

# Paths
parsed_database = f'{env}/database/'
output_dir = f'{env}/train/'


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



def Export_Database():
    
    trust_threshold = 2
    
    os.makedirs(output_dir, exist_ok = True)
    
    image_csv_file = f'{parsed_database}ImageData.csv'
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    
    # Read the case study data and filter it
    case_study_df = pd.read_csv(case_study_csv_file)
    filtered_case_study_df = case_study_df[case_study_df['trustworthiness'] <= trust_threshold]
    
    # Read the image data
    image_df = pd.read_csv(image_csv_file)
    
    # Filter the image data based on the filtered case study data and the 'label' column
    filtered_image_df = image_df[(image_df['Patient_ID'].isin(filtered_case_study_df['Patient_ID'])) & (image_df['label'] == True)]
    
    # Crop the images for the relevant studies
    Crop_Images(filtered_image_df)
    
    # Copy the CSV files to the output directory
    shutil.copy(image_csv_file, output_dir)
    shutil.copy(case_study_csv_file, output_dir)