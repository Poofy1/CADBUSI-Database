import os, pydicom, hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import io
from tqdm import tqdm
import warnings, logging, cv2
from storage_adapter import *
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))

 
def has_blue_pixels(image, n=100, min_b=200):
    # Create a mask where blue is dominant
    channel_max = np.argmax(image, axis=-1)
    blue_dominant = (channel_max == 2) & (
        (image[:, :, 2] - image[:, :, 0] >= n) &
        (image[:, :, 2] - image[:, :, 1] >= n)
    )
    
    strong = image[:, :, 2] >= min_b
    return np.any(blue_dominant & strong)

def has_red_pixels(image, n=100, min_r=200):
    # Create a mask where red is dominant
    channel_max = np.argmax(image, axis=-1)
    red_dominant = (channel_max == 0) & (
        (image[:, :, 0] - image[:, :, 2] >= n) &
        (image[:, :, 0] - image[:, :, 1] >= n)
    )
    
    strong = image[:, :, 0] >= min_r
    return np.any(strong & red_dominant)

    
def generate_hash(data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()


def parse_video_data(dcm, current_index, parsed_database):
    data_dict = {}
    dcm_data = read_binary(dcm)
    
    for elem in dataset:
        if elem.VR == "SQ" and elem.value and len(elem.value) > 0:  # if sequence
            for sub_elem in elem.value[0]:  # only take the first item in the sequence
                tag_name = pydicom.datadict.keyword_for_tag(sub_elem.tag)
                if tag_name == "PixelData":
                    continue
                data_dict[tag_name] = str(sub_elem.value)
        else:
            tag_name = pydicom.datadict.keyword_for_tag(elem.tag)
            if tag_name == "PixelData":
                continue
            data_dict[tag_name] = str(elem.value)
    
    #create video folder
    video_path = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}"
    make_dirs(f"{parsed_database}/videos/{video_path}/")
    
    #get image frames
    image_count = 0
    
   # Get total number of frames
    total_frames = getattr(dataset, 'NumberOfFrames', 1)
    
    # Extract frames directly from dataset using pydicom
    for i in range(0, total_frames, 4):  # Process every 4th frame
        # Use pydicom to extract the frame
        if hasattr(dataset, 'pixel_array'):
            if total_frames > 1:
                frame = dataset.pixel_array[i]
            else:
                frame = dataset.pixel_array
                
            # Convert to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:  # if the frame has 3 channels
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            
            image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}_{image_count}.png"
            
            save_data(frame, f"{parsed_database}/videos/{video_path}/{image_name}")
            
            image_count += 1

    # Add custom data
    data_dict['DataType'] = 'video'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImagesPath'] = video_path
    data_dict['SavedFrames'] = image_count
    data_dict['DicomHash'] = generate_hash(dcm_data)
    
    return data_dict



def parse_single_dcm(dcm, current_index, parsed_database):
    
    media_type = os.path.basename(dcm)[:5]
    if media_type != 'image':
        return parse_video_data(dcm, current_index, parsed_database)
    
    data_dict = {}
    region_count = 0
    
    # Read the DICOM file
    dcm_data = read_binary(dcm)
    dataset = pydicom.dcmread(io.BytesIO(dcm_data))

    for elem in dataset:
        # Check if element is a sequence
        if elem.VR == "SQ":
            # Count regions
            for i, sub_elem in enumerate(elem):
                if elem.keyword == 'SequenceOfUltrasoundRegions':
                    region_count += 1
            
            #Get Data
            if elem.value and len(elem.value) > 0:
                for sub_elem in elem.value[0]:  # only take the first item in the sequence
                    tag_name = pydicom.datadict.keyword_for_tag(sub_elem.tag)
                    if tag_name == "PixelData":
                        continue
                    data_dict[tag_name] = str(sub_elem.value)

        else:
            tag_name = pydicom.datadict.keyword_for_tag(elem.tag)
            if tag_name == "PixelData":
                continue
            data_dict[tag_name] = str(elem.value)
    
    # get image data
    im = dataset.pixel_array
    if data_dict.get('PhotometricInterpretation', '') == 'RGB':
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        np_im = np.array(im)

        
        
        # check if there is any blue pixel
        if has_red_pixels(np_im) and has_blue_pixels(np_im):
            im = cv2.cvtColor(np_im, cv2.COLOR_BGR2RGB)
        else:
            # Convert yellow pixels to white and convert to grayscale
            yellow = [255, 255, 0]  # RGB values for yellow
            mask = np.all(np_im == yellow, axis=-1)
            np_im[mask] = [255, 255, 255]
            im = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
            data_dict['PhotometricInterpretation'] = 'MONOCHROME2_OVERRIDE'

    image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}.png"
    save_data(im, f"{parsed_database}/images/{image_name}")

    # Add custom data
    data_dict['DataType'] = 'image'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImageName'] = image_name
    data_dict['DicomHash'] = generate_hash(dcm_data)
    data_dict['RegionCount'] = region_count
    
    return data_dict





def parse_files(dcm_files_list, parsed_database):
    print("Parsing DCM Data")

    # Load the current index from a file
    index_file = os.path.join(parsed_database, "IndexCounter.txt")
    if file_exists(index_file):
        content = read_txt(index_file)
        if content:
            current_index = int(content)
    else:
        current_index = 0

    print(f'New Dicom Files: {len(dcm_files_list)}')

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(parse_single_dcm, dcm, i+current_index, parsed_database): dcm for i, dcm in enumerate(dcm_files_list)}
        data_list = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            try:
                data = future.result()
                data_list.append(data)
            except Exception as exc:
                print(f'An exception occurred: {exc}')

        # Save index
        new_index = str(current_index + len(data_list))
        save_data(new_index, index_file)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df


def parse_anon_file(anon_location, database_path, image_df):
    
    video_df = image_df[image_df['DataType'] == 'video']
    image_df = image_df[image_df['DataType'] == 'image']
    
    # Define common columns
    common_columns = ['Patient_ID', 'Accession_Number', 'RegionSpatialFormat', 'RegionDataType', 
                    'RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 
                    'RegionLocationMaxY1', 'PhotometricInterpretation', 'Rows', 'Columns',
                    'FileName', 'DicomHash']

    # Keep only necessary columns from dataframes
    if not video_df.empty:
        video_df = video_df[common_columns + ['ImagesPath', 'SavedFrames']]

    image_df = image_df[common_columns + ['ImageName', 'RegionCount']]
    
    
    # Find all csv files and combine into df
    breast_csv = read_csv(anon_location)
    breast_csv = breast_csv.sort_values('PATIENT_ID')
    breast_csv = breast_csv.rename(columns={
        'PATIENT_ID': 'Patient_ID',
        'ACCESSION_NUMBER': 'Accession_Number'
    })

    
    # Convert 'Patient_ID' to str in both dataframes before merging
    image_df[['Patient_ID', 'Accession_Number']] = image_df[['Patient_ID', 'Accession_Number']].astype(str)
    breast_csv[['Patient_ID', 'Accession_Number']] = breast_csv[['Patient_ID', 'Accession_Number']].astype(str)
    
    # Remove leading zeros from Patient_ID in both dataframes
    image_df['Patient_ID'] = image_df['Patient_ID'].str.lstrip('0')
    breast_csv['Patient_ID'] = breast_csv['Patient_ID'].str.lstrip('0')
    
    #image_df.to_csv(f"{env}/image_df.csv", index=False) # DEBUG
    #anon_df.to_csv(f"{env}/anon_df.csv", index=False) # DEBUG
    
    # Populate Has_Malignant and Has_Benign based on final_interpretation
    breast_csv['Has_Malignant'] = breast_csv['final_interpretation'] == 'MALIGNANT'
    breast_csv['Has_Benign'] = breast_csv['final_interpretation'] == 'BENIGN'
    
    image_csv_file = f'{database_path}ImageData.csv'
    video_csv_file = f'{database_path}VideoData.csv'
    breast_csv_file = f'{database_path}BreastData.csv'
    
    image_combined_df = image_df
    if os.path.isfile(image_csv_file):
        existing_image_df = pd.read_csv(image_csv_file)
        existing_image_df['Patient_ID'] = existing_image_df['Patient_ID'].astype(str)
        image_df['Patient_ID'] = image_df['Patient_ID'].astype(str)
        
        # keep only old IDs that don't exist in new data
        image_df = image_df[~image_df['Patient_ID'].isin(existing_image_df['Patient_ID'].unique())]
        
        # now concatenate the old data with the new data
        image_combined_df = pd.concat([existing_image_df, image_df], ignore_index=True)

        
    if os.path.isfile(video_csv_file):
        existing_video_df = pd.read_csv(video_csv_file)
        video_df = pd.concat([existing_video_df, video_df], ignore_index=True)


    if os.path.isfile(breast_csv_file):
        existing_breast_df = pd.read_csv(breast_csv_file)
        breast_csv = pd.concat([existing_breast_df, breast_csv], ignore_index=True)
        breast_csv = breast_csv.sort_values(['Patient_ID', 'Accession_Number', 'Breast'])
        breast_csv = breast_csv.drop_duplicates(subset=['Patient_ID', 'Accession_Number', 'Breast'], keep='last')
        breast_csv = breast_csv.reset_index(drop=True)

    # Export the DataFrames to CSV files
    save_data(image_combined_df, image_csv_file)
    save_data(video_df, video_csv_file)
    save_data(breast_csv, breast_csv_file)
    

# Main Method
def Parse_Dicom_Files(database_path, anon_location, raw_storage_database, data_range):
    
    #Create database dir
    make_dirs(database_path)
    make_dirs(f'{database_path}/images/')
    make_dirs(f'{database_path}/videos/')
    
    # Load the list of already parsed files
    parsed_files_list = []
    parsed_files_list_file = f"{database_path}/ParsedFiles.txt"
    if file_exists(parsed_files_list_file):
        content = read_txt(parsed_files_list_file)
        if content:
            parsed_files_list = content.splitlines()

    # Get every Dicom File
    dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
    print(f'Total Dicom Archive: {len(dcm_files_list)}')

    # Apply data range only if it's specified
    if data_range and len(data_range) == 2:
        dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
    parsed_files_set = set(parsed_files_list)
    dcm_files_list = [file for file in dcm_files_list if file not in parsed_files_set]

    
    # Update the list of parsed files and save it
    parsed_files_list.extend(dcm_files_list)
    content = '\n'.join(parsed_files_list)  # Convert list to string with newlines
    save_data(content, parsed_files_list_file)
    
    # Get DCM Data
    image_df = parse_files(dcm_files_list, database_path)
    image_df = image_df.rename(columns={'PatientID': 'Patient_ID'})
    image_df = image_df.rename(columns={'AccessionNumber': 'Accession_Number'})
    
    #Remove missing IDs
    original_row_count = len(image_df)
    # Handle both NaN and empty strings
    image_df['Patient_ID'] = image_df['Patient_ID'].replace('', np.nan)
    image_df['Accession_Number'] = image_df['Accession_Number'].replace('', np.nan)
    image_df = image_df.dropna(subset=['Patient_ID', 'Accession_Number'])
    new_row_count = len(image_df)
    removed_rows = original_row_count - new_row_count
    print(f"Removed {removed_rows} rows with empty values.")
    
    
    parse_anon_file(anon_location, database_path, image_df)
    
    
    

    
    
    
    






