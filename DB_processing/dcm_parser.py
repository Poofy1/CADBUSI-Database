import os, pydicom, zipfile, hashlib, ast
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import io
from tqdm import tqdm
from highdicom.io import ImageFileReader
import warnings, logging, cv2
from storage_adapter import *
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))


biopsy_mapping = {
    'Pathology Malignant': 'malignant',
    'Known Biopsy-Proven Malignancy': 'malignant',
    'Malignant': 'malignant',
    'Pathology Benign': 'benign',
    'Probably Benign': 'benign',
    'Pathology Elevated Risk': 'benign',
    'Benign': 'benign',
    'Malignant': 'malignant',
    'Waiting for Pathology': 'unknown',
    'Low Suspicion for Malignancy': 'unknown',
    'Suspicious': 'unknown',
    'Need Additional Imaging Evaluation': 'unknown',
    'Post Procedure Mammogram for Marker Placement': 'unknown',
    'High Suspicion for Malignancy': 'unknown',
    'Highly Suggestive of Malignancy': 'unknown',
    'Moderate Suspicion for Malignancy': 'unknown',
    'Negative': 'unknown', 
    'Elevated Risk': 'unknown',
}




 
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
    dataset = pydicom.dcmread(dcm, stop_before_pixels=True)
    
    # Convert dataset to binary
    memory_file = io.BytesIO()
    pydicom.filewriter.write_file(memory_file, dataset)
    binary_data = memory_file.getvalue()
    
    for elem in dataset:
        if elem.VR == "SQ":  # if sequence
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
    
    with ImageFileReader(dcm) as image:
        total_frames = image.number_of_frames
        
        # Save every 4th frame
        for i in range(0, total_frames, 4):
            frame = image.read_frame(i, correct_color=False)
            
            # Convert to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:  # if the frame has 3 channels
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
            image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}_{image_count}.png"
            
            save_data(frame, f"{parsed_database}/videos/{video_path}/{image_name}")
            
            image_count += 1

    # Add custom data
    data_dict['DataType'] = 'video'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImagesPath'] = video_path
    data_dict['SavedFrames'] = image_count
    data_dict['DicomHash'] = generate_hash(binary_data)
    
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




# Main Method
def Parse_Dicom_Files(database_path, anon_location, raw_storage_database, data_range):
    image_csv_file = f'{database_path}ImageData.csv'
    video_csv_file = f'{database_path}VideoData.csv'
    case_study_csv_file = f'{database_path}CaseStudyData.csv' 
    breast_csv_file = f'{database_path}BreastData.csv'

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
    
    

    if len(dcm_files_list) <= 0:
        UpdateAnonFile(anon_location) # Not Tested
        return
    
    # Get DCM Data
    image_df = parse_files(dcm_files_list, database_path)
    save_data(image_df.to_csv(index=False), f'{database_path}/Intermediate_Dicom.csv')
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
    
    video_df = image_df[image_df['DataType'] == 'video']
    image_df = image_df[image_df['DataType'] == 'image']
    
    if not video_df.empty:
        video_df = video_df[['Patient_ID', 
                'Accession_Number', 
                'ImagesPath',
                'SavedFrames',
                'RegionSpatialFormat', 
                'RegionDataType', 
                'RegionLocationMinX0', 
                'RegionLocationMinY0', 
                'RegionLocationMaxX1', 
                'RegionLocationMaxY1',
                'PhotometricInterpretation',
                'Rows',
                'Columns',
                'FileName',
                'DicomHash']]
    
    #Prepare to move data to csv_df
    temp_df = image_df.drop_duplicates(subset='Accession_Number')
    #Remove useless data
    image_df = image_df[['Patient_ID', 
             'Accession_Number', 
             'ImageName',
             'RegionSpatialFormat', 
             'RegionCount',
             'RegionDataType', 
             'RegionLocationMinX0', 
             'RegionLocationMinY0', 
             'RegionLocationMaxX1', 
             'RegionLocationMaxY1',
             'PhotometricInterpretation',
             'Rows',
             'Columns',
             'FileName',
             'DicomHash']]
    
    # Update the list of parsed files and save it
    parsed_files_list.extend(dcm_files_list)
    content = '\n'.join(parsed_files_list)  # Convert list to string with newlines
    save_data(content, parsed_files_list_file)

    # Find all csv files and combine into df
    csv_df = read_csv(anon_location)
    
    # group the dataframe by Patient_ID and Accession_Number
    csv_df = csv_df.sort_values('Patient_ID')
    grouped_df = csv_df.groupby(['Patient_ID','Accession_Number'])
    csv_df = grouped_df.agg({'Biopsy_Accession': list,
                            'Biopsy_Laterality': list,
                            'BI-RADS': 'first',
                            'Study_Laterality': 'first',
                            'Biopsy': list,
                            'Path_Desc': list,
                            'Density_Desc': list,
                            'Time_Biop': 'last',
                            #'Facility': 'first',
                            'Age': 'first', 
                            'Race': 'first', 
                            'Ethnicity': 'first'}).reset_index()

    
    # Convert 'Patient_ID' to str in both dataframes before merging
    temp_df[['Patient_ID', 'Accession_Number']] = temp_df[['Patient_ID', 'Accession_Number']].astype(int)
    csv_df[['Patient_ID', 'Accession_Number']] = csv_df[['Patient_ID', 'Accession_Number']].astype(int)
    csv_df = pd.merge(csv_df, temp_df[['Patient_ID', 'Accession_Number', 'StudyDescription', 'StudyDate', 'PatientSex', 'PatientSize', 'PatientWeight']], on=['Patient_ID', 'Accession_Number'], how='inner')
    
    
    # Get count of duplicate rows for each Patient_ID in df
    duplicate_count = image_df.groupby('Patient_ID').size()
    duplicate_count = duplicate_count.reset_index(name='Image_Count')
    duplicate_count['Patient_ID'] = duplicate_count['Patient_ID'].astype(int)

    # Merge duplicate_count with csv_df
    csv_df = pd.merge(csv_df, duplicate_count, on='Patient_ID', how='left')
    #csv_df.to_csv("D:\DATA\CASBUSI/temp.csv", index=False)

        
    # Create Breast level data
    breast_csv = csv_df[['Patient_ID', 'Accession_Number']].copy()

    # Duplicate the rows and add 'Breast' column
    breast_csv['Breast'] = 'LEFT'
    breast_csv_right = breast_csv.copy()
    breast_csv_right['Breast'] = 'RIGHT'

    # Concatenate the original and duplicate rows
    breast_csv = pd.concat([breast_csv, breast_csv_right], ignore_index=True)
    breast_csv = breast_csv.sort_values('Accession_Number')
    
    breast_csv['Path_Desc'] = None
    breast_csv['Density_Desc'] = None
    breast_csv['Has_Malignant'] = False
    breast_csv['Has_Benign'] = False
    breast_csv['Has_Unknown'] = False
    
    #breast_csv.to_csv(f'{env}/breast1.csv', index=False)


    for idx, row in csv_df.iterrows():
        # Check if Biopsy_Laterality is a list-like object
        if isinstance(row['Biopsy_Laterality'], (list, np.ndarray)):
            for i, laterality in enumerate(row['Biopsy_Laterality']):
                if isinstance(laterality, str):
                    matching_rows = (breast_csv['Patient_ID'] == row['Patient_ID']) & \
                                (breast_csv['Accession_Number'] == row['Accession_Number']) & \
                                (breast_csv['Breast'] == laterality.upper())
                    
                    if isinstance(row['Biopsy'], list) and len(row['Biopsy']) > i:
                        biopsy_result = biopsy_mapping.get(row['Biopsy'][i], 'unknown')
                        if biopsy_result == 'malignant':
                            breast_csv.loc[matching_rows, 'Has_Malignant'] = True
                        elif biopsy_result == 'benign':
                            breast_csv.loc[matching_rows, 'Has_Benign'] = True
                        elif biopsy_result == 'unknown':
                            breast_csv.loc[matching_rows, 'Has_Unknown'] = True

                    if isinstance(row['Path_Desc'], list) and len(row['Path_Desc']) > i:
                        path_desc = row['Path_Desc'][i]
                        if pd.notna(breast_csv.loc[matching_rows, 'Path_Desc'].iloc[0]):
                            existing_path_desc = breast_csv.loc[matching_rows, 'Path_Desc'].iloc[0]
                            try:
                                path_desc_list = ast.literal_eval(existing_path_desc)
                                path_desc_list.append(path_desc)
                                breast_csv.loc[matching_rows, 'Path_Desc'] = str(path_desc_list)
                            except (ValueError, SyntaxError):
                                breast_csv.loc[matching_rows, 'Path_Desc'] = str([existing_path_desc, path_desc])
                        else:
                            breast_csv.loc[matching_rows, 'Path_Desc'] = str([path_desc])
                    
                    if isinstance(row['Density_Desc'], list) and len(row['Density_Desc']) > i:
                        density_desc = row['Density_Desc'][i]
                        if pd.notna(breast_csv.loc[matching_rows, 'Density_Desc'].iloc[0]):
                            existing_density_desc = breast_csv.loc[matching_rows, 'Density_Desc'].iloc[0]
                            try:
                                density_desc_list = ast.literal_eval(existing_density_desc)
                                density_desc_list.append(density_desc)
                                breast_csv.loc[matching_rows, 'Density_Desc'] = str(density_desc_list)
                            except (ValueError, SyntaxError):
                                breast_csv.loc[matching_rows, 'Density_Desc'] = str([existing_density_desc, density_desc])
                        else:
                            breast_csv.loc[matching_rows, 'Density_Desc'] = str([density_desc])
    
    #breast_csv.to_csv(f'{env}/breast2.csv', index=False)
    
    # Count lesions
    # Create a DataFrame that records the 'Biopsy_Laterality' for each 'Patient_ID'
    lesion_df = csv_df[['Patient_ID', 'Biopsy_Laterality']].explode('Biopsy_Laterality')
    # Count the lesions for each 'Patient_ID' and 'Biopsy_Laterality'
    lesion_count = lesion_df.groupby(['Patient_ID', 'Biopsy_Laterality']).size().reset_index(name='LesionCount')
    # Rename 'Biopsy_Laterality' to 'Breast' in lesion_count
    lesion_count = lesion_count.rename(columns={'Biopsy_Laterality': 'Breast'})
    # Merge lesion_count with breast_csv
    breast_csv = pd.merge(breast_csv, lesion_count, on=['Patient_ID', 'Breast'], how='left')
    
    
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

    if os.path.isfile(case_study_csv_file):
        existing_case_study_df = pd.read_csv(case_study_csv_file)
        csv_df = pd.concat([existing_case_study_df, csv_df])
        csv_df = csv_df.sort_values('Accession_Number').drop_duplicates('Accession_Number', keep='last')
        csv_df = csv_df.reset_index(drop=True)
        
    if os.path.isfile(breast_csv_file):
        existing_breast_df = pd.read_csv(breast_csv_file)
        breast_csv = pd.concat([existing_breast_df, breast_csv], ignore_index=True)
        breast_csv = breast_csv.sort_values(['Patient_ID', 'Accession_Number', 'Breast'])
        breast_csv = breast_csv.drop_duplicates(subset=['Patient_ID', 'Accession_Number', 'Breast'], keep='last')
        breast_csv = breast_csv.reset_index(drop=True)

    # Export the DataFrames to CSV files
    save_data(image_combined_df, image_csv_file)
    save_data(video_df, video_csv_file)
    save_data(csv_df, case_study_csv_file)
    save_data(breast_csv, breast_csv_file)
    
    
    
    









def UpdateAnonFile(anon_location):
    
    print("Updating Data with Anon File")
    
    parsed_database = f'{env}/database/'
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    breast_csv_file = f'{parsed_database}BreastData.csv'
    
    if not file_exists(case_study_csv_file):
        return
    
    
    # Case Data
    existing_case_study_df = read_csv(case_study_csv_file)
    csv_df = read_csv(anon_location)
    
    # group the dataframe by Patient_ID and Accession_Number
    csv_df = csv_df.sort_values('Patient_ID')
    grouped_df = csv_df.groupby(['Patient_ID','Accession_Number'])
    csv_df = grouped_df.agg({'Biopsy_Accession': list,
                            'Biopsy_Laterality': list,
                            'BI-RADS': 'first',
                            'Study_Laterality': 'first',
                            'Biopsy': list,
                            'Path_Desc': list,
                            'Density_Desc': list,
                            'Time_Biop': 'last',
                            #'Facility': 'first',
                            'Age': 'first', 
                            'Race': 'first', 
                            'Ethnicity': 'first'}).reset_index()

    
    # Convert 'Patient_ID' to str in both dataframes before merging
    existing_case_study_df['Patient_ID'] = existing_case_study_df['Patient_ID'].astype(int)
    csv_df['Patient_ID'] = csv_df['Patient_ID'].astype(int)
    csv_df = pd.merge(csv_df, existing_case_study_df[['Patient_ID', 'StudyDate', 'PatientSex', 'PatientSize', 'PatientWeight', 'Image_Count']], on='Patient_ID', how='inner')
    

    if 'Time_Biop' not in existing_case_study_df.columns:
        existing_case_study_df['Time_Biop'] = np.nan
    #existing_case_study_df.update(csv_df)
    # Set index to ['Patient_ID', 'Accession_Number']
    existing_case_study_df.set_index(['Patient_ID', 'Accession_Number'], inplace=True)
    csv_duplicate = csv_df.copy()
    csv_duplicate.set_index(['Patient_ID', 'Accession_Number'], inplace=True)
    

    # Drop the rows in existing_case_study_df that are in csv_df
    existing_case_study_df = existing_case_study_df.drop(csv_duplicate.index, errors='ignore')

    # Append csv_df to existing_case_study_df
    existing_case_study_df = existing_case_study_df.append(csv_duplicate)

    # Reset index
    existing_case_study_df.reset_index(inplace=True)
    
    # Breast Data
    # Create Breast level data
    breast_csv = csv_df[['Patient_ID', 'Accession_Number']].copy()

    # Duplicate the rows and add 'Breast' column
    breast_csv['Breast'] = 'LEFT'
    breast_csv_right = breast_csv.copy()
    breast_csv_right['Breast'] = 'RIGHT'

    # Concatenate the original and duplicate rows
    breast_csv = pd.concat([breast_csv, breast_csv_right], ignore_index=True)
    breast_csv = breast_csv.sort_values('Accession_Number')
    
    breast_csv['Path_Desc'] = None
    breast_csv['Density_Desc'] = None
    breast_csv['Has_Malignant'] = False
    breast_csv['Has_Benign'] = False
    breast_csv['Has_Unknown'] = False
    
    for idx, row in csv_df.iterrows():
        if row['Biopsy_Laterality'] is not None:
            for i, laterality in enumerate(row['Biopsy_Laterality']):
                if isinstance(laterality, str):
                    matching_rows = (breast_csv['Patient_ID'] == row['Patient_ID']) & (breast_csv['Accession_Number'] == row['Accession_Number']) & (breast_csv['Breast'] == laterality.upper())
                    if isinstance(row['Biopsy'], list) and len(row['Biopsy']) > i:
                        biopsy_result = biopsy_mapping.get(row['Biopsy'][i], 'unknown')
                        if biopsy_result == 'malignant':
                            breast_csv.loc[matching_rows, 'Has_Malignant'] = True
                        elif biopsy_result == 'benign':
                            breast_csv.loc[matching_rows, 'Has_Benign'] = True
                        elif biopsy_result == 'unknown':
                            breast_csv.loc[matching_rows, 'Has_Unknown'] = True
                        if not (breast_csv.loc[matching_rows, 'Has_Malignant'] | breast_csv.loc[matching_rows, 'Has_Benign'] | breast_csv.loc[matching_rows, 'Has_Unknown']).any():
                            print(biopsy_result)
                    if isinstance(row['Path_Desc'], list) and len(row['Path_Desc']) > i:
                        breast_csv.loc[matching_rows, 'Path_Desc'] = row['Path_Desc'][i]
                    if isinstance(row['Density_Desc'], list) and len(row['Density_Desc']) > i:
                        breast_csv.loc[matching_rows, 'Density_Desc'] = row['Density_Desc'][i]
    
    # Count lesions
    # Create a DataFrame that records the 'Biopsy_Laterality' for each 'Patient_ID'
    lesion_df = csv_df[['Patient_ID', 'Accession_Number', 'Biopsy_Laterality']].explode('Biopsy_Laterality')
    # Count the lesions for each 'Patient_ID' and 'Biopsy_Laterality'
    lesion_count = lesion_df.groupby(['Patient_ID', 'Accession_Number', 'Biopsy_Laterality']).size().reset_index(name='LesionCount')

    # Rename 'Biopsy_Laterality' to 'Breast' in lesion_count
    lesion_count = lesion_count.rename(columns={'Biopsy_Laterality': 'Breast'})
    # Merge lesion_count with breast_csv
    breast_csv = pd.merge(breast_csv, lesion_count, on=['Patient_ID', 'Accession_Number', 'Breast'], how='left')
    
    breast_csv = breast_csv.drop(columns=['unknown', 'unknown_x', 'unknown_y'], errors='ignore')
    breast_csv = breast_csv.drop_duplicates(subset=['Patient_ID', 'Accession_Number', 'Breast'], keep='last')
    
    existing_case_study_df = existing_case_study_df.drop_duplicates(subset=['Patient_ID', 'Accession_Number'], keep='last')
    
    save_data(breast_csv, breast_csv_file)
    save_data(existing_case_study_df, case_study_csv_file)