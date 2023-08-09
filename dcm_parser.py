import os, pydicom, zipfile, hashlib
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from highdicom.io import ImageFileReader
import warnings, logging, cv2
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))


def generate_hash(filename):
    sha256_hash = hashlib.sha256()

    with open(filename,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_single_zip_file(file_name, output_folder):
    if os.path.exists(output_folder):
        return

    try:
        zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
        zip_ref.extractall(output_folder)  # extract file to dir
        zip_ref.close()  # close file
        #os.remove(file_name) # delete zipped file
    except Exception as e:
        print(f'Skipping Bad Zip File: {file_name}. Exception: {e}')

def extract_zip_files(input, output):
    os.makedirs(output, exist_ok=True)

    if len(os.listdir(input)) == 0:
        print("No zip files found")
        return

    print("Unzipping Files")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                extract_single_zip_file, 
                os.path.abspath(input) + "/" + item, 
                os.path.join(output, os.path.splitext(item)[0])
            ) 
            for item in os.listdir(input) 
            if item.endswith('.zip')
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            try:
                future.result()
            except Exception as exc:
                print(f'An exception occurred: {exc}')
            


def get_files_by_extension(directory, extension):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))

    return file_paths


def parse_video_data(dcm, current_index, parsed_database):

    data_dict = {}
    dataset = pydicom.dcmread(dcm, stop_before_pixels=True)
    
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
    os.makedirs(f"{parsed_database}/videos/{video_path}/", exist_ok = True)
    
    #get image frames
    image_count = 0
    
    with ImageFileReader(dcm) as image:
        # Calculate the index of the middle frame
        middle_frame_index = image.number_of_frames // 2

        # Save the first and middle frames
        for i in [0, middle_frame_index]:
            frame = image.read_frame(i, correct_color=False)
            
            # Convert to grayscale if the frame is not already grayscale
            if len(frame.shape) == 3:  # if the frame has 3 channels
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
            image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}_{image_count}.png"
            
            cv2.imwrite(f"{parsed_database}/videos/{video_path}/{image_name}", frame)
            
            image_count += 1

    # Add custom data
    data_dict['DataType'] = 'video'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImagesPath'] = video_path
    #data_dict['SavedFrames'] = image_count
    data_dict['DicomHash'] = generate_hash(dcm)
    
    return data_dict
    

def parse_single_dcm(dcm, current_index, parsed_database):
    
    media_type = os.path.basename(dcm)[:5]
    if media_type != 'image':
        return parse_video_data(dcm, current_index, parsed_database)
    
    data_dict = {}
    region_count = 0
    dataset = pydicom.dcmread(dcm)

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
        np_im = np.array(im)
        
        # check if there is any blue pixel
        is_blue = (np_im[:, :, 0] < 50) & (np_im[:, :, 1] < 50) & (np_im[:, :, 2] > 200)
        if np.any(is_blue):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            data_dict['PhotometricInterpretation'] = 'MONOCHROME2_OVERRIDE'

    image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}.png"
    cv2.imwrite(f"{parsed_database}/images/{image_name}", im)

    # Add custom data
    data_dict['DataType'] = 'image'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImageName'] = image_name
    data_dict['DicomHash'] = generate_hash(dcm)
    data_dict['RegionCount'] = region_count
    
    return data_dict





def parse_dcm_files(dcm_files_list, parsed_database):
    print("Parsing DCM Data")

    # Load the current index from a file
    index_file = os.path.join(parsed_database, "IndexCounter.txt")
    if os.path.isfile(index_file):
        with open(index_file, "r") as file:
            current_index = int(file.read())
    else:
        current_index = 0

    data_list = []
    lock = Lock()

    # Load existing parsed files from csv
    image_csv_file = f'{parsed_database}ImageData.csv'
    if os.path.isfile(image_csv_file):
        existing_df = pd.read_csv(image_csv_file)
        existing_files = set(existing_df['FileName'].values)
    else:
        existing_files = set()

    # Exclude already parsed files from the dcm_files_list
    #dcm_files_list = [dcm_file for dcm_file in dcm_files_list if os.path.basename(dcm_file) not in existing_files]
    print(f'New Dicom Files: {len(dcm_files_list)}')

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(parse_single_dcm, dcm, i+current_index, parsed_database): dcm for i, dcm in enumerate(dcm_files_list)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            try:
                data = future.result()
                with lock:
                    data_list.append(data)
                    # Save index
                    with open(index_file, "w") as file:
                        file.write(str(current_index + len(data_list)))
            except Exception as exc:
                print(f'An exception occurred: {exc}')

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df



# Main Method
def Parse_Zip_Files(input, raw_storage_database, data_range):
    parsed_database = f'{env}/database/'

    #Create database dir
    os.makedirs(parsed_database, exist_ok = True)
    os.makedirs(f'{parsed_database}/images/', exist_ok = True)
    os.makedirs(f'{parsed_database}/videos/', exist_ok = True)
    
    # Load the list of already parsed files
    parsed_files_list = []
    parsed_files_list_file = f"{parsed_database}/ParsedFiles.txt"
    if os.path.exists(parsed_files_list_file):
        with open(parsed_files_list_file, 'r') as file:
            parsed_files_list = file.read().splitlines()

    # Unzip input data and get every Dicom File
    extract_zip_files(input, raw_storage_database)
    dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
    print(f'Total Dicom Archive: {len(dcm_files_list)}')

    # Filter out the already parsed files
    dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
    dcm_files_list = [file for file in dcm_files_list if file not in parsed_files_list]
    
    

    if len(dcm_files_list) <= 0:
        return
    
    # Get DCM Data
    image_df = parse_dcm_files(dcm_files_list, parsed_database)
    image_df = image_df.rename(columns={'PatientID': 'Patient_ID'})
    image_df = image_df.rename(columns={'AccessionNumber': 'Accession_Number'})
    
    video_df = image_df[image_df['DataType'] == 'video']
    image_df = image_df[image_df['DataType'] == 'image']
    
    if not video_df.empty:
        video_df = video_df[['Patient_ID', 
                'Accession_Number', 
                'ImagesPath',
                #'SavedFrames',
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
    temp_df = image_df.drop_duplicates(subset='Patient_ID')
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
    with open(parsed_files_list_file, 'w') as file:
        for item in parsed_files_list:
            file.write('%s\n' % item)

    # Find all csv files and combine into df
    csv_df = pd.read_csv("D:/DATA/CASBUSI/cases_anon/total_cases_anon.csv")
    
    # group the dataframe by Patient_ID and Accession_Number
    grouped_df = csv_df.groupby(['Patient_ID','Accession_Number'])
    csv_df = grouped_df.agg({'Biopsy_Accession': list,
                            'Biopsy_Laterality': list,
                            'BI-RADS': 'first',
                            'Study_Laterality': 'first',
                            'Biopsy': list,
                            'Path_Desc': list,
                            'Density_Desc': list,
                            'Facility': 'first',
                            'Age': 'first', 
                            'Race': 'first', 
                            'Ethnicity': 'first'}).reset_index()

    
    # Convert 'Patient_ID' to str in both dataframes before merging
    temp_df['Patient_ID'] = temp_df['Patient_ID'].astype(int)
    csv_df['Patient_ID'] = csv_df['Patient_ID'].astype(int)
    csv_df = pd.merge(csv_df, temp_df[['Patient_ID', 'StudyDescription', 'StudyDate', 'PatientSex', 'PatientSize', 'PatientWeight']], on='Patient_ID', how='inner')
    
    # Get count of duplicate rows for each Patient_ID in df
    duplicate_count = image_df.groupby('Patient_ID').size()
    duplicate_count = duplicate_count.reset_index(name='Image_Count')
    duplicate_count['Patient_ID'] = duplicate_count['Patient_ID'].astype(int)

    # Merge duplicate_count with csv_df
    csv_df = pd.merge(csv_df, duplicate_count, on='Patient_ID', how='left')

        
    # Create Breast level data
    breast_csv = csv_df[['Patient_ID', 'Accession_Number']].copy()

    # Duplicate the rows and add 'Breast' column
    breast_csv['Breast'] = 'left'
    breast_csv_right = breast_csv.copy()
    breast_csv_right['Breast'] = 'right'

    # Concatenate the original and duplicate rows
    breast_csv = pd.concat([breast_csv, breast_csv_right], ignore_index=True)
    breast_csv = breast_csv.sort_values('Accession_Number')
    
    breast_csv['Path_Desc'] = None
    breast_csv['Density_Desc'] = None
    breast_csv['Has_Malignant'] = False
    breast_csv['Has_Benign'] = False
    breast_csv['Has_Unknown'] = False

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

    for idx, row in csv_df.iterrows():
        if row['Biopsy_Laterality'] is not None:
            for i, laterality in enumerate(row['Biopsy_Laterality']):
                if isinstance(laterality, str):
                    matching_rows = (breast_csv['Patient_ID'] == row['Patient_ID']) & \
                                    (breast_csv['Accession_Number'] == row['Accession_Number']) & \
                                    (breast_csv['Breast'] == laterality.lower())
                    if isinstance(row['Biopsy'], list) and len(row['Biopsy']) > i:
                        biopsy_result = biopsy_mapping.get(row['Biopsy'][i], 'unknown')
                        if biopsy_result == 'malignant':
                            breast_csv.loc[matching_rows, 'Has_Malignant'] = True
                        elif biopsy_result == 'benign':
                            breast_csv.loc[matching_rows, 'Has_Benign'] = True
                        elif biopsy_result == 'unknown':
                            breast_csv.loc[matching_rows, 'Has_Unknown'] = True
                    if isinstance(row['Path_Desc'], list) and len(row['Path_Desc']) > i:
                        breast_csv.loc[matching_rows, 'Path_Desc'] = row['Path_Desc'][i]
                    if isinstance(row['Density_Desc'], list) and len(row['Density_Desc']) > i:
                        breast_csv.loc[matching_rows, 'Density_Desc'] = row['Density_Desc'][i]

    # Count lesions
    lesion_count = csv_df['Biopsy_Laterality'].apply(lambda x: pd.Series(x).value_counts()).fillna(0)
    lesion_count['Patient_ID'] = csv_df['Patient_ID']
    lesion_count['Accession_Number'] = csv_df['Accession_Number']
    breast_csv = pd.merge(breast_csv, lesion_count, on=['Patient_ID', 'Accession_Number'], how='left')
    breast_csv['LesionCount'] = np.where(breast_csv['Breast'] == 'left', breast_csv['left'], breast_csv['right'])
    breast_csv = breast_csv.drop(['left', 'right'], axis=1)
    
    # Check if CSV files already exist
    image_csv_file = f'{parsed_database}ImageData.csv'
    video_csv_file = f'{parsed_database}VideoData.csv'
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    breast_csv_file = f'{parsed_database}BreastData.csv' 
    
    image_combined_df = image_df
    if os.path.isfile(image_csv_file):
        existing_image_df = pd.read_csv(image_csv_file)
        existing_image_df['Patient_ID'] = existing_image_df['Patient_ID'].astype(str)
        image_df['Patient_ID'] = image_df['Patient_ID'].astype(str)
        
        new_ids = image_df['Patient_ID'].unique()
        
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
        csv_df = csv_df.sort_values('Patient_ID').drop_duplicates('Patient_ID', keep='last')
        csv_df = csv_df.reset_index(drop=True)
        
    if os.path.isfile(breast_csv_file):
        existing_breast_df = pd.read_csv(breast_csv_file)
        breast_csv = pd.concat([existing_breast_df, breast_csv], ignore_index=True)
        breast_csv = breast_csv.sort_values(['Patient_ID', 'Accession_Number', 'Breast'])
        breast_csv = breast_csv.drop_duplicates(subset=['Patient_ID', 'Accession_Number', 'Breast'], keep='last')
        breast_csv = breast_csv.reset_index(drop=True)


    # Export the DataFrames to CSV files
    image_combined_df.to_csv(image_csv_file, index=False)
    video_df.to_csv(video_csv_file, index=False)
    csv_df.to_csv(case_study_csv_file, index=False)
    breast_csv.to_csv(breast_csv_file, index=False)