import os, pydicom, zipfile, hashlib
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from highdicom.io import ImageFileReader
import warnings
import logging
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))


def generate_hash(filename):
    sha256_hash = hashlib.sha256()

    with open(filename,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_zip_files(input, output):
    #Create database dir
    os.makedirs(output, exist_ok = True)
    
    if (len(os.listdir(input)) == 0):
        print("No zip files found")
        return
    
    print("Unzipping Files")
    for item in tqdm(os.listdir(input)): # loop through items in dir
        if item.endswith('.zip'): # check for ".zip" extension
            file_name = os.path.abspath(input) + "/" + item # get full path of files
            try:
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            except:
                print(f'Skipping Bad Zip File: {file_name}')
                continue
            zip_ref.extractall(output) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file


def get_files_by_extension(directory, extension):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))

    return file_paths



# this part sucks so bad because we are loading the entire dcm file and then passing it here. The dcm file holds ALL THE FRAMES. 
# This plus we have it one each thread, very bad.
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
        for i in range(image.number_of_frames):
            if i % 25 == 0:
                frame = image.read_frame(i, correct_color=False)
                im = Image.fromarray(frame)
                
                image_name = f"{i}.png"
                
                im = im.convert("L")  # Convert to grayscale
                
                im.save(f"{parsed_database}/videos/{video_path}/{image_name}")
                
                image_count += 1
                

    # Add custom data
    data_dict['DataType'] = 'video'
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImagesPath'] = video_path
    data_dict['SavedFrames'] = image_count
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
    im = Image.fromarray(dataset.pixel_array)
    if data_dict.get('PhotometricInterpretation', '') == 'RGB':
        np_im = np.array(im)
        
        # check if there is any blue pixel
        is_blue = (np_im[:, :, 0] < 50) & (np_im[:, :, 1] < 50) & (np_im[:, :, 2] > 200)
        if np.any(is_blue):
            im = im.convert("RGB")
        else:
            im = im.convert("L")  # Convert to grayscale
            data_dict['PhotometricInterpretation'] = 'MONOCHROME2_OVERRIDE'
    else:
        im = im.convert("L")  # Convert to grayscale
    image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}.png"
    im.save(f"{parsed_database}/images/{image_name}")

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
    csv_files_list = get_files_by_extension(raw_storage_database, '.csv')
    dataframes = [pd.read_csv(csv_file) for csv_file in csv_files_list]
    csv_df = pd.concat(dataframes, ignore_index=True)
    # group the dataframe by Patient_ID and Accession_Number
    grouped_df = csv_df.groupby(['Patient_ID','Accession_Number'])
    csv_df = grouped_df.agg({'BI-RADS': 'first',
                            'Biopsy': list,
                            'Path_Desc': list,
                            'Density_Desc': list,
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

    # Check if CSV files already exist
    image_csv_file = f'{parsed_database}ImageData.csv'
    video_csv_file = f'{parsed_database}VideoData.csv'
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv' 
    
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

    # Export the DataFrames to CSV files
    image_combined_df.to_csv(image_csv_file, index=False)
    video_df.to_csv(video_csv_file, index=False)
    csv_df.to_csv(case_study_csv_file, index=False)
    
    

def Transfer_Laterality():
    
    csv_df_path = f"{env}/database/CaseStudyData.csv"
    image_df_path = f"{env}/database/ImageData.csv"
    
    csv_df = pd.read_csv(csv_df_path)
    image_df = pd.read_csv(image_df_path)
    
    # create a dictionary to store the result
    patient_laterality = {}

    # group by Patient_ID
    for name, group in image_df.groupby('Patient_ID'):
        if 'unknown' in group['laterality'].values:
            patient_laterality[name] = 'unknown'
        elif 'left' in group['laterality'].values and 'right' in group['laterality'].values:
            patient_laterality[name] = 'bilateral'
        elif 'left' in group['laterality'].values:
            patient_laterality[name] = 'left'
        elif 'right' in group['laterality'].values:
            patient_laterality[name] = 'right'
        else:
            # in case there are other values we haven't accounted for
            patient_laterality[name] = 'unknown'
    
    # create a new column in csv_df based on patient_laterality dictionary
    csv_df['laterality'] = csv_df['Patient_ID'].map(patient_laterality)
    
    csv_df.to_csv(csv_df_path, index=False)