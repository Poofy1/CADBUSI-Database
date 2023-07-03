import os, pydicom, zipfile, hashlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import warnings

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
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
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


def parse_dcm_files(dcm_files_list, parsed_database):
    print("Parsing DCM Data")
    data_list = []
    for i, dcm in tqdm(enumerate(dcm_files_list[:50]), total=len(dcm_files_list)):
        
        # Skip any data that is not an image
        media_type = os.path.basename(dcm)[:5]
        if media_type != 'image':
            continue
            
        data_dict = {}
        dataset = pydicom.dcmread(dcm)

        # Traverse the DICOM dataset
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

        # Save image
        im = Image.fromarray(dataset.pixel_array)
        if data_dict.get('PhotometricInterpretation', '') == 'RGB':
            im = im.convert("RGB")
        else:
            im = im.convert("L")  # Convert to grayscale
        image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{i}.png"
        im.save(f"{parsed_database}images/{image_name}")


        # Add custom data
        data_dict['FileName'] = os.path.basename(dcm)
        data_dict['ImageName'] = image_name
        data_dict['DicomHash'] = generate_hash(dcm)

        # Append the dictionary to the list
        data_list.append(data_dict)
        
    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_list)






# Main Method
def Parse_Zip_Files(input, raw_storage_database):
    parsed_database = f'{env}/database/'

    #Create database dir
    os.makedirs(parsed_database, exist_ok = True)
    os.makedirs(f'{parsed_database}/images/', exist_ok = True)
        
        
    # Unzip input data and get every Dicom File
    extract_zip_files(input, raw_storage_database)
    dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
    print(f'Dicom Files: {len(dcm_files_list)}')

    # Get DCM Data
    image_df = parse_dcm_files(dcm_files_list, parsed_database)
    image_df = image_df.rename(columns={'PatientID': 'Patient_ID'})
    image_df = image_df.rename(columns={'AccessionNumber': 'Accession_Number'})
    #Prepare to move data to csv_df
    temp_df = image_df.drop_duplicates(subset='Patient_ID')
    #Remove useless data
    image_df = image_df[['Patient_ID', 
             'Accession_Number', 
             'ImageName',
             'StudyDescription', 
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
    temp_df['Patient_ID'] = temp_df['Patient_ID'].astype(str)
    csv_df['Patient_ID'] = csv_df['Patient_ID'].astype(str)
    csv_df = pd.merge(csv_df, temp_df[['Patient_ID', 'StudyDate', 'PatientSex', 'PatientSize', 'PatientWeight']], on='Patient_ID', how='left')

    # Get count of duplicate rows for each Patient_ID in df
    duplicate_count = image_df.groupby('Patient_ID').size()
    duplicate_count = duplicate_count.reset_index(name='Image_Count')

    # Merge duplicate_count with csv_df
    csv_df = pd.merge(csv_df, duplicate_count, on='Patient_ID', how='left')

    # Export the DataFrame to a CSV file
    image_df.to_csv(f'{parsed_database}ImageData.csv', index=False)
    csv_df.to_csv(f'{parsed_database}CaseStudyData.csv', index=False)
    
    

def Transfer_Laterality():
    
    csv_df_path = f"{env}/database/CaseStudyData.csv"
    image_df_path = f"{env}/database/ImageData.csv"
    
    csv_df = pd.read_csv(csv_df_path)
    image_df = pd.read_csv(image_df_path)
    
    temp_df = image_df.drop_duplicates(subset='Patient_ID')
    csv_df = pd.merge(csv_df, temp_df[['Patient_ID', 'laterality']], on='Patient_ID', how='left')
    
    image_df = image_df.drop('laterality', axis=1)
    
    csv_df.to_csv(csv_df_path, index=False)
    image_df.to_csv(image_df_path, index=False)