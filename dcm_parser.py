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


def parse_dcm_files(dcm_files_list):
    print("Parsing DCM Data")
    data_list = []
    for i, dcm in tqdm(enumerate(dcm_files_list[:10]), total=len(dcm_files_list)):
        
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
        im.save(f'{parsed_database}images/{i}.png')

        # Add the file name to the data
        data_dict['FileName'] = os.path.basename(dcm)
        # Add the file hash to the data
        data_dict['DicomHash'] = generate_hash(dcm)

        # Append the dictionary to the list
        data_list.append(data_dict)
        
    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_list)





input = f'{env}/zip_files/'
raw_storage_database = f'D:/DATA/CASBUSI/dicoms/'
parsed_database = f'{env}/zip_processed/'


#Create database dir
os.makedirs(parsed_database, exist_ok = True)
os.makedirs(f'{parsed_database}/images/', exist_ok = True)
    
    
# Unzip input data and get every Dicom File
extract_zip_files(input, raw_storage_database)
dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
print(f'Dicom Files: {len(dcm_files_list)}')


# Find all csv files and combine into df
csv_files_list = get_files_by_extension(raw_storage_database, '.csv')
dataframes = [pd.read_csv(csv_file) for csv_file in csv_files_list]
csv_df = pd.concat(dataframes, ignore_index=True)


# Get DCM Data
df = parse_dcm_files(dcm_files_list)



print(csv_df.columns)
print(df.columns)

# Export the DataFrame to a CSV file
df.to_csv(f'{parsed_database}output.csv', index=False)