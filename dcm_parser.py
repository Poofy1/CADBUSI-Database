import os, pydicom, zipfile, hashlib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def parse_single_dcm(dcm, current_index, parsed_database):
    # Skip any data that is not an image
    media_type = os.path.basename(dcm)[:5]
    if media_type != 'image':
        return None
            
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
    image_name = f"{data_dict.get('PatientID', '')}_{data_dict.get('AccessionNumber', '')}_{current_index}.png"
    im.save(f"{parsed_database}/images/{image_name}")

    # Add custom data
    data_dict['FileName'] = os.path.basename(dcm)
    data_dict['ImageName'] = image_name
    data_dict['DicomHash'] = generate_hash(dcm)
    
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

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(parse_single_dcm, dcm, i+current_index, parsed_database): dcm for i, dcm in enumerate(dcm_files_list)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data = future.result()
            except Exception as exc:
                print(f'An exception occurred: {exc}')
            else:
                if data is not None:
                    data_list.append(data)

    # Save index
    with open(index_file, "w") as file:
        file.write(str(current_index + len(data_list)))

    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_list)





# Main Method
def Parse_Zip_Files(input, raw_storage_database, data_range):
    parsed_database = f'{env}/database/'

    #Create database dir
    os.makedirs(parsed_database, exist_ok = True)
    os.makedirs(f'{parsed_database}/images/', exist_ok = True)
    
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
    dcm_files_list = [file for file in dcm_files_list if file not in parsed_files_list]
    dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
    print(f'New Dicom Files: {len(dcm_files_list)}')

    if len(dcm_files_list) <= 0:
        return
    
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
    temp_df['Patient_ID'] = temp_df['Patient_ID'].astype(str)
    csv_df['Patient_ID'] = csv_df['Patient_ID'].astype(str)
    csv_df = pd.merge(csv_df, temp_df[['Patient_ID', 'StudyDate', 'PatientSex', 'PatientSize', 'PatientWeight']], on='Patient_ID', how='inner')

    # Get count of duplicate rows for each Patient_ID in df
    duplicate_count = image_df.groupby('Patient_ID').size()
    duplicate_count = duplicate_count.reset_index(name='Image_Count')

    # Merge duplicate_count with csv_df
    csv_df = pd.merge(csv_df, duplicate_count, on='Patient_ID', how='left')

    # Check if CSV files already exist
    image_csv_file = f'{parsed_database}ImageData.csv'
    case_study_csv_file = f'{parsed_database}CaseStudyData.csv'
    if os.path.isfile(image_csv_file):
        existing_image_df = pd.read_csv(image_csv_file)
        image_df = pd.concat([existing_image_df, image_df], ignore_index=True)

    if os.path.isfile(case_study_csv_file):
        existing_case_study_df = pd.read_csv(case_study_csv_file)
        csv_df = pd.concat([existing_case_study_df, csv_df], ignore_index=True)

    # Export the DataFrames to CSV files
    image_df.to_csv(image_csv_file, index=False)
    csv_df.to_csv(case_study_csv_file, index=False)
    
    

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