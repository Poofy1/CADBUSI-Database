import os, pydicom, hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import io
from tqdm import tqdm
import warnings, logging, cv2
from storage_adapter import *
from src.encrypt_keys import *
from src.DB_processing.tools import append_audit
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))


ENCRYPTION_KEY = None


# Define sets at module level for better performance
NAMES_TO_REMOVE = {
    'SOP Instance UID', 'Study Time', 'Series Time', 'Content Time',
    'Study Instance UID', 'Series Instance UID', 'Private Creator',
    'Media Storage SOP Instance UID', 'Implementation Class UID',
    "Patient's Name", "Referring Physician's Name", "Acquisition DateTime",
    "Institution Name", "Station Name", "Physician(s) of Record",
    "Referenced SOP Class UID", "Referenced SOP Instance UID",
    "Device Serial Number", "Patient Comments", "Issuer of Patient ID",
    "Study ID", "Study Comments", "Current Patient Location",
    "Requested Procedure ID", "Performed Procedure Step ID",
    "Other Patient IDs", "Operators' Name", "Institutional Department Name",
    "Manufacturer", "Requesting Physician",
}

NAMES_TO_ANON_TIME = {
    'Study Time', 'Series Time', 'Content Time',
}

def anon_callback(ds, element):
    # Check if the element name is in the removal set
    if element.name in NAMES_TO_REMOVE:
        del ds[element.tag]
        
    # Handle date elements
    if element.VR == "DA":
        element.value = element.value[0:4] + "0101"  # set all dates to YYYY0101
    # Handle time elements not in the exception list
    elif element.VR == "TM" and element.name not in NAMES_TO_ANON_TIME:
        element.value = "000000"  # set time to zeros



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


def deidentify_dicom(ds, dcm):
    
    global ENCRYPTION_KEY
    
    ds.PatientID = encrypt_single_id(ENCRYPTION_KEY, ds.PatientID)
    ds.AccessionNumber = encrypt_single_id(ENCRYPTION_KEY, ds.AccessionNumber)

    ds.remove_private_tags()  # take out private tags added by notion or otherwise
    
    # Avoid separate walks by combining them
    ds.walk(anon_callback)
    # Only walk file_meta if it exists
    if hasattr(ds, 'file_meta') and ds.file_meta is not None:
        ds.file_meta.walk(anon_callback)

    # Safely check for Multi-frame content
    is_video = False
    is_secondary = False
    try:
        if hasattr(ds, 'file_meta') and 0x00020002 in ds.file_meta:
            media_type = ds.file_meta[0x00020002]
            is_video = str(media_type).find('Multi-frame') > -1
            is_secondary = 'Secondary' in str(media_type)
        # Additional check for multi-frame files
        elif hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
            is_video = True
            
        # Method 3: Check SOP Class UID (backup method)
        elif hasattr(ds, 'SOPClassUID') and 'Multi-frame' in str(ds.SOPClassUID):
            is_video = True
    except Exception as e:
        print(f"Error determining media type: {e}")
        print(ds)

    y0 = 101
    
    if not is_secondary and (0x0018, 0x6011) in ds:
            y0 = ds['SequenceOfUltrasoundRegions'][0]['RegionLocationMinY0'].value

    if 'OriginalAttributesSequence' in ds:
        del ds.OriginalAttributesSequence
        
    # Check if Pixel Data is compressed - safely check TransferSyntaxUID
    is_compressed = False
    if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
        is_compressed = ds.file_meta.TransferSyntaxUID.is_compressed
    else:
        return None, is_video

    # Attempt to decompress if needed
    if is_compressed:
        try:
            ds.decompress()
        except Exception as e:
            print(f"Decompression error: {e}")
            return None
        
    # crop patient info above US region 
    arr = ds.pixel_array
    
    if is_video:
        arr[:,:y0] = 0
    else:
        arr[:y0] = 0
    
    # Update the Pixel Data
    ds.PixelData = arr.tobytes()
    
    # Keep the original transfer syntax
    ds.file_meta.TransferSyntaxUID = ds.file_meta.TransferSyntaxUID

    return ds, is_video



def parse_video_data(dcm, dcm_data, dataset, current_index, parsed_database):
    data_dict = {}
    
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

    # Read the DICOM file
    dcm_data = read_binary(dcm)
    dataset = pydicom.dcmread(io.BytesIO(dcm_data), force=True)
    
    # Anonymize 
    dataset, is_video = deidentify_dicom(dataset, dcm)
    
    if (dataset is None):
        return None
    
    # Safely check for Multi-frame content
    if is_video:
        return parse_video_data(dcm, dcm_data, dataset, current_index, parsed_database)

    
    
    # Continue with image
    
    data_dict = {}
    region_count = 0
    
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
    
    failure_counter = 0  # Initialize failure counter

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(parse_single_dcm, dcm, i+current_index, parsed_database): dcm for i, dcm in enumerate(dcm_files_list)}
        data_list = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            try:
                data = future.result()
                if data is None:
                    failure_counter += 1  # Count as failure if None is returned
                else:
                    data_list.append(data)
            except Exception as exc:
                failure_counter += 1  # Count exceptions as failures too
                print(f'An exception occurred: {exc}')

        # Save index - only count successful parses
        new_index = str(current_index + len(data_list))
        save_data(new_index, index_file)

    # Print total failures at the end
    if failure_counter > 0:
        print(f'Total failures: {failure_counter}')

    # Create a DataFrame from the list of dictionaries (only successful parses)
    df = pd.DataFrame(data_list)
    return df


def parse_anon_file(anon_location, database_path, image_df, ):
    
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
    anon_location = os.path.normpath(anon_location)
    breast_csv = pd.read_csv(anon_location)
    breast_csv = breast_csv.sort_values('PATIENT_ID')
    breast_csv = breast_csv.rename(columns={
        'PATIENT_ID': 'Patient_ID',
        'ACCESSION_NUMBER': 'Accession_Number'
    })

    
    # Convert 'Patient_ID' to str in both dataframes before merging
    image_df[['Patient_ID', 'Accession_Number']] = image_df[['Patient_ID', 'Accession_Number']].astype(str)
    breast_csv[['Patient_ID', 'Accession_Number']] = breast_csv[['Patient_ID', 'Accession_Number']].astype(str)
    
    #image_df.to_csv(f"{env}/image_df.csv", index=False) # DEBUG
    #anon_df.to_csv(f"{env}/anon_df.csv", index=False) # DEBUG
    
    # Populate Has_Malignant and Has_Benign based on final_interpretation
    breast_csv['Has_Malignant'] = breast_csv['final_interpretation'] == 'MALIGNANT'
    breast_csv['Has_Benign'] = breast_csv['final_interpretation'] == 'BENIGN'
    
    image_csv_file = f'{database_path}ImageData.csv'
    video_csv_file = f'{database_path}VideoData.csv'
    breast_csv_file = f'{database_path}BreastData.csv'
    
    image_combined_df = image_df
    if file_exists(image_csv_file):
        existing_image_df = read_csv(image_csv_file)
        existing_image_df['Patient_ID'] = existing_image_df['Patient_ID'].astype(str)
        image_df['Patient_ID'] = image_df['Patient_ID'].astype(str)
        
        # keep only old IDs that don't exist in new data
        image_df = image_df[~image_df['Patient_ID'].isin(existing_image_df['Patient_ID'].unique())]
        
        # now concatenate the old data with the new data
        image_combined_df = pd.concat([existing_image_df, image_df], ignore_index=True)

        
    if file_exists(video_csv_file):
        existing_video_df = read_csv(video_csv_file)
        video_df = pd.concat([existing_video_df, video_df], ignore_index=True)


    if file_exists(breast_csv_file):
        existing_breast_df = read_csv(breast_csv_file)
        breast_csv = pd.concat([existing_breast_df, breast_csv], ignore_index=True)
        breast_csv = breast_csv.sort_values(['Patient_ID', 'Accession_Number', 'Breast'])
        breast_csv = breast_csv.drop_duplicates(subset=['Patient_ID', 'Accession_Number', 'Breast'], keep='last')
        breast_csv = breast_csv.reset_index(drop=True)

    # Export the DataFrames to CSV files
    save_data(image_combined_df, image_csv_file)
    save_data(video_df, video_csv_file)
    save_data(breast_csv, breast_csv_file)
    append_audit(database_path, f"Saved data: {len(image_combined_df)} images, {len(video_df)} videos, {len(breast_csv)} breast records")
    

# Main Method
def Parse_Dicom_Files(database_path, anon_location, raw_storage_database, data_range, encryption_key):
    
    # Set the global encryption key
    global ENCRYPTION_KEY
    ENCRYPTION_KEY = encryption_key
    
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
    append_audit(database_path, f"Found {len(dcm_files_list)} total DICOM files")
    print(f'Total Dicoms in Input: {len(dcm_files_list)}')

    # Apply data range only if it's specified
    if data_range and len(data_range) == 2:
        dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
        print(f'Applied Data Range: {len(dcm_files_list)}')
        append_audit(database_path, f"Applied data limit: processing {len(dcm_files_list)} of {len(dcm_files_list)} files")
        
        
    # Filter out already processed files
    files_before_filter = len(dcm_files_list)
    parsed_files_set = set(parsed_files_list)
    dcm_files_list = [file for file in dcm_files_list if file not in parsed_files_set]
    files_skipped = files_before_filter - len(dcm_files_list)
    append_audit(database_path, f"Skipped {files_skipped} previously processed files")

    
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
    append_audit(database_path, f"Removed {removed_rows} DICOMs - Missing Patient_ID or Accession_Number")
    
    
    parse_anon_file(anon_location, database_path, image_df)
    
    