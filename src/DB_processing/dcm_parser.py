import os, pydicom
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import io
from tqdm import tqdm
import warnings, logging, cv2
from functools import lru_cache
from PIL import Image
import time
from storage_adapter import *
from src.encrypt_keys import *
from src.DB_processing.tools import append_audit
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')
env = os.path.dirname(os.path.abspath(__file__))

try:
    import gdcm
    GDCM_AVAILABLE = True
    print("GDCM loaded - enhanced DICOM decompression available")
except ImportError:
    GDCM_AVAILABLE = False
    print("GDCM not available - compressed DICOM handling may fail")


ENCRYPTION_KEY = None


# Define sets at module level for better performance
NAMES_TO_REMOVE = frozenset({
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
})

NAMES_TO_ANON_TIME = frozenset({
    'Study Time', 'Series Time', 'Content Time',
})

def anon_callback(ds, element):
    # Use faster membership testing with frozensets
    if element.name in NAMES_TO_REMOVE:
        del ds[element.tag]
    elif element.VR == "DA":
        element.value = element.value[:4] + "0101"
    elif element.VR == "TM" and element.name not in NAMES_TO_ANON_TIME:
        element.value = "000000"

# Optimized vectorized color detection using numba for JIT compilation
from numba import jit

@jit(nopython=True)
def has_blue_pixels(image, n=100, min_b=200):
    height, width = image.shape[:2]
    for i in range(0, height, 10):  # Sample every 10th pixel for speed
        for j in range(0, width, 10):
            b, g, r = image[i, j, 2], image[i, j, 1], image[i, j, 0]
            if b >= min_b and b - r >= n and b - g >= n:
                return True
    return False

@jit(nopython=True)
def has_red_pixels(image, n=100, min_r=200):
    height, width = image.shape[:2]
    for i in range(0, height, 10):  # Sample every 10th pixel for speed
        for j in range(0, width, 10):
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            if r >= min_r and r - b >= n and r - g >= n:
                return True
    return False

'''
def manual_decompress(ds):
    """
    Replacement for ds.decompress() for specific edge cases where normal decompression fails
    
    Parameters:
    - ds: pydicom Dataset that needs decompression
    
    Returns:
    - Modified dataset with uncompressed pixel data, or None if decompression fails
    """
    # Check if dataset has pixel data
    if 'PixelData' not in ds:
        print("No pixel data found in dataset")
        return None
    
    pixel_data = ds.PixelData
    
    # Find all JPEG frame start positions
    frame_starts = []
    pos = 0
    while pos < len(pixel_data) - 1:
        if pixel_data[pos] == 0xFF and pixel_data[pos+1] == 0xD8:
            frame_starts.append(pos)
            pos += 2
        else:
            pos += 1
    
    # Return None if no frames found
    if not frame_starts:
        print("No JPEG frames found in pixel data")
        return None
    
    # Get image dimensions from DICOM attributes
    rows = ds.Rows if 'Rows' in ds else 0
    columns = ds.Columns if 'Columns' in ds else 0
    
    # Check if we have enough information
    if rows == 0 or columns == 0:
        print("Missing image dimensions in DICOM")
        return None
    
    # Determine if multi-frame
    is_multi_frame = False
    if 'NumberOfFrames' in ds and ds.NumberOfFrames > 1:
        is_multi_frame = True

    # Process each frame
    frames = []
    for i in range(len(frame_starts)):
        # Calculate frame size
        start = frame_starts[i]
        end = frame_starts[i+1] if i < len(frame_starts)-1 else len(pixel_data)
        
        # Get frame data
        frame_data = pixel_data[start:end]
        
        # Ensure it ends with JPEG EOI marker
        jpeg_end = frame_data.rfind(b'\xFF\xD9')
        if jpeg_end > 0:
            frame_data = frame_data[:jpeg_end+2]
        
        # Skip invalid frames
        if frame_data[:2] != b'\xFF\xD8':
            continue
        
        try:
            # Open JPEG with PIL
            img = Image.open(io.BytesIO(frame_data))
            
            # Ensure dimensions match
            if img.width != columns or img.height != rows:
                # Resize if needed
                img = img.resize((columns, rows))
            
            # Convert to numpy array
            frame_array = np.array(img)
            
            # Append to frames list
            frames.append(frame_array)
        except Exception as e:
            print(f"Error processing frame {i+1}: {e}")
    
    # Return None if no frames were successfully decoded
    if not frames:
        print("No frames could be successfully decompressed")
        return None
    
    # Create uncompressed pixel data
    if is_multi_frame:
        # Stack frames for multi-frame
        pixel_array = np.stack(frames)
    else:
        # Single frame
        pixel_array = frames[0]
    
    # Update dataset
    try:
        # Calculate pixel representation based on array
        if pixel_array.dtype.kind == 'u':  # unsigned integer
            ds.PixelRepresentation = 0
        else:  # signed integer
            ds.PixelRepresentation = 1
        
        # Update bits stored if needed
        max_value = np.max(pixel_array)
        if max_value > 255:
            ds.BitsStored = 16
        else:
            ds.BitsStored = 8
        
        # Update high bit
        ds.HighBit = ds.BitsStored - 1
        
        # Update photometric interpretation if needed
        if len(pixel_array.shape) >= 3 and pixel_array.shape[-1] == 3:
            ds.PhotometricInterpretation = "RGB"
        else:
            # Default to MONOCHROME2 for grayscale
            if 'PhotometricInterpretation' not in ds:
                ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Set uncompressed transfer syntax
        if hasattr(ds, 'file_meta') and ds.file_meta is not None:
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        else:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID if 'SOPClassUID' in ds else '1.2.840.10008.5.1.4.1.1.4'  # Default to MR
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else pydicom.uid.generate_uid()
            ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
            ds.file_meta.ImplementationVersionName = 'MANUAL_DECOMPRESS'
        
        # Set pixel data from array
        ds.PixelData = pixel_array.tobytes()
        
        return ds
    except Exception as e:
        print(f"Error updating dataset: {e}")
        return None

'''
    
def deidentify_dicom(ds, is_video, is_secondary):
    
    global ENCRYPTION_KEY
    
    ds.PatientID = encrypt_single_id(ENCRYPTION_KEY, ds.PatientID)
    ds.AccessionNumber = encrypt_single_id(ENCRYPTION_KEY, ds.AccessionNumber)

    ds.remove_private_tags()  # take out private tags added by notion or otherwise
    
    # Avoid separate walks by combining them
    ds.walk(anon_callback)
    # Only walk file_meta if it exists
    if hasattr(ds, 'file_meta') and ds.file_meta is not None:
        ds.file_meta.walk(anon_callback)

    y0 = 101
    
    if not is_secondary and (0x0018, 0x6011) in ds:
            y0 = ds['SequenceOfUltrasoundRegions'][0]['RegionLocationMinY0'].value

    if 'OriginalAttributesSequence' in ds:
        del ds.OriginalAttributesSequence

    # Check if Pixel Data is compressed - safely check TransferSyntaxUID
    is_compressed = False
    if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
        is_compressed = ds.file_meta.TransferSyntaxUID.is_compressed
    elif (0x0028, 0x2110) in ds and ds[0x0028, 0x2110].value == '01':
        is_compressed = True
    else:
        print("couldn't determine if compressed")
        return None

    # Attempt to decompress if needed
    if is_compressed:
        try:
            ds.decompress()
        except Exception as e:
            print(f"GDCM decompression failed: {e}")
            try:
                arr = ds.pixel_array  # Sometimes this works even when decompress() fails
                print("Got pixel data despite decompression failure")
            except Exception as e2:
                print(f"Cannot access pixel data: {e2}")
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

    return ds



def parse_video_data(dcm, dataset, current_index, parsed_database, video_n_frames):
    
    if video_n_frames <= 0:
        return None
    
    # Extract Software Version
    software_version = ""
    if hasattr(dataset, 'SoftwareVersions'):
        software_version = dataset.SoftwareVersions
    manufacturer_model = ""
    if hasattr(dataset, 'ManufacturerModelName'):
        manufacturer_model = dataset.ManufacturerModelName
        
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
    for i in range(0, total_frames, video_n_frames):  # Process every nth frame
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
    data_dict['FileName'] = os.path.join(os.path.basename(os.path.dirname(dcm)), os.path.basename(dcm))
    data_dict['ImagesPath'] = video_path
    data_dict['SavedFrames'] = image_count
    data_dict['DicomHash'] = os.path.splitext(os.path.basename(dcm))[0]
    data_dict['SoftwareVersions'] = str(software_version)
    data_dict['ManufacturerModelName'] = str(manufacturer_model)
    
    return data_dict



def parse_single_dcm(dcm, current_index, parsed_database, video_n_frames, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Read the DICOM file
            dcm_data = read_binary(dcm)
            dataset = pydicom.dcmread(io.BytesIO(dcm_data), force=True)
            break  # Success, exit retry loop
        except Exception as e:
            if "Checksum mismatch" in str(e) and attempt < max_retries - 1:
                print(f"Checksum error on {dcm}, retrying... (attempt {attempt + 1})")
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                print(f"Failed to read {dcm} after {attempt + 1} attempts: {e}")
                return None
    
    # Not a ultrasound image, likely a image of the settings or some sketch notes
    if not hasattr(dataset, 'SequenceOfUltrasoundRegions'):
        return None
    
    # Safely check for Multi-frame content
    is_video = False
    is_secondary = False
    try:
        if hasattr(dataset, 'file_meta') and 0x00020002 in dataset.file_meta:
            media_type = dataset.file_meta[0x00020002]
            is_video = str(media_type).find('Multi-frame') > -1
            is_secondary = 'Secondary' in str(media_type)
        # Additional check for multi-frame files
        elif hasattr(dataset, 'NumberOfFrames') and dataset.NumberOfFrames > 1:
            is_video = True
        # Method 3: Check SOP Class UID (backup method)
        elif hasattr(dataset, 'SOPClassUID') and 'Multi-frame' in str(dataset.SOPClassUID):
            is_video = True
    except Exception as e:
        print(f"Error determining media type: {e}")
        
    if (is_video and video_n_frames == 0):
        return None
    
    # Anonymize 
    dataset = deidentify_dicom(dataset, is_video, is_secondary)
    
    if (dataset is None):
        return None
    
    # Safely check for Multi-frame content
    if is_video:
        return parse_video_data(dcm, dataset, current_index, parsed_database, video_n_frames)

    
    # Extract Software Version
    software_version = ""
    if hasattr(dataset, 'SoftwareVersions'):
        software_version = dataset.SoftwareVersions
    manufacturer_model = ""
    if hasattr(dataset, 'ManufacturerModelName'):
        manufacturer_model = dataset.ManufacturerModelName
    
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
    data_dict['FileName'] = os.path.join(os.path.basename(os.path.dirname(dcm)), os.path.basename(dcm))
    data_dict['ImageName'] = image_name
    data_dict['DicomHash'] = os.path.splitext(os.path.basename(dcm))[0]
    data_dict['RegionCount'] = region_count
    data_dict['SoftwareVersions'] = str(software_version)
    data_dict['ManufacturerModelName'] = str(manufacturer_model)
    
    return data_dict


def process_batch(batch_data):
    """Process a batch of DICOM files"""
    results = []
    for dcm, current_index, parsed_database, video_n_frames in batch_data:
        try:
            result = parse_single_dcm(dcm, current_index, parsed_database, video_n_frames)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error processing {dcm}: {e}")
    return results

def parse_files(CONFIG, dcm_files_list, database_path, batch_size=100):
    """Optimized parsing with batching and process pooling"""
    print("Parsing DCM Data")
    video_n_frames = CONFIG["VIDEO_SAMPLING"]
    
    print(f'New Dicom Files: {len(dcm_files_list)}')
    
    # Check if there are no new files to process
    if len(dcm_files_list) == 0:
        raise ValueError("No *new* DICOM files found to process.")
    
    # Prepare batches
    batches = []
    for i in range(0, len(dcm_files_list), batch_size):
        batch = []
        for j, dcm in enumerate(dcm_files_list[i:i+batch_size]):
            batch.append((dcm, i+j, database_path, video_n_frames))
        batches.append(batch)
    
    # Process batches in parallel using ProcessPoolExecutor
    data_list = []
    failure_counter = 0
    
    # Use more workers for CPU-bound tasks
    num_workers = min(32, multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                data_list.extend(batch_results)
                # Count failures in batch
                batch_size_actual = len(futures[future])
                failure_counter += batch_size_actual - len(batch_results)
            except Exception as exc:
                print(f'Batch processing exception: {exc}')
                failure_counter += len(futures[future])
    
    if failure_counter > 0:
        print(f'Skipped {failure_counter} DICOMS from missing data or irrelevant data')
    
    append_audit("dicom_parsing.failed_dicoms", failure_counter)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df


def parse_anon_file(anon_location, database_path, image_df, ):
    
    video_df = image_df[image_df['DataType'] == 'video']
    image_df = image_df[image_df['DataType'] == 'image']
    
    # Define common columns
    common_columns = ['Patient_ID', 'Accession_Number', 'RegionSpatialFormat', 'RegionDataType', 
                    'RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 
                    'RegionLocationMaxY1', 'PhotometricInterpretation', 'Rows', 'Columns',
                    'FileName', 'DicomHash', 'SoftwareVersions', 'ManufacturerModelName', 'PhysicalDeltaX']

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
    
    # ADD THIS: Filter to only keep rows where Accession_Number exists in both datasets
    image_accession_numbers = set(image_df['Accession_Number'].unique())
    breast_accession_numbers = set(breast_csv['Accession_Number'].unique())
    
    # Keep only matching accession numbers
    matching_accession_numbers = image_accession_numbers.intersection(breast_accession_numbers)
    
    image_df = image_df[image_df['Accession_Number'].isin(matching_accession_numbers)]
    breast_csv = breast_csv[breast_csv['Accession_Number'].isin(matching_accession_numbers)]
    
    total_breast_accessions = len(breast_accession_numbers)
    non_matching_breast_accessions = len(breast_accession_numbers - matching_accession_numbers)
    percentage_without_images = (non_matching_breast_accessions / total_breast_accessions) * 100 if total_breast_accessions > 0 else 0
    print(f"{percentage_without_images:.1f}% of breast accessions did not have images")
    
    # Populate Has_Malignant and Has_Benign based on final_interpretation
    breast_csv['Has_Malignant'] = breast_csv['final_interpretation'] == 'MALIGNANT'
    breast_csv['Has_Benign'] = breast_csv['final_interpretation'] == 'BENIGN'
    
    image_csv_file = f'{database_path}/ImageData.csv'
    video_csv_file = f'{database_path}/VideoData.csv'
    breast_csv_file = f'{database_path}/BreastData.csv'
    
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
    append_audit("dicom_parsing.images_success", len(image_combined_df))
    append_audit("dicom_parsing.video_success", len(video_df))
    append_audit("dicom_parsing.breast_success", len(breast_csv))
    
    

# Main Method
def Parse_Dicom_Files(CONFIG, anon_location, raw_storage_database, encryption_key):
    database_path = CONFIG["DATABASE_DIR"]
    data_range = CONFIG["DEBUG_DATA_RANGE"]
    
    # Set the global encryption key
    global ENCRYPTION_KEY
    ENCRYPTION_KEY = encryption_key
    
    #Create database dir
    make_dirs(database_path)
    make_dirs(f'{database_path}/images/')
    make_dirs(f'{database_path}/videos/')

    # Get every Dicom File
    dcm_files_list = get_files_by_extension(raw_storage_database, '.dcm')
    append_audit("dicom_parsing.input_dicoms", len(dcm_files_list))
    print(f'Total Dicoms in Input: {len(dcm_files_list)}')

    # Apply data range only if it's specified
    if data_range and len(data_range) == 2:
        dcm_files_list = dcm_files_list[data_range[0]:data_range[1]]
        print(f'Applied Data Range: {len(dcm_files_list)}')
        append_audit("dicom_parsing.data_range", [data_range[0], data_range[1]])
    
    # Get DCM Data
    image_df = parse_files(CONFIG, dcm_files_list, database_path)
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
    append_audit("dicom_parsing.missing_ID_removed", removed_rows)
    
    parse_anon_file(anon_location, database_path, image_df)
    
    