import pydicom
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
from src.encrypt_keys import *
from google.cloud import storage
from io import BytesIO
import time
from tools.audit import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory


# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

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

def deidentify_dicom(ds):
    ds.remove_private_tags()  # take out private tags added by notion or otherwise
    
    # Avoid separate walks by combining them
    ds.walk(anon_callback)
    # Only walk file_meta if it exists
    if hasattr(ds, 'file_meta') and ds.file_meta is not None:
        ds.file_meta.walk(anon_callback)

    media_type = ds.file_meta[0x00020002]
    is_video = 'Multi-frame' in str(media_type)
    is_secondary = 'Secondary' in str(media_type)
    
    y0 = 101
    
    if not is_secondary and (0x0018, 0x6011) in ds:
            y0 = ds['SequenceOfUltrasoundRegions'][0]['RegionLocationMinY0'].value

    if 'OriginalAttributesSequence' in ds:
        del ds.OriginalAttributesSequence
        
    # Check if Pixel Data is compressed
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        # Attempt to decompress the Pixel Data
        try:
            ds.decompress()
        except NotImplementedError as e:
            print(f"Decompression not implemented for this transfer syntax: {e}")
            return None  # or handle this appropriately for your use case
        except Exception as e:
            print(f"An error occurred during decompression: {e}")
            return None  # or handle this appropriately for your use case

    # crop patient info above US region 
    arr = ds.pixel_array
    
    if is_video:
        arr[:,:y0] = 0
    else:
        arr[:y0] = 0
    
    # Update the Pixel Data
    ds.PixelData = arr.tobytes()
    
    # Important: Keep the original transfer syntax - DO NOT MODIFY THIS LINE
    ds.file_meta.TransferSyntaxUID = ds.file_meta.TransferSyntaxUID

    return ds



def create_dcm_filename(ds, key):
        
    # Extract the necessary identifiers
    accession_number = ds.AccessionNumber
    patient_id = ds.PatientID
    
    # Encrypt identifiers using the new method
    anonymized_patient_id = encrypt_single_id(key, patient_id)
    anonymized_accession_number = encrypt_single_id(key, accession_number)
    
    # Check the media type
    media_type = ds.file_meta[0x00020002]
    is_video = str(media_type).find('Multi-frame') > -1
    is_secondary = str(media_type).find('Secondary') > -1

    if is_video:
        media = 'video'
    elif is_secondary:
        media = 'second'
    else:
        media = 'image'
    
    # Create a hash object
    hash_obj = hashlib.sha256()
    hash_obj.update(ds.pixel_array.tobytes())  # Convert pixel_array to bytes before hashing
    
    image_hash = hash_obj.hexdigest()
    
    # Try to convert encrypted IDs to integers and pad to 8 digits
    try:
        anon_patient_id_int = int(anonymized_patient_id)
        formatted_patient_id = f"{anon_patient_id_int:08}"
    except ValueError:
        formatted_patient_id = anonymized_patient_id
        
    try:
        anon_accession_number_int = int(anonymized_accession_number)
        formatted_accession_number = f"{anon_accession_number_int:08}"
    except ValueError:
        formatted_accession_number = anonymized_accession_number
    
    # Construct the filename using the anonymized identifiers
    filename = f'{media}_{formatted_patient_id}_{formatted_accession_number}_{image_hash}.dcm'

    # Anonymize the DICOM data - set the new IDs
    ds.PatientID = anonymized_patient_id
    ds.AccessionNumber = anonymized_accession_number

    return filename, ds  # return the modified DICOM dataset along with the filename

def process_single_blob(blob, client, output_bucket_name, output_bucket_path, encryption_key, max_retries=3, error_counters=None):
    """Process a single DICOM blob from GCP bucket using RAM with download retry logic"""
    # Initialize error counters if not provided
    if error_counters is None:
        error_counters = {"metadata_errors": 0, "pixel_data_errors": 0, "decompression_errors": 0, "other_errors": 0}
    
    # First try to download the blob with retries
    dataset = None
    
    for attempt in range(max_retries):
        try:
            bytes_data = blob.download_as_bytes()
            dataset = pydicom.dcmread(BytesIO(bytes_data), force=True)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download from GCP after {max_retries} attempts: {str(e)}")
                error_counters["other_errors"] += 1
                return None, error_counters
            print(f"Download attempt {attempt + 1} failed, retrying...")
            time.sleep(1 * (attempt + 1))
            
    # If we have a valid dataset, proceed with processing
    try:
        # Create a new filename using encryption
        try:
            new_filename, dataset = create_dcm_filename(dataset, encryption_key)
        except KeyError as e:
            print(f"Metadata tag error in {blob.name}: Missing tag {e}")
            error_counters["metadata_errors"] += 1
            return None, error_counters
        except Exception as e:
            print(f"Filename creation error in {blob.name}: {str(e)}")
            error_counters["other_errors"] += 1
            return None, error_counters
            
        # Check for pixel data before deidentification
        if not hasattr(dataset, 'pixel_array'):
            print(f"No pixel data found in {blob.name}")
            error_counters["pixel_data_errors"] += 1
            return None, error_counters
        
        # De-identify the DICOM dataset
        try:
            dataset = deidentify_dicom(dataset)
            if dataset is None:
                print(f"Deidentification failed for {blob.name}")
                error_counters["other_errors"] += 1
                return None, error_counters
        except NotImplementedError as e:
            print(f"Decompression not supported for {blob.name}: Missing required libraries")
            error_counters["decompression_errors"] += 1
            return None, error_counters
        except Exception as e:
            if "compression" in str(e).lower():
                print(f"Decompression error in {blob.name}: Missing required libraries")
                error_counters["decompression_errors"] += 1
            else:
                print(f"Deidentification error in {blob.name}: {str(e)}")
                error_counters["other_errors"] += 1
            return None, error_counters
        
        # Create folder structure based on PatientID_AccessionNumber
        folder_name = f"{dataset.PatientID}_{dataset.AccessionNumber}"
        
        # Set the target path in GCP - now including study_id
        output_blob_path = os.path.join(output_bucket_path, folder_name, new_filename)
        
        # Save the deidentified DICOM directly to memory
        output_buffer = BytesIO()
        dataset.save_as(output_buffer)
        output_buffer.seek(0)  # Reset buffer position to beginning
        
        # Upload the deidentified DICOM back to GCP directly from memory
        output_bucket = client.bucket(output_bucket_name)
        output_blob = output_bucket.blob(output_blob_path)
        output_blob.upload_from_file(output_buffer)
        
        return blob.name, error_counters
        
    except Exception as e:
        error_msg = str(e)
        if "(0002,0002)" in error_msg:
            print(f"Metadata tag error in {blob.name}: Issue with Media Storage SOP Class UID")
            error_counters["metadata_errors"] += 1
        elif "no pixel data" in error_msg.lower():
            print(f"No pixel data found in {blob.name}")
            error_counters["pixel_data_errors"] += 1
        elif "decompress" in error_msg.lower() and "missing dependencies" in error_msg.lower():
            print(f"Decompression error in {blob.name}: Missing required libraries")
            error_counters["decompression_errors"] += 1
        else:
            print(f"Error processing {blob.name}: {error_msg[:100]}")  # Limit error message length
            error_counters["other_errors"] += 1
        return None, error_counters

def process_batch(blob_batch, client, output_bucket_name, output_bucket_path, encryption_key, error_counters):
    """Process a batch of DICOM blobs"""
    successful = 0
    failed = 0
    
    # Use ThreadPoolExecutor with a limited number of workers
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        # Submit tasks to the executor
        futures = {
            executor.submit(
                process_single_blob, 
                blob, 
                client, 
                output_bucket_name, 
                output_bucket_path, 
                encryption_key,
                3,  # max_retries
                error_counters
            ): blob for blob in blob_batch
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                result, updated_counters = future.result()
                # Merge the error counters from this thread
                for key in error_counters:
                    error_counters[key] = updated_counters[key]
                
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f'An exception occurred: {exc}')
                error_counters["other_errors"] += 1
                failed += 1
                
    return successful, failed, error_counters


def deidentify_bucket_dicoms(bucket_path, output_bucket_path, encryption_key, batch_size=100):
    """Process DICOM files from a GCP bucket and upload deidentified versions to output bucket"""
    # Initialize storage client
    client = storage.Client()
    
    # Get the bucket
    bucket = client.bucket(CONFIG["storage"]["bucket_name"])
    
    # Initialize error counters
    error_counters = {
        "metadata_errors": 0,
        "pixel_data_errors": 0,
        "decompression_errors": 0,
        "other_errors": 0
    }
    
    # Count DICOM files first to calculate number of batches
    dicom_files = [blob for blob in bucket.list_blobs(prefix=bucket_path) 
                  if blob.name.lower().endswith('.dcm')]
    total_files = len(dicom_files)
    total_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    append_audit(os.path.join(env, "raw_data"), f"Found {total_files} DICOMs")
    
    total_processed = 0
    successful = 0
    failed = 0
    
    # Process in batches
    print(f"Starting batch processing of {total_batches} DICOM batches...")
    
    # Create a single progress bar for all batches
    with tqdm(total=total_batches, desc="Processing DICOM batches") as pbar:
        # Process files in batches
        for i in range(0, len(dicom_files), batch_size):
            current_batch = dicom_files[i:i+batch_size]
            
            success, fail, error_counters = process_batch(
                current_batch, client, CONFIG["storage"]["bucket_name"], 
                output_bucket_path, encryption_key, error_counters
            )
            successful += success
            failed += fail
            total_processed += len(current_batch)
            
            # Update progress bar by 1 batch
            pbar.update(1)
    
    # Print detailed error counts
    append_audit(os.path.join(env, "raw_data"), f"{error_counters['metadata_errors']} DICOMs Failed - Issues with DICOM metadata tags")
    append_audit(os.path.join(env, "raw_data"), f"{error_counters['pixel_data_errors']} DICOMs Failed - Missing pixel data in the DICOM file")
    append_audit(os.path.join(env, "raw_data"), f"{error_counters['decompression_errors']} DICOMs Failed - Decompression errors")
    append_audit(os.path.join(env, "raw_data"), f"{error_counters['other_errors']} DICOMs Failed - Other errors")
    append_audit(os.path.join(env, "raw_data"), f"Remaining DICOMs: {successful}")
    
    print(f"Processing complete. Total: {total_processed}, Success: {successful}, Failed: {failed}")
    print(f"Error breakdown:")
    print(f"- Metadata errors: {error_counters['metadata_errors']}")
    print(f"- Pixel data errors: {error_counters['pixel_data_errors']}")
    print(f"- Decompression errors: {error_counters['decompression_errors']}")
    print(f"- Other errors: {error_counters['other_errors']}")
    
    return successful, failed