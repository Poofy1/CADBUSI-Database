import os
import zipfile
import tempfile
from google.cloud import storage
from tqdm import tqdm
# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def download_and_zip_dicom_files():
    # Get configuration values
    project_id = CONFIG['env']['project_id']
    bucket_name = CONFIG['storage']['bucket_name']
    anonymized_path = CONFIG['storage']['anonymized_path']
    
    # Initialize GCS client
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Create a temporary directory to store downloaded files
    temp_dir = tempfile.mkdtemp()
    
    # List all blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=anonymized_path + '/'))
    
    # Keep track of downloaded files
    downloaded_files = []
    
    # Download each blob while preserving structure
    for blob in tqdm(blobs, desc="Downloading files", unit="file"):
        # Skip the directory itself
        if blob.name.endswith('/'):
            continue
            
        # Create the local file path with the same structure
        relative_path = blob.name[len(anonymized_path) + 1:]  # +1 for the trailing slash
        local_file_path = os.path.join(temp_dir, relative_path)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the blob
        blob.download_to_filename(local_file_path)
        downloaded_files.append(local_file_path)
    
    if not downloaded_files:
        return None
    
    # Create a zip file in the same directory as the script
    timestamp = os.path.basename(os.path.normpath(anonymized_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_filename = os.path.join(script_dir, f"dicom_files_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(downloaded_files, desc="Creating zip", unit="file"):
            # Add file to zip with relative path
            arcname = os.path.relpath(file, temp_dir)
            zipf.write(file, arcname=arcname)
    
    return zip_filename

if __name__ == "__main__":
    zip_path = download_and_zip_dicom_files()
    if zip_path:
        print(f"Successfully created zip file at: {zip_path}")
    else:
        print("Failed to create zip file.")