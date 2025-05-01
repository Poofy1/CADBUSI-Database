import os
import zipfile
import tempfile
import argparse
from google.cloud import storage
from tqdm import tqdm
# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def download_and_zip_dicom_files(target_path):
    # Get configuration values
    project_id = CONFIG['env']['project_id']
    bucket_name = CONFIG['storage']['bucket_name']
    
    # Initialize GCS client
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Create a temporary directory to store downloaded files
    temp_dir = tempfile.mkdtemp()
    
    # List all blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=target_path + '/'))
    
    # Keep track of downloaded files
    downloaded_files = []
    
    # Download each blob while preserving structure
    for blob in tqdm(blobs, desc="Downloading files", unit="file"):
        # Skip the directory itself
        if blob.name.endswith('/'):
            continue
            
        # Create the local file path with the same structure
        relative_path = blob.name[len(target_path) + 1:]  # +1 for the trailing slash
        local_file_path = os.path.join(temp_dir, relative_path)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the blob
        blob.download_to_filename(local_file_path)
        downloaded_files.append(local_file_path)
    
    if not downloaded_files:
        print(f"No files found at path: {target_path}")
        return None
    
    # Create a zip file in the same directory as the script
    timestamp = os.path.basename(os.path.normpath(target_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_filename = os.path.join(script_dir, f"dicom_files_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(downloaded_files, desc="Creating zip", unit="file"):
            # Add file to zip with relative path
            arcname = os.path.relpath(file, temp_dir)
            zipf.write(file, arcname=arcname)
    
    # Cleanup temporary directory
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(temp_dir)
    
    return zip_filename

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Download and zip DICOM files from Google Cloud Storage')
    parser.add_argument('target_path', type=str, help='Path to the target directory in the bucket')
    parser.add_argument('--output', '-o', type=str, help='Custom output path for the zip file (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Download and zip files
    zip_path = download_and_zip_dicom_files(args.target_path)
    
    # If custom output path is provided, move the zip file
    if zip_path and args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        new_path = args.output if args.output.endswith('.zip') else args.output + '.zip'
        os.rename(zip_path, new_path)
        zip_path = new_path
    
    if zip_path:
        print(f"Successfully created zip file at: {zip_path}")
    else:
        print("Failed to create zip file.")
        
        
# EXAMPLE: python script.py path/to/target/to/download/and/stuff -o /custom/output/path.zip