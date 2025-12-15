import os, sys
import pydicom
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import io
from tools.storage_adapter import read_binary, file_exists, list_files, make_dirs, StorageClient

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.DB_processing.dcm_parser import deidentify_dicom
from src.encrypt_keys import get_encryption_key
import src.DB_processing.dcm_parser as dcm_parser
from config import CONFIG

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')

try:
    import gdcm
    GDCM_AVAILABLE = True
    print("GDCM loaded - enhanced DICOM decompression available")
except ImportError:
    GDCM_AVAILABLE = False
    print("GDCM not available - compressed DICOM handling may be limited")


def determine_dicom_type(ds):
    """
    Determine if DICOM is video or secondary capture
    (Reuses logic from parse_single_dcm)
    
    Returns:
    - Tuple of (is_video, is_secondary)
    """
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
        print(f"Warning: Error determining media type: {e}")
        
    return is_video, is_secondary


def process_dicom_file(input_path, output_path):
    """
    Process a single DICOM file using the existing deidentify_dicom method
    Uses custom storage adapter for reading and writing
    
    Parameters:
    - input_path: Path to input DICOM file
    - output_path: Path to save anonymized DICOM file
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Read DICOM file using storage adapter
        dicom_bytes = read_binary(input_path)
        
        if dicom_bytes is None:
            print(f"Failed to read {input_path}")
            return False
        
        # Parse DICOM from bytes
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        
        # Determine if video or secondary capture
        is_video, is_secondary = determine_dicom_type(ds)
        
        # Anonymize using existing method from dcm_parser
        ds_anon = deidentify_dicom(ds, is_video, is_secondary)
        
        if ds_anon is None:
            return False
        
        # Create output directory if needed
        output_dir = str(Path(output_path).parent)
        make_dirs(output_dir)
        
        # Save anonymized DICOM to bytes buffer
        buffer = io.BytesIO()
        ds_anon.save_as(buffer, write_like_original=False)
        buffer.seek(0)
        
        # Get storage client and save using storage adapter
        storage = StorageClient.get_instance()
        
        if storage.is_gcp:
            # For GCP, upload directly
            blob = storage._bucket.blob(output_path.replace('//', '/').rstrip('/'))
            blob.upload_from_file(buffer, content_type='application/dicom')
        else:
            # For local storage, write to file
            full_path = os.path.join(storage.windir, output_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(buffer.getvalue())
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_dicom_files(input_dir):
    """
    Recursively find all DICOM files in a directory using storage adapter
    
    Parameters:
    - input_dir: Root directory to search
    
    Returns:
    - List of DICOM file paths
    """
    dicom_files = []
    storage = StorageClient.get_instance()
    
    if storage.is_gcp:
        # For GCP, list all blobs with the prefix
        prefix = input_dir.replace('\\', '/').rstrip('/') + '/'
        blobs = storage._bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            # Check for .dcm extension or .DCM
            if blob.name.lower().endswith('.dcm'):
                dicom_files.append(blob.name)
    else:
        # For local storage, walk the directory
        full_path = os.path.join(storage.windir, input_dir) if storage.windir else input_dir
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    # Get relative path from input_dir
                    file_path = os.path.join(root, file)
                    if storage.windir:
                        # Make path relative to windir
                        rel_path = os.path.relpath(file_path, storage.windir)
                        dicom_files.append(rel_path)
                    else:
                        dicom_files.append(file_path)
    
    return dicom_files


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize DICOM files while preserving directory structure'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing DICOM files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for anonymized DICOM files'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of DICOM files to process (default: process all)'
    )
    parser.add_argument(
        '--preserve-structure',
        action='store_true',
        default=True,
        help='Preserve input directory structure in output (default: True)'
    )
    
    args = parser.parse_args()
    
    # Initialize storage client
    storage = StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    input_dir = args.input
    output_dir = args.output
    
    # Get encryption key using the existing key management system
    encryption_key = get_encryption_key()
    dcm_parser.ENCRYPTION_KEY = encryption_key
    print("Using encryption key from key management system")
    
    # Find all DICOM files
    print(f"\nScanning for DICOM files in {input_dir}...")
    dicom_files = find_dicom_files(input_dir)
    
    if not dicom_files:
        print("No DICOM files found!")
        return
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        dicom_files = dicom_files[:args.limit]
        print(f"Limited to processing {len(dicom_files)} files")
    
    print(f"Found {len(dicom_files)} DICOM files to process")
    print(f"Anonymization settings:")
    print(f"  - ID encryption: Enabled")
    print(f"  - Preserve structure: {'Yes' if args.preserve_structure else 'No'}")
    print(f"  - Output directory: {output_dir}")
    if args.limit:
        print(f"  - Processing limit: {args.limit} files")
    print()
    
    # Process each file
    success_count = 0
    failure_count = 0
    
    for input_file in tqdm(dicom_files, desc="Anonymizing DICOMs"):
        # Determine output path
        if args.preserve_structure:
            # Preserve directory structure
            input_path = Path(input_file)
            input_base = Path(input_dir)
            
            # Calculate relative path
            try:
                rel_path = input_path.relative_to(input_base)
            except ValueError:
                # If relative_to fails, just use the filename
                rel_path = input_path.name
            
            output_file = str(Path(output_dir) / rel_path)
        else:
            # Flat structure in output
            output_file = str(Path(output_dir) / Path(input_file).name)
        
        # Process file
        if process_dicom_file(input_file, output_file):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    print()
    print("="*60)
    print("Anonymization Complete")
    print("="*60)
    print(f"Successfully anonymized: {success_count} files")
    print(f"Failed: {failure_count} files")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()