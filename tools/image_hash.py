import os, sys
import pandas as pd
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tools.storage_adapter import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import CONFIG
env = os.path.dirname(os.path.abspath(__file__))

def calculate_md5_from_image(image):
    """Calculate MD5 hash from image data"""
    if image is None:
        return None
    
    try:
        # Convert image to bytes for hashing
        image_bytes = image.tobytes()
        
        hash_md5 = hashlib.md5()
        # Process in chunks if bytes are large
        chunk_size = 8192
        for i in range(0, len(image_bytes), chunk_size):
            chunk = image_bytes[i:i + chunk_size]
            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None

def process_image(args):
    """Process a single image - for multiprocessing"""
    image_path, image_name, dicom_hash = args
    
    # Use the existing read_image function
    image = read_image(image_path, use_pil=True)
    content_hash = calculate_md5_from_image(image)
    
    return {
        'ImageName': image_name,
        'DicomHash': dicom_hash,
        'ContentHash': content_hash
    }

def create_image_hash_csv(database_path, num_threads=16, limit=None):
    """Create CSV with ImageName, DicomHash, and ContentHash"""
    
    images_dir = os.path.join(database_path, 'images')
    image_data_csv = os.path.join(database_path, 'ImageData.csv')
    
    # Read ImageData.csv
    print("Reading ImageData.csv...")
    image_df = read_csv(image_data_csv)
    image_to_dicom_hash = dict(zip(image_df['ImageName'], image_df['DicomHash']))
    
    # Get all PNG files from storage
    print("Scanning for PNG files...")
    storage = StorageClient.get_instance()
    
    png_files = []
    image_paths = []
    
    if storage.is_gcp:
        # List files from GCP bucket - iterate only once
        blobs = storage._bucket.list_blobs(prefix=images_dir)
        for blob in blobs:
            if blob.name.endswith('.png'):
                png_files.append(os.path.basename(blob.name))
                image_paths.append(blob.name)
    else:
        # List files from local directory
        local_images_dir = os.path.join(storage.windir, images_dir)
        png_files = [f for f in os.listdir(local_images_dir) if f.endswith('.png')]
        image_paths = [os.path.join(images_dir, f) for f in png_files]
    
    print(f"Found {len(png_files)} PNG files")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        png_files = png_files[:limit]
        image_paths = image_paths[:limit]
        print(f"Limited to processing first {len(png_files)} files")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for i, image_name in enumerate(png_files):
        image_path = image_paths[i]
        dicom_hash = image_to_dicom_hash.get(image_name, 'NOT_FOUND')
        tasks.append((image_path, image_name, dicom_hash))
    
    # Process images with multithreading
    results = []
    print(f"Processing {len(tasks)} images with {num_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_image, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result['ContentHash'] is not None:
                results.append(result)
    
    # Create and save CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join(env, 'ImageContentHashes.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Processed {len(results)} images successfully")
    print(f"Output saved to: {output_file}")
    
def link_hashes():
    # File paths
    new_hash_path = "C:/Users/Tristan/Desktop/newHashes.csv"  # hashed new database
    old_hash_path = "C:/Users/Tristan/Desktop/oldHashes.csv"  # hashed old database
    old_image_path = "D:/DATA/CASBUSI/database/ImageData.csv"  # old database
    old_label_path = "D:/DATA/CASBUSI/labelbox_data/InstanceLabels.csv"  # old labels
    
    print("Loading data files...")
    
    # Load all CSV files
    try:
        new_hash_df = pd.read_csv(new_hash_path)
        old_hash_df = pd.read_csv(old_hash_path)
        old_image_df = pd.read_csv(old_image_path)
        old_label_df = pd.read_csv(old_label_path)
        
        print(f"Loaded {len(new_hash_df)} rows from newHashes.csv")
        print(f"Loaded {len(old_hash_df)} rows from oldHashes.csv")
        print(f"Loaded {len(old_image_df)} rows from ImageData.csv")
        print(f"Loaded {len(old_label_df)} rows from InstanceLabels.csv")
        
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    # Step 1: Merge old_image_path with old_label_path using FileName
    print("\nStep 1: Merging ImageData with InstanceLabels on FileName...")
    
    output1 = pd.merge(
        old_label_df,  # Keep all columns from labels
        old_image_df[['FileName', 'DicomHash']],  # Only take FileName and DicomHash from images
        on='FileName',
        how='inner'  # Only keep matching rows
    )
    
    print(f"Output1: {len(output1)} rows after merging on FileName")
    
    # Save output1
    output1_path = "C:/Users/Tristan/Desktop/output1_merged_labels_images.csv"
    output1.to_csv(output1_path, index=False)
    print(f"Saved output1 to: {output1_path}")
    
    # Step 2: Compare new_hash_path with old_hash_path using ContentHash
    print("\nStep 2: Merging hash files on ContentHash...")
    
    output2 = pd.merge(
        old_hash_df,  # Keep all old hash data
        new_hash_df[['ContentHash', 'DicomHash']].rename(columns={'DicomHash': 'RealDicomHash'}),  # Add DicomHash as RealDicomHash
        on='ContentHash',
        how='inner'  # Only keep matching rows
    )
    
    print(f"Output2: {len(output2)} rows after merging on ContentHash")
    
    # Save output2
    output2_path = "C:/Users/Tristan/Desktop/output2_merged_hashes.csv"
    output2.to_csv(output2_path, index=False)
    print(f"Saved output2 to: {output2_path}")
    
    # Step 3: Merge output1 and output2
    print("\nStep 3: Final merge of output1 and output2...")
    
    # Need to determine the merge key between output1 and output2
    # Assuming we merge on DicomHash from output1 with DicomHash from output2
    final_output = pd.merge(
        output1,  # Keep all output1 data
        output2[['DicomHash', 'RealDicomHash']],  # Only take DicomHash and RealDicomHash from output2
        on='DicomHash',
        how='left'  # Keep all output1 rows, add RealDicomHash where available
    )
    
    print(f"Final output: {len(final_output)} rows")
    print(f"Rows with RealDicomHash: {final_output['RealDicomHash'].notna().sum()}")
    
    # Save final output
    final_output_path = "C:/Users/Tristan/Desktop/final_linked_data.csv"
    final_output.to_csv(final_output_path, index=False)
    print(f"Saved final output to: {final_output_path}")
    
    # Step 4: Create filtered final output with specific columns
    print("\nStep 4: Creating filtered final output...")
    
    # Select only the specified columns
    required_columns = [
        'Reject Image',
        'Only Normal Tissue',
        'Cyst Lesion Present',
        'Benign Lesion Present',
        'Malignant Lesion Present',
        'RealDicomHash'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in final_output.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {list(final_output.columns)}")
        # Use only the columns that exist
        available_columns = [col for col in required_columns if col in final_output.columns]
        filtered_output = final_output[available_columns].copy()
    else:
        filtered_output = final_output[required_columns].copy()
    
    # Rename RealDicomHash to DicomHash
    if 'RealDicomHash' in filtered_output.columns:
        filtered_output = filtered_output.rename(columns={'RealDicomHash': 'DicomHash'})
    
    print(f"Filtered output: {len(filtered_output)} rows")
    print(f"Final columns: {list(filtered_output.columns)}")
    
    # Save filtered final output
    filtered_output_path = "C:/Users/Tristan/Desktop/final_filtered_data.csv"
    filtered_output.to_csv(filtered_output_path, index=False)
    print(f"Saved filtered final output to: {filtered_output_path}")
    
    # Display summary information
    print("\n=== SUMMARY ===")
    print(f"Step 1 - Labels + Images: {len(output1)} rows")
    print(f"Step 2 - Hash matching: {len(output2)} rows")
    print(f"Step 3 - Final merge: {len(final_output)} rows")
    print(f"Step 4 - Filtered output: {len(filtered_output)} rows")
    print(f"Filtered columns: {list(filtered_output.columns)}")
    
    return filtered_output


if __name__ == "__main__":
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    link_hashes()
    #create_image_hash_csv(database_path=CONFIG["DATABASE_DIR"], num_threads=32)