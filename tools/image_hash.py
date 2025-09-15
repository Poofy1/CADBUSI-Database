import os, sys
import pandas as pd
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from storage_adapter import *

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

def create_image_hash_csv(database_path, num_threads=16):
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
    
def process_hash_comparison():
    # File paths
    csv1_path = "D:/DATA/CASBUSI/hash_translation4.csv"
    csv2_path = "C:/Users/Tristan/Desktop/hash_translation.csv"
    output_path = "C:/Users/Tristan/Desktop/hash_translation5.csv"
    
    try:
        # Read the CSV files
        print("Reading CSV files...")
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        print(f"CSV1 loaded: {len(df1)} rows")
        print(f"CSV2 loaded: {len(df2)} rows")
        
        # Create a set of ContentHash values from csv2 for efficient lookup
        csv2_hashes = set(df2['ContentHash'])
        print(f"Unique hashes in CSV2: {len(csv2_hashes)}")
        
        # Add the new column 'Exists_In_New' based on whether ContentHash exists in csv2
        df1['Exists_In_New'] = df1['ContentHash'].isin(csv2_hashes)
        
        # Calculate the percentage of matching rows
        matching_count = df1['Exists_In_New'].sum()
        total_count = len(df1)
        percentage = (matching_count / total_count) * 100
        
        print(f"\nResults:")
        print(f"Total rows in CSV1: {total_count}")
        print(f"Matching rows: {matching_count}")
        print(f"Percentage of CSV1 rows with matching ContentHash in CSV2: {percentage:.2f}%")
        
        # Save the result to a new CSV file
        df1.to_csv(output_path, index=False)
        print(f"\nOutput saved to: {output_path}")
        
        # Display some sample results
        print(f"\nSample of results:")
        print(df1[['ImageName', 'ContentHash', 'Exists_In_New']].head(10))
        
        return percentage, matching_count, total_count
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Please check that the CSV files have the expected column names.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    #process_hash_comparison()
    create_image_hash_csv(database_path=CONFIG["DATABASE_DIR"], num_threads=32)