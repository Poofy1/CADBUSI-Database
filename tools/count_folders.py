from google.cloud import storage
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import CONFIG


def count_immediate_folders(bucket_name, folder_prefix):
    """
    Count the number of immediate subdirectories in a GCP bucket folder.
    
    Args:
        bucket_name (str): Name of the GCP bucket
        folder_prefix (str): Folder path prefix to search in
    
    Returns:
        int: Number of immediate subdirectories
    """
    # Initialize the storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Ensure folder_prefix ends with '/'
    if not folder_prefix.endswith('/'):
        folder_prefix += '/'
    
    # Get all objects with the prefix
    blobs = bucket.list_blobs(prefix=folder_prefix, delimiter='/')
    
    # Count immediate folders (prefixes returned by delimiter)
    immediate_folders = set()
    
    # Consume the iterator to get prefixes
    list(blobs)  # This processes the blobs and populates prefixes
    
    # Count the prefixes (immediate subdirectories)
    if hasattr(blobs, 'prefixes'):
        immediate_folders = set(blobs.prefixes)
    
    return len(immediate_folders)

def main():
    bucket_name = CONFIG['storage']['bucket_name']
    folder_prefix = "Downloads/2025-05-30_021926/"
    
    try:
        folder_count = count_immediate_folders(bucket_name, folder_prefix)
        print(f"Number of immediate folders in '{folder_prefix}': {folder_count}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()