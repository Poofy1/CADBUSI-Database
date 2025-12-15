from google.cloud import storage
import cv2
import numpy as np
import io
import pandas as pd
import os
from PIL import Image
from threading import Lock
from io import BytesIO
import time
from google.api_core import exceptions

class StorageClient:
    _instance = None
    _lock = Lock()
    
    def __init__(self, windir=None, bucket_name=None):
        # Consider empty string bucket_name as None
        self.is_gcp = bucket_name is not None and bucket_name != ""
        self.windir = windir if windir else ""
        # This will only print during actual initialization
        if not hasattr(StorageClient, '_initialized'):
            print(f"Storage mode: {'GCP' if self.is_gcp else 'Windows'}")
            StorageClient._initialized = True
        if self.is_gcp:
            self._client = storage.Client()
            self._bucket = self._client.bucket(bucket_name)
            
    @classmethod
    def get_instance(cls, windir=None, bucket_name=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(windir, bucket_name)
        return cls._instance

def read_image(path, use_pil=False, max_retries=3, local_override = False):
    storage = StorageClient.get_instance()
    
    try:
        if storage.is_gcp and not local_override:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            
            for attempt in range(max_retries):
                try:
                    bytes_data = blob.download_as_bytes()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to download from GCP after {max_retries} attempts: {str(e)}")
                        return None
                    print(f"Download attempt {attempt + 1} failed, retrying...")
                    time.sleep(1 * (attempt + 1))
            
            if use_pil:
                return Image.open(BytesIO(bytes_data))
            else:
                nparr = np.frombuffer(bytes_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            full_path = os.path.join(storage.windir, path)
            if use_pil:
                return Image.open(full_path)
            else:
                return cv2.imread(full_path)
                
    except Exception as e:
        print(f"Error reading image {path}: {str(e)}")
        return None

def read_txt(path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        try:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            content = blob.download_as_string()
            # Decode bytes to string if content is in bytes
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
        except exceptions.NotFound:
            print(f"File not found: {path}")
            return None
    else:
        try:
            full_path = os.path.join(storage.windir, path)
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except Exception as e:
            print(f"Error reading text file {path}: {str(e)}")
            return None


def read_csv(path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        try:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            content = blob.download_as_string()
            return pd.read_csv(io.BytesIO(content))
        except exceptions.NotFound:
            print(f"File not found: {path}")
            raise None
    else:
        full_path = os.path.join(storage.windir, path)
        return pd.read_csv(full_path)


def get_files_by_extension(directory, extension, local_override = False):
    storage = StorageClient.get_instance()
    file_paths = []

    if storage.is_gcp and not local_override:
        # List all blobs in the directory and filter by extension
        prefix = directory.replace('\\', '/').rstrip('/') + '/'
        blobs = storage._bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            if blob.name.endswith(extension):
                file_paths.append(blob.name)
    else:
        # Use os.walk for local filesystem
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))

    return file_paths

def save_data(data, path, local_override=False, max_retries=5):
    storage = StorageClient.get_instance()

    # Convert PIL Image to numpy array if needed
    if isinstance(data, Image.Image):
        data = np.array(data)
        if len(data.shape) == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    if storage.is_gcp and not local_override:
        blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))

        # Retry logic for GCP uploads
        for attempt in range(max_retries):
            try:
                if isinstance(data, np.ndarray):
                    _, encoded = cv2.imencode('.png', data)
                    blob.upload_from_string(encoded.tobytes())
                elif isinstance(data, (pd.DataFrame, pd.Series)):
                    buffer = io.StringIO()
                    data.to_csv(buffer, index=False)
                    blob.upload_from_string(buffer.getvalue())
                else:
                    blob.upload_from_string(str(data))
                return True  # Success

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to upload {path} after {max_retries} attempts: {e}")
                    return False

                wait_time = 2 ** attempt  # Simple exponential backoff: 1s, 2s, 4s, 8s, 16s
                time.sleep(wait_time)
    else:
        # Local storage path (no retries needed)
        try:
            full_path = os.path.join(storage.windir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if isinstance(data, np.ndarray):
                success = cv2.imwrite(full_path, data)
                if not success:
                    print(f"Failed to write image to {path}")
                    return False
                return True
            elif isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_csv(full_path, index=False)
                return True
            else:
                with open(full_path, 'w') as f:
                    f.write(str(data))
                return True
        except Exception as e:
            print(f"Failed to save data to {path}: {e}")
            return False

def read_binary(path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        try:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            return blob.download_as_bytes()
        except exceptions.NotFound:
            print(f"File not found: {path}")
            return None
        except Exception as e:
            print(f"Error reading binary file {path}: {str(e)}")
            return None
    else:
        try:
            full_path = os.path.join(storage.windir, path)
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading binary file {path}: {str(e)}")
            return None

def rename_file(old_path, new_path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        try:
            source_blob = storage._bucket.blob(old_path.replace('//', '/').rstrip('/'))
            dest_blob = storage._bucket.blob(new_path.replace('//', '/').rstrip('/'))
            
            # Copy source to destination
            storage._bucket.copy_blob(source_blob, storage._bucket, dest_blob.name)
            # Delete the source
            source_blob.delete()
        except exceptions.NotFound:
            print(f"Source file not found: {old_path}")
        except Exception as e:
            print(f"Error renaming file from {old_path} to {new_path}: {str(e)}")
    else:
        try:
            old_full_path = os.path.join(storage.windir, old_path)
            new_full_path = os.path.join(storage.windir, new_path)
            os.rename(old_full_path, new_full_path)
        except Exception as e:
            print(f"Error renaming file from {old_path} to {new_path}: {str(e)}")
                  
def file_exists(path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
        return blob.exists()
    else:
        full_path = os.path.join(storage.windir, path)
        return os.path.exists(full_path)
    
def delete_file(path, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        try:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            blob.delete()
        except exceptions.NotFound:
            print(f"File not found to delete: {path}")
        except Exception as e:
            print(f"Error deleting file {path}: {str(e)}")
    else:
        try:
            full_path = os.path.join(storage.windir, path)
            if os.path.exists(full_path):
                os.remove(full_path)
        except Exception as e:
            print(f"Error deleting file {path}: {str(e)}")

def list_files(folder_path, extension=None, local_override = False):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp and not local_override:
        # Normalize path and ensure it ends with /
        prefix = folder_path.replace('//', '/').rstrip('/') + '/'
        blobs = storage._bucket.list_blobs(prefix=prefix)
        
        if extension:
            return [blob.name for blob in blobs if blob.name.endswith(extension)]
        return [blob.name for blob in blobs]
    else:
        full_path = os.path.join(storage.windir, folder_path)
        files = os.listdir(full_path)
        
        if extension:
            return [os.path.join(full_path, f) for f in files if f.endswith(extension)]
        return [os.path.join(full_path, f) for f in files]
    
def make_dirs(path, local_override = False):
    storage = StorageClient.get_instance()
    
    # Only create directories for local storage
    if not storage.is_gcp or local_override:
        full_path = os.path.join(storage.windir, path)
        os.makedirs(full_path, exist_ok=True)
