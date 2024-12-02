from google.cloud import storage
import cv2
import numpy as np
import io
import pandas as pd
import os
import json
from threading import Lock

class StorageClient:
    _instance = None
    _lock = Lock()
    
    def __init__(self, windir=None, bucket_name=None):
        self.is_gcp = bucket_name is not None
        self.windir = windir if windir else ""
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

def read_image(path):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp:
        try:
            blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
            nparr = np.frombuffer(blob.download_as_bytes(), np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading image {path}: {str(e)}")
            return None
    else:
        full_path = os.path.join(storage.windir, path)
        return cv2.imread(full_path)

def read_csv(path):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp:
        blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
        content = blob.download_as_string()
        return pd.read_csv(io.BytesIO(content))
    else:
        full_path = os.path.join(storage.windir, path)
        return pd.read_csv(full_path)

def save_data(data, path):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp:
        blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
        
        if isinstance(data, np.ndarray):
            _, encoded = cv2.imencode('.png', data)
            blob.upload_from_string(encoded.tobytes())
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            blob.upload_from_string(buffer.getvalue())
        else:
            blob.upload_from_string(str(data))
    else:
        full_path = os.path.join(storage.windir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if isinstance(data, np.ndarray):
            cv2.imwrite(full_path, data)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_csv(full_path, index=False)
        else:
            with open(full_path, 'w') as f:
                f.write(str(data))

def file_exists(path):
    storage = StorageClient.get_instance()
    
    if storage.is_gcp:
        blob = storage._bucket.blob(path.replace('//', '/').rstrip('/'))
        return blob.exists()
    else:
        full_path = os.path.join(storage.windir, path)
        return os.path.exists(full_path)
    
    
    
def make_dirs(path):
    storage = StorageClient.get_instance()
    
    # Only create directories for local storage
    if not storage.is_gcp:
        full_path = os.path.join(storage.windir, path)
        os.makedirs(full_path, exist_ok=True)
