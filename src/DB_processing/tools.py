
import easyocr
import threading
thread_local = threading.local()
import os
from storage_adapter import *
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir) # 2 dirs back

def append_audit(key, value, new_file=False):
    """
    Add or update a key-value pair in a JSON data file.
    
    Args:
        key (str): The key for the data, can use dot notation for nested objects (e.g., "general.num_patients")
        value: The value to store
        new_file (bool): If True, create a new JSON file instead of updating
    """
    # Create the fixed path for the JSON file
    json_file_path = os.path.join(parent_dir, "raw_data", "audit.json")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
    # Initialize data dictionary
    data = {}
    
    # If not creating a new file and the file exists, read existing data
    if not new_file and os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but isn't valid JSON, initialize empty
            data = {}
    
    # Convert NumPy data types to Python native types
    value = convert_numpy_types(value)
    
    # Handle nested keys with dot notation (e.g., "general.num_patients")
    keys = key.split('.')
    current = data
    
    # Navigate to the deepest level of the nested structure
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the value at the deepest level
    current[keys[-1]] = value
    
    # Write the updated data back to the file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return data

def convert_numpy_types(obj):
    """
    Convert NumPy data types and other non-JSON serializable types to Python native types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


# configure easyocr reader
reader = easyocr.Reader(['en'])
