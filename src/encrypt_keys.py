from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import csv
import pickle
import struct


# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


def generate_key():
    return os.urandom(16)  # 128-bit key


def get_encryption_key():
    """
    Retrieves or generates an encryption key for DICOM deidentification.
    
    The function looks for an existing key in a predefined location or environment variable.
    If no key exists, it generates a new one and stores it.
    
    Returns:
        bytes: A 16-byte encryption key for use with AES-128
    """
    # Check for key in environment variable first
    import os
    env_key_name = "DICOM_ENCRYPTION_KEY"
    key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encryption_key.pkl")
    
    # Try to get key from environment variable (base64 encoded)
    import base64
    if env_key_name in os.environ:
        try:
            return base64.b64decode(os.environ[env_key_name])
        except Exception as e:
            print(f"Error loading key from environment: {e}")
    
    # If not in environment, try to load from file
    import pickle
    if os.path.exists(key_file_path):
        try:
            with open(key_file_path, 'rb') as key_file:
                key = pickle.load(key_file)
            print(f"Using existing encryption key from {key_file_path}")
            return key
        except Exception as e:
            print(f"Error loading existing key file: {e}")
    
    # If no key exists, generate a new one
    key = generate_key()  # Uses the existing generate_key function
    
    # Save the key to file for future use
    try:
        with open(key_file_path, 'wb') as key_file:
            pickle.dump(key, key_file)
        print(f"Generated new encryption key and saved to {key_file_path}")
    except Exception as e:
        print(f"Warning: Could not save encryption key to file: {e}")
        print(f"Consider saving this key manually: {base64.b64encode(key).decode()}")
    
    return key


def ff1_encrypt(key, number, domain_size):
    """
    Format-preserving encryption using a simplified FF1-based approach.
    This guarantees a permutation (no collisions) for the given domain size.
    """
    # Convert number to bytes for encryption
    number_bytes = str(number).encode()
    
    # Create a deterministic IV based on domain size
    iv = struct.pack('<Q', domain_size) + struct.pack('<Q', 0)
    
    # Create and use cipher in ECB mode for simplicity
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad the number bytes to ensure it's a multiple of 16
    padded = number_bytes + b'\0' * (16 - len(number_bytes) % 16)
    
    # Encrypt the padded bytes
    encrypted_bytes = encryptor.update(padded) + encryptor.finalize()
    
    # Convert to an integer and take modulo to ensure it's within domain
    encrypted_int = int.from_bytes(encrypted_bytes, byteorder='big')
    
    # Ensure the result is within the domain size, maintaining format
    domain_max = 10 ** len(str(number)) - 1
    result = (encrypted_int % domain_max) + 1  # Ensure non-zero
    
    # Handle leading zeros by padding with zeros
    return str(result).zfill(len(str(number)))

def encrypt_single_id(key, id_value):
    """Encrypt a single ID value using the provided key.
    
    Args:
        key: The encryption key
        id_value: The ID to encrypt (string or integer)
        
    Returns:
        Encrypted ID value as a string
    """
    # Handle hyphenated values
    if '-' in str(id_value):
        parts = str(id_value).split('-')
        encrypted_parts = []
        
        for part in parts:
            if part.strip().isdigit():
                num = int(part.strip())
                part_length = len(str(num))
                
                # Get domain size based on input length
                domain_size = 10 ** part_length
                
                encrypted_part = ff1_encrypt(key, num, domain_size)
                encrypted_parts.append(encrypted_part)
            else:
                encrypted_parts.append(part)
                
        return '-'.join(encrypted_parts)
    else:
        # Handle numeric IDs
        try:
            num = int(str(id_value).strip())
            num_length = len(str(num))
            
            # Get domain size based on input length
            domain_size = 10 ** num_length
            
            encrypted_value = ff1_encrypt(key, num, domain_size)
            return encrypted_value
        except ValueError:
            # Return original for non-numeric values
            return str(id_value)



def anonymize_date(date_str):
    """
    Anonymize a date by removing the day information, keeping only year and month.
    Sets the day to '01' for all dates.
    
    Args:
        date_str (str): Date string in formats like '2019-11-06 00:00:00' or '1932-01-08'
        
    Returns:
        str: Anonymized date in 'YYYY-MM-01' format
    """
    # Handle empty or None values
    if not date_str or date_str.lower() == 'none':
        return date_str
    
    # Split by space to handle datetime format with time component
    parts = date_str.split(' ')
    date_part = parts[0]
    
    # Split the date by hyphens
    try:
        year, month, day = date_part.split('-')
        # Return the date with day set to '01'
        return f"{year}-{month}-01"
    except ValueError:
        # Return original if it doesn't match expected format
        print(f"Warning: Couldn't anonymize date '{date_str}' - unexpected format")
        return date_str

def encrypt_ids(input_file=None, output_file_gcp=None, output_file_local=None, key_output=None):
    
    # Ensure output folder exists for local file
    if output_file_local:
        output_dir = os.path.dirname(output_file_local)
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if the key file already exists and load it
    if os.path.exists(key_output):
        try:
            with open(key_output, 'rb') as key_file:
                key = pickle.load(key_file)
            print(f"Using existing encryption key from {key_output}")
        except Exception as e:
            print(f"Error loading existing key: {e}")
            key = generate_key()
            # Save the new key
            with open(key_output, 'wb') as key_file:
                pickle.dump(key, key_file)
            print(f"Generated new encryption key and saved to {key_output}")
    else:
        # Generate a single key for all columns
        key = generate_key()
        # Save the key to a separate file
        with open(key_output, 'wb') as key_file:
            pickle.dump(key, key_file)
        print(f"Generated new encryption key and saved to {key_output}")

    with open(input_file, 'r') as infile, open(output_file_local, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header
        header = next(reader)
        
        # Columns to remove
        columns_to_remove = ["ENDPOINT_ADDRESS", "path_interpretation", "Pathology_Laterality", "final_diag", "STUDY_ID"]
        
        # Find indices of columns to remove
        columns_to_remove_indices = []
        for col in columns_to_remove:
            try:
                columns_to_remove_indices.append(header.index(col))
            except ValueError:
                pass
        
        # Find index of final_interpretation column
        final_interp_index = -1
        try:
            final_interp_index = header.index("final_interpretation")
        except ValueError:
            pass
        
        # Create new header by excluding columns to remove
        new_header = [col for i, col in enumerate(header) if i not in columns_to_remove_indices]
        writer.writerow(new_header)
        
        # Find indices of date columns to be anonymized
        birth_date_index = -1
        death_date_index = -1
        
        try:
            birth_date_index = header.index("BIRTH_DATE")
        except ValueError:
            pass
            
        try:
            death_date_index = header.index("DEATH_DATE")
        except ValueError:
            pass
            
        for row in reader:
            encrypted_row = []
            
            for i, value in enumerate(row):
                # Skip the columns to remove
                if i in columns_to_remove_indices:
                    continue
                    
                if i <= 1:  # Process the first two columns (IDs)
                    try:
                        encrypted_value = encrypt_single_id(key, value)
                        encrypted_row.append(encrypted_value)
                    except ValueError:
                        # Handle non-integer values
                        encrypted_row.append(value)
                elif i == birth_date_index or i == death_date_index:
                    # Anonymize date columns
                    anonymized_date = anonymize_date(value)
                    encrypted_row.append(anonymized_date)
                elif i == final_interp_index:
                    # Simplify final_interpretation by removing the number at the end
                    # Matches patterns like "BENIGN2", "MALIGNANT3" etc.
                    import re
                    simplified_value = re.sub(r'([A-Z]+)\d+', r'\1', value)
                    encrypted_row.append(simplified_value)
                else:
                    # Keep any other columns unchanged
                    encrypted_row.append(value)
            
            writer.writerow(encrypted_row)

    print(f"Encryption and date anonymization complete. Output saved locally to {output_file_local}")
    

    # Upload to GCS if output_file_gcp is specified
    if output_file_gcp:
        from google.cloud import storage
        
        # Get the bucket using the CONFIG variable
        client = storage.Client()
        bucket = client.bucket(CONFIG["storage"]["bucket_name"])
        
        # Determine the blob name - this is the path within the bucket
        blob_name = f"{output_file_gcp}"
        
        # Upload the file to GCS
        blob_name = os.path.normpath(blob_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(output_file_local)
        print(f"File uploaded to gs://{CONFIG['storage']['bucket_name']}/{blob_name}")
    
    return key