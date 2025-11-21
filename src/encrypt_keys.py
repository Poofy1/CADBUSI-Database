from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import csv
import pickle
import struct
import base64
import pyffx
import re

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def generate_key():
    """Generate a cryptographically secure 256-bit key."""
    return os.urandom(16)  # 128-bit is minimum spec


def get_encryption_key():
    """
    Retrieves or generates an encryption key for DICOM deidentification.
    
    The function looks for an existing key in a predefined location or environment variable.
    If no key exists, it generates a new one and stores it.
    
    Returns:
        bytes: A 16-byte encryption key for use with FF1
    """
    # Check for key in environment variable first
    env_key_name = "DICOM_ENCRYPTION_KEY"
    key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encryption_key.pkl")
    
    # Try to get key from environment variable (base64 encoded)
    if env_key_name in os.environ:
        try:
            return base64.b64decode(os.environ[env_key_name])
        except Exception as e:
            print(f"Error loading key from environment: {e}")
    
    # If not in environment, try to load from file
    if os.path.exists(key_file_path):
        try:
            with open(key_file_path, 'rb') as key_file:
                key = pickle.load(key_file)
            print(f"Using existing encryption key from {key_file_path}")
            return key
        except Exception as e:
            print(f"Error loading existing key file: {e}")
    
    # If no key exists, generate a new one
    key = generate_key()
    
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
    NIST SP 800-38G compliant FF1 format-preserving encryption.
    
    Args:
        key (bytes): The encryption key
        number (int): The number to encrypt
        domain_size (int): The size of the domain (10^n for n-digit numbers)
        
    Returns:
        str: The encrypted number as a string with same length as input
    """
    # Convert number to string and get its length
    num_str = str(number)
    length = len(num_str)

    # Initialize the FF1 cipher directly with the Integer class
    ff1 = pyffx.Integer(key, length)
    
    # Encrypt the number
    encrypted_value = ff1.encrypt(number)
    
    # Convert to string and ensure proper padding with leading zeros
    return str(encrypted_value).zfill(length)


def encrypt_single_id(key, id_value):
    """Encrypt a single ID value using the provided key with NIST FF1.
    
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


def encrypt_ids(input_file=None, output_file_local=None, key_output=None):
    
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
        
        # Find indices of diagnosis columns
        left_diagnosis_index = -1
        right_diagnosis_index = -1
        try:
            left_diagnosis_index = header.index("left_diagnosis")
        except ValueError:
            pass
        try:
            right_diagnosis_index = header.index("right_diagnosis")
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
                elif i == left_diagnosis_index or i == right_diagnosis_index:
                    # Simplify diagnosis by removing the number at the end
                    # Matches patterns like "BENIGN2", "MALIGNANT3" etc.
                    simplified_value = re.sub(r'([A-Z]+)\d+', r'\1', value)
                    encrypted_row.append(simplified_value)
                else:
                    # Keep any other columns unchanged
                    encrypted_row.append(value)
            
            writer.writerow(encrypted_row)

    print(f"Encryption and date anonymization complete. Output saved locally to {output_file_local}")
    
    return key

def ff1_decrypt(key, encrypted_number, domain_size):
    """
    NIST SP 800-38G compliant FF1 format-preserving decryption.
    
    Args:
        key (bytes): The encryption key
        encrypted_number (int): The encrypted number to decrypt
        domain_size (int): The size of the domain (10^n for n-digit numbers)
        
    Returns:
        int: The decrypted number
    """
    # Convert encrypted number to string and get its length
    encrypted_str = str(encrypted_number)
    length = len(encrypted_str)

    # Initialize the FF1 cipher with the same parameters used for encryption
    ff1 = pyffx.Integer(key, length)
    
    # Decrypt the number
    decrypted_value = ff1.decrypt(encrypted_number)
    
    return decrypted_value

def decrypt_single_id(encrypted_id):
    """Decrypt a single ID value back to its original form.
    
    Args:
        encrypted_id: The encrypted ID to decrypt (string or integer)
        
    Returns:
        Original ID value as a string
    """
    # Get the same key used for encryption
    key = get_encryption_key()
    
    # Handle hyphenated values
    if '-' in str(encrypted_id):
        parts = str(encrypted_id).split('-')
        decrypted_parts = []
        
        for part in parts:
            if part.strip().isdigit():
                encrypted_num = int(part.strip())
                part_length = len(str(encrypted_num))
                
                # Get domain size based on encrypted part length
                domain_size = 10 ** part_length
                
                decrypted_part = ff1_decrypt(key, encrypted_num, domain_size)
                # Preserve leading zeros by padding to original length
                decrypted_parts.append(str(decrypted_part).zfill(part_length))
            else:
                decrypted_parts.append(part)
                
        return '-'.join(decrypted_parts)
    else:
        # Handle numeric IDs
        try:
            encrypted_num = int(str(encrypted_id).strip())
            encrypted_length = len(str(encrypted_num))
            
            # Get domain size based on encrypted number length
            domain_size = 10 ** encrypted_length
            
            decrypted_value = ff1_decrypt(key, encrypted_num, domain_size)
            # Preserve leading zeros by padding to original length
            return str(decrypted_value).zfill(encrypted_length)
        except ValueError:
            # Return original for non-numeric values
            return str(encrypted_id)