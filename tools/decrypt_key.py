#!/usr/bin/env python3
"""
Test script for encryption/decryption functionality
"""

import sys
import os
import pickle
import base64

# Add the parent directory to the path to find the 'src' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the encryption key manually from ./encryption_key.pkl
key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encryption_key.pkl")

try:
    with open(key_file_path, 'rb') as key_file:
        encryption_key = pickle.load(key_file)
    print(f"✓ Successfully loaded encryption key from {key_file_path}")
    print(f"Key (base64): {base64.b64encode(encryption_key).decode()}")
    
    # Set the key as an environment variable so get_encryption_key() can find it
    os.environ["DICOM_ENCRYPTION_KEY"] = base64.b64encode(encryption_key).decode()
    print("✓ Key set in environment variable for decrypt functions")
    
except FileNotFoundError:
    print(f"✗ Error: Could not find key file at {key_file_path}")
    print("Make sure encryption_key.pkl exists in the same directory as this script")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading key file: {e}")
    sys.exit(1)

# Import the functions from your encryption module
from src.encrypt_keys import get_encryption_key, encrypt_single_id, decrypt_single_id

def test_encryption_decryption():
    """Test the encryption and decryption functions"""
    
    print("\n=== Encryption/Decryption Test Script ===\n")
    
    # Verify the key is working
    try:
        loaded_key = get_encryption_key()
        print(f"✓ get_encryption_key() is now working correctly")
        print(f"Keys match: {encryption_key == loaded_key}")
        print()
    except Exception as e:
        print(f"✗ Error with get_encryption_key(): {e}")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Encrypt a value")
        print("2. Decrypt a value")
        print("3. Test encryption/decryption round-trip")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Encryption test
            value = input("Enter value to encrypt: ").strip()
            if value:
                try:
                    encrypted = encrypt_single_id(encryption_key, value)
                    print(f"Original:  {value}")
                    print(f"Encrypted: {encrypted}")
                except Exception as e:
                    print(f"Encryption error: {e}")
            else:
                print("Please enter a valid value")
                
        elif choice == '2':
            # Decryption test
            encrypted_value = input("Enter encrypted value to decrypt: ").strip()
            if encrypted_value:
                try:
                    decrypted = decrypt_single_id(encrypted_value)
                    print(f"Encrypted: {encrypted_value}")
                    print(f"Decrypted: {decrypted}")
                except Exception as e:
                    print(f"Decryption error: {e}")
            else:
                print("Please enter a valid encrypted value")
                
        elif choice == '3':
            # Round-trip test
            value = input("Enter value for round-trip test: ").strip()
            if value:
                try:
                    encrypted = encrypt_single_id(encryption_key, value)
                    decrypted = decrypt_single_id(encrypted)
                    
                    print(f"Original:  {value}")
                    print(f"Encrypted: {encrypted}")
                    print(f"Decrypted: {decrypted}")
                    print(f"Round-trip successful: {value == decrypted}")
                    
                    if value != decrypted:
                        print("WARNING: Round-trip failed! Original and decrypted values don't match.")
                        
                except Exception as e:
                    print(f"Round-trip test error: {e}")
            else:
                print("Please enter a valid value")
                
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    print("Starting encryption/decryption tests...\n")
    
    # Then run interactive tests
    test_encryption_decryption()