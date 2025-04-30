import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
import random

def display_dicom(dicom_file_path):
    try:
        # Load the DICOM file
        ds = pydicom.dcmread(dicom_file_path, force=True)
        
        # Print all metadata
        print("DICOM Metadata:")
        print("=" * 50)
        for elem in ds:
            if elem.VR != "SQ":  # Skip sequence items to avoid excessive output
                try:
                    if elem.name != "Pixel Data":
                        print(f"{elem.name}: {elem.value}")
                except:
                    print(f"{elem.tag}: Unable to display value")
        
        # Fix metadata BEFORE accessing pixel_array to prevent warnings and errors
        if hasattr(ds, 'BitsStored') and hasattr(ds, 'BitsAllocated') and hasattr(ds, 'HighBit'):
            print("\nOriginal bit depth settings:")
            print(f"Bits Allocated: {ds.BitsAllocated}, Stored: {ds.BitsStored}, High Bit: {ds.HighBit}")
            
            # Check if this might be a JPEG2000 encoded image with 16-bit data
            if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
                # JPEG2000 transfer syntaxes
                jpeg2000_syntaxes = [
                    '1.2.840.10008.1.2.4.90',  # JPEG 2000 Lossless
                    '1.2.840.10008.1.2.4.91'   # JPEG 2000 Lossy
                ]
                
                # If it's JPEG2000 and has 8-bit stored in 16-bit, fix it
                if (str(ds.file_meta.TransferSyntaxUID) in jpeg2000_syntaxes and 
                    ds.BitsAllocated == 16 and ds.BitsStored == 8):
                    print("Fixing bit depth for JPEG2000 image...")
                    ds.BitsStored = 16
                    ds.HighBit = 15
                    print(f"Updated: Bits Allocated: {ds.BitsAllocated}, Stored: {ds.BitsStored}, High Bit: {ds.HighBit}")
        
        # Display the image
        print("\nDisplaying image...")
        
        # Check if the file has pixel data
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            
            # Print shape information to debug
            print(f"Pixel array shape: {pixel_array.shape}")
            print(f"Pixel data type: {pixel_array.dtype}")
            print(f"Value range: {np.min(pixel_array)} to {np.max(pixel_array)}")
            
            # Check if this is a multi-frame (video) DICOM
            if len(pixel_array.shape) > 2:
                # Check dimensions to determine if it's really a multi-frame
                # or just a 2D image with color channels
                if len(pixel_array.shape) == 3 and pixel_array.shape[2] <= 4:
                    # This is likely a 2D image with color channels (RGB/RGBA)
                    print("This appears to be a 2D color image")
                    plt.figure(figsize=(10, 8))
                    plt.imshow(pixel_array)
                    plt.title(f"DICOM Image: {os.path.basename(dicom_file_path)}")
                else:
                    # This is likely a true multi-frame image
                    print(f"Multi-frame DICOM detected with shape: {pixel_array.shape}")
                    
                    # For multi-frame data, select a random frame
                    # Assuming first dimension is the frame count
                    num_frames = pixel_array.shape[0]
                    random_frame_idx = random.randint(0, num_frames - 1)
                    print(f"Displaying frame {random_frame_idx} of {num_frames}")
                    
                    # Extract the selected frame
                    frame = pixel_array[random_frame_idx]
                    
                    plt.figure(figsize=(10, 8))
                    if len(frame.shape) > 2:
                        plt.imshow(frame)  # For RGB data
                    else:
                        plt.imshow(frame, cmap=plt.cm.bone)  # For grayscale data
                    
                    plt.title(f"DICOM Video - Frame {random_frame_idx}/{num_frames}")
            else:
                # For single-frame grayscale images
                plt.figure(figsize=(10, 8))
                plt.imshow(pixel_array, cmap=plt.cm.bone)
                plt.title(f"DICOM Image: {os.path.basename(dicom_file_path)}")
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("This DICOM file does not contain image data.")
            
    except Exception as e:
        print(f"Error processing DICOM file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    display_dicom("D:/DATA/CASBUSI/dicoms/2025-04-16_225753/3379232_17381272/1.2.840.114350.2.451.2.798268.2.2222618735938.1/image_03379232_17381272_186d07eda02e05266693ac3e2d71d894a521df348a561b86a10f69c2f805d7f1.dcm")