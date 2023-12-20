import pydicom, os
from PIL import Image
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))

def dicom_to_image(dicom_file_path, output_image_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Convert the pixel data to a numpy array
    image_array = ds.pixel_array

    # Normalize the image array to 0-255 and convert to uint8
    image_array = (np.maximum(image_array,0) / image_array.max()) * 255.0
    image_array = np.uint8(image_array)

    # Convert to a PIL image and save
    image = Image.fromarray(image_array)
    image.save(output_image_path)

# Usage example
dicom_file_path = 'D:/DATA/CASBUSI/dicoms/Batch_10_23_23/image_00015797_00016888_050bacc092754012b7a05f885c530e3786d3be3b2ccd88f90c3f1739bc2bda8a.dcm'
output_image_path = f'{env}/test.png'

dicom_to_image(dicom_file_path, output_image_path)