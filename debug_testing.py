import csv
import os
from collections import Counter
import pydicom
from cv2 import cv2
from PIL import Image
import numpy as np

env = os.path.dirname(os.path.abspath(__file__))

def check_image_files(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        image_names = []
        for row in reader:
            image_name = row.get('ImagesPath')
            #if not os.path.isfile(f'{env}/export/videos/{image_name}'):
                #print(f"Image file '{image_name}' does not exist or cannot be accessed.")
            image_names.append(image_name)

        duplicate_image_names = [name for name, count in Counter(image_names).items() if count > 1]
        if duplicate_image_names:
            print("Duplicate image names found:")
            for name in duplicate_image_names:
                print(f"Image name '{name}' is duplicated.")

# Usage example
csv_file_path = f'{env}/export/VideoData.csv'
#check_image_files(csv_file_path)




def read_dicom(dicom_file_path):

    # Load the DICOM file
    dicom = pydicom.dcmread(dicom_file_path)

    im = dicom.pixel_array

    np_im = np.array(im)
    
    # check if there is any blue pixel
    is_blue = (np_im[:, :, 0] < 50) & (np_im[:, :, 1] < 50) & (np_im[:, :, 2] > 200)
    if np.any(is_blue):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        # Convert yellow pixels to white
        yellow = [255, 255, 0]  # RGB values for yellow
        white = [255, 255, 255]  # RGB values for white
        mask = np.all(np_im == yellow, axis=-1)
        np_im[mask] = white
        im = np_im

        # Convert to grayscale
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    image_name = f"bruh.png"
    cv2.imwrite(f"{env}/{image_name}", im)


dicom_file_path = f'D:\DATA\CASBUSI\dicoms/00053_dicoms_anon/00053_dicoms_anon/image_00003723_00003946_b7123309f8c9c9040ec97959dc6c04507119f271.dcm'
#read_dicom(dicom_file_path)
