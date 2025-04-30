import numpy as np
import cv2
import os
env = os.path.dirname(os.path.abspath(__file__))
image_folder_path = f'{env}/test_images/'

def has_blue_pixels(image, n=100, min_b=200):
    # Create a mask where blue is dominant
    channel_max = np.argmax(image, axis=-1)
    blue_dominant = (channel_max == 2) & (
        (image[:, :, 2] - image[:, :, 0] >= n) &
        (image[:, :, 2] - image[:, :, 1] >= n)
    )
    
    strong = image[:, :, 2] >= min_b
    return np.any(blue_dominant & strong)

def has_red_pixels(image, n=100, min_r=200):
    # Create a mask where red is dominant
    channel_max = np.argmax(image, axis=-1)
    red_dominant = (channel_max == 0) & (
        (image[:, :, 0] - image[:, :, 2] >= n) &
        (image[:, :, 0] - image[:, :, 1] >= n)
    )
    
    strong = image[:, :, 0] >= min_r
    return np.any(strong & red_dominant)

def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if necessary
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"{filename}: {has_red_pixels(image_rgb)} {has_blue_pixels(image_rgb)}")

# Set the path to your image folder

process_images(image_folder_path)
