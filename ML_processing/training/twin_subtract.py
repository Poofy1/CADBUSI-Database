import cv2
import numpy as np
import pandas as pd
import os

# Read the CSV file
dir = r"D:\DATA\CASBUSI\PairExport/"
df = pd.read_csv(f'{dir}/PairData.csv')

# Create output directory if it doesn't exist
output_dir = f'{dir}caliper_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each pair of images
for index, row in df.iterrows():
    # Read the images
    caliper_img = cv2.imread(os.path.join(f'{dir}images', row['Caliper_Image']), cv2.IMREAD_UNCHANGED)
    duplicate_img = cv2.imread(os.path.join(f'{dir}images', row['Duplicate_Image']), cv2.IMREAD_UNCHANGED)
    
    # Ensure both images are the same size
    if caliper_img.shape != duplicate_img.shape:
        print(f"Skipping {row['Caliper_Image']} due to size mismatch")
        continue
    
    # Compute the absolute difference
    diff = cv2.absdiff(caliper_img, duplicate_img)
    
    # If the image is already grayscale, we don't need to convert
    if len(diff.shape) == 2 or diff.shape[2] == 1:
        gray_diff = diff if len(diff.shape) == 2 else diff[:,:,0]
    else:
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold to create a binary mask
    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)

    # If the original image is grayscale, convert to BGR
    if len(caliper_img.shape) == 2 or caliper_img.shape[2] == 1:
        caliper_img = cv2.cvtColor(caliper_img, cv2.COLOR_GRAY2BGR)
    
    # Create a 4-channel image (BGR + alpha)
    b, g, r = cv2.split(caliper_img)
    caliper_only = cv2.merge([b, g, r, mask])
    
    # Save the result
    output_filename = f"{row['Patient_ID']}_{row['Accession_Number']}_caliper.png"
    cv2.imwrite(os.path.join(output_dir, output_filename), caliper_only)


print("All images processed.")