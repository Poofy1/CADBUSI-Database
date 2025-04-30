import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim

def extract_shapes(image):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the borders to ignore (25% on each side)
    border_x = int(width * 0.25)
    border_y = int(height * 0.25)
    
    # Crop the image to ignore borders
    cropped_image = image[border_y:height-border_y, border_x:width-border_x]
    
    # Use alpha channel as mask
    if cropped_image.shape[2] == 4:
        mask = cropped_image[:,:,3]
    else:
        raise ValueError("Image doesn't have an alpha channel")
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the shape with alpha channel
        shape = cropped_image[y:y+h, x:x+w]
        
        # Only consider shapes above a certain size
        if w > 10 and h > 10:
            shapes.append(shape)
    
    return shapes

def is_similar(img1, img2, threshold=0.9):
    # Resize images to same dimensions
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    
    # Compare only the alpha channels
    alpha1 = img1[:,:,3]
    alpha2 = img2[:,:,3]
    
    # Compute SSIM between two alpha channels
    score, _ = ssim(alpha1, alpha2, full=True)
    return score > threshold

def extract_calipers(caliper_img, duplicate_img):
    # Ensure both images are the same size
    if caliper_img.shape != duplicate_img.shape:
        return None
    
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
    return cv2.merge([b, g, r, mask])

def main():
    # Set up directories
    dir = r"D:\DATA\CASBUSI\PairExport/"
    df = pd.read_csv(f'{dir}/PairData.csv')
    
    # Create output directories
    caliper_dir = f'{dir}caliper_images'
    unique_shapes_dir = f'{dir}unique_caliper_shapes'
    
    for output_dir in [caliper_dir, unique_shapes_dir]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # First pass: Extract calipers from image pairs
    print("Phase 1: Extracting calipers from image pairs...")
    for index, row in df.iterrows():
        # Read the images
        caliper_img = cv2.imread(os.path.join(f'{dir}images', row['Caliper_Image']), cv2.IMREAD_UNCHANGED)
        duplicate_img = cv2.imread(os.path.join(f'{dir}images', row['Duplicate_Image']), cv2.IMREAD_UNCHANGED)
        
        caliper_only = extract_calipers(caliper_img, duplicate_img)
        if caliper_only is None:
            print(f"Skipping {row['Caliper_Image']} due to size mismatch")
            continue
        
        # Save the result
        output_filename = f"{row['Patient_ID']}_{row['Accession_Number']}_caliper.png"
        cv2.imwrite(os.path.join(caliper_dir, output_filename), caliper_only)
    
    # Second pass: Find unique shapes
    print("\nPhase 2: Finding unique caliper shapes...")
    unique_shapes = []
    
    for index, row in df.iterrows():
        caliper_filename = f"{row['Patient_ID']}_{row['Accession_Number']}_caliper.png"
        caliper_path = os.path.join(caliper_dir, caliper_filename)
        
        if not os.path.exists(caliper_path):
            print(f"File not found: {caliper_path}")
            continue
        
        caliper_img = cv2.imread(caliper_path, cv2.IMREAD_UNCHANGED)
        
        if caliper_img.shape[2] != 4:
            print(f"Image {caliper_filename} doesn't have an alpha channel. Skipping.")
            continue
        
        # Extract shapes from the image
        shapes = extract_shapes(caliper_img)
        
        for shape in shapes:
            is_unique = True
            for unique_shape in unique_shapes:
                if is_similar(shape, unique_shape):
                    is_unique = False
                    break
            
            if is_unique:
                unique_shapes.append(shape)
                # Save the unique shape
                output_filename = f"unique_shape_{len(unique_shapes)}.png"
                cv2.imwrite(os.path.join(unique_shapes_dir, output_filename), shape)
    
    print(f"\nProcessing complete. Found {len(unique_shapes)} unique caliper shapes.")

if __name__ == "__main__":
    main()