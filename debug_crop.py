from OCR import *
import cv2
import pandas as pd
import os
from tqdm import tqdm

def Crop_Debug():
    
    image_output = f"{env}/debug_output/"
    os.makedirs(image_output, exist_ok=True)
    
    image_folder_path = f"{env}/database/images/"
    input_file = f'{env}/database/ImageData.csv'
    df = pd.read_csv(input_file)
    
    for index, row in tqdm(df.iterrows()):
        
        image_path = os.path.join(image_folder_path, row['ImageName'])
        
        image = cv2.imread(image_path)
        
        # Check if the image was loaded properly
        if image is None:
            print(f"Failed to load image at: {image_path}")
            continue
        
        # Get box coordinates
        x = int(row['crop_x'])
        y = int(row['crop_y'])
        w = int(row['crop_w'])
        h = int(row['crop_h'])
        
        # Check if the box is too vertical
        aspect_ratio = w / h
        if aspect_ratio < 0.25:  # Adjust this threshold as needed
            print(f"Image is too vertical: {row['ImageName']}")
        
        # Draw the box
        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (0, 0, 255)  # Blue color in BGR
        thickness = 4
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(row['description']).upper()  # Convert text to uppercase
        text_color = (0, 255, 0)  # Green color in BGR
        font_scale = 1  # Increase this to make the text bigger
        image = cv2.putText(image, text, (x, y - 10), font, font_scale, text_color, 2, cv2.LINE_AA)
        
        # Save the image with bounding box
        output_path = os.path.join(image_output, row['ImageName'])
        cv2.imwrite(output_path, image)

Crop_Debug()
