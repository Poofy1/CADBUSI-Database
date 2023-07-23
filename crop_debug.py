from pre_image_processing import *
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
        
        # Draw the box
        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        # Save the image with bounding box
        output_path = os.path.join(image_output, row['ImageName'])
        print(output_path)
        cv2.imwrite(output_path, image)


Crop_Debug()
