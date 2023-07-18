from ML_processing.SegmentationModel import predictAnnotation
from ML_processing.UNetModel import genUNetMask
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

env = os.path.dirname(os.path.abspath(__file__))

def Inpaint_Dataset(csv_file_path, input_folder, output_folder, tile_size=256, overlap=84, dilate_radius=5):    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Get only relevant data rows
    data = data[data['label'] == True]
    data = data[data['has_calipers'] == True]
    
    # Defining the structuring element for dilation
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
    
    for index, row in tqdm(data.iterrows()):
        image_name = row['ImageName']
        input_image_path = input_folder + image_name 
        radius = 5
        flags = cv2.INPAINT_TELEA

        original_image = cv2.imread(input_image_path)
        cv2.imwrite(output_folder + 'ORIGINAL' + image_name, original_image)
        height, width, _ = original_image.shape
        final_image = original_image.copy()

        # Adjust step size based on overlap
        step_size = tile_size - overlap

        for i in range(0, height-tile_size, step_size):
            for j in range(0, width-tile_size, step_size):
                tile = original_image[i:i+tile_size, j:j+tile_size]

                # Resize the crop to required dimensions for UNet
                resized_tile = cv2.resize(tile, (tile_size, tile_size))

                mask = genUNetMask(resized_tile)

                # Dilate the mask
                dilated_mask = cv2.dilate(mask, structuring_element)

                inpainted_resized_tile = cv2.inpaint(resized_tile, dilated_mask, radius, flags=flags)

                # Resize back to original tile dimensions
                inpainted_tile = cv2.resize(inpainted_resized_tile, (tile.shape[1], tile.shape[0]))

                # Replace only center part of the tile on the final image to avoid edge artifacts
                final_image[i + overlap//2:i + tile_size - overlap//2, j + overlap//2:j + tile_size - overlap//2] = \
                    inpainted_tile[overlap//2:-overlap//2, overlap//2:-overlap//2]

        cv2.imwrite(output_folder + image_name, final_image)
