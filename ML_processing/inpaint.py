from ML_processing.SegmentationModel import predictAnnotation
from ML_processing.UNetModel import genUNetMask
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

env = os.path.dirname(os.path.abspath(__file__))

def Inpaint_Dataset(csv_file_path, input_folder, output_folder):    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Get only relevant data rows
    data = data[data['label'] == True]
    data = data[data['has_calipers'] == True]
    
    for index, row in tqdm(data.iterrows()):
        image_name = row['ImageName']
        input_image_path = input_folder + image_name        
        radius = 3
        flags = cv2.INPAINT_TELEA

        original_image = cv2.imread(input_image_path)
        cv2.imwrite(output_folder + 'ORIGINAL' + image_name, original_image)
        height, width, _ = original_image.shape
        final_image = np.zeros_like(original_image)

        tile_size = 256  # the size of the crop for inpainting

        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                tile = original_image[i:min(i+tile_size, height), j:min(j+tile_size, width)]

                # Resize the crop to required dimensions for UNet
                resized_tile = cv2.resize(tile, (tile_size, tile_size))

                mask = genUNetMask(resized_tile)

                inpainted_resized_tile = cv2.inpaint(resized_tile, mask, radius, flags=flags)

                # Resize back to original tile dimensions
                inpainted_tile = cv2.resize(inpainted_resized_tile, (tile.shape[1], tile.shape[0]))


                final_image[i:min(i+tile_size, height), j:min(j+tile_size, width)] = inpainted_tile

        cv2.imwrite(output_folder + image_name, final_image)
