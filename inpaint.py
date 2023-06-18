from SegmentationModel import predictAnnotation
from UNetModel import genUNetMask
import cv2
import os
import pandas as pd
from tqdm import tqdm

env = os.path.dirname(os.path.abspath(__file__))

def run(input, output_path):
    """
    Method which takes in an image and outputs a cropped version of it.

    We use a ResNet18 model to predict if the image has any annotations or not, then use a UNet model to predict the mask of the image.

    We then perform inpainting on the image with the generated mask to remove these annotations.

    If the image goes through inpainting, the outputted image will always be of dimensions (256, 256) due to the UNet model.

    Returns the text found within the image.
    """

    # Generate mask and use inpainting to remove annotations
    mask = genUNetMask(input)

    radius = 3
    flags = cv2.INPAINT_TELEA

    image = cv2.imread(input)
    image = cv2.resize(image, (256, 256))

    finalImage = cv2.inpaint(image, mask, radius, flags=flags)

    # Visualize image, generated mask, and inpainted image
    #cv2.imwrite(f"{env}/output/1.png", image)
    #cv2.imwrite(f"{env}/output/2.png", mask)
    #cv2.imwrite(f"{env}/output/3.png", finalImage)

    cv2.imwrite(output_path, finalImage)

def Inpaint_Dataset(csv_file_path, input_folder, output_folder):
        
         # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Load the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Get only relavent data rows
        data = data[data['label'] == True]
        data = data[data['has_calipers'] == True]
        
        
        for index, row in tqdm(data.iterrows()):
            run(input_folder + row['image_filename'], output_folder + row['image_filename'])