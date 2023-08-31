import cv2, os, sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

# Get the directory of the current script
env = os.path.dirname(os.path.abspath(__file__))
sys.path.append(env)
from model import UNet

radius = 5
flags = cv2.INPAINT_TELEA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_IMAGE = 256


def genUNetMask(image):
    unet = torch.load(f"{env}/models/UNetModel.pth")  # specify the correct path to your model file
    unet.to(DEVICE)

    return np.array(makePrediction(unet, image))

def makePrediction(model, image):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		image = cv2.resize(image, (INPUT_IMAGE, INPUT_IMAGE))

        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
		predMask = predMask * 255
		predMask = predMask.astype(np.uint8)
		
		image = Image.fromarray(predMask)
		return image

def Inpaint_Dataset(csv_file_path, input_folder, tile_size=256, overlap=84, dilate_radius=5):    
    print("Inpainting Useful Caliper Images")
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Add 'Inpainted' column if not present in the CSV
    if 'Inpainted' not in data.columns:
        data['Inpainted'] = False
    else:
        data['Inpainted'] = data['Inpainted'].where(data['Inpainted'], False)

    # Filter the copied data
    processed_data = data[(data['label'] == True) & 
                          (data['has_calipers'] == True) & 
                          (data['Inpainted'] == False)]

    # Defining the structuring element for dilation
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
    
    for index, row in tqdm(processed_data.iterrows(), total=len(processed_data)):
        

        input_image_path = os.path.join(input_folder, row['ImageName'])
        
        

        original_image = cv2.imread(input_image_path)

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

        #Replace image
        os.remove(input_image_path)
        cv2.imwrite(input_image_path, final_image)

        # Find the index of this row in the original DataFrame and update 'Inpainted'
        original_index = data[data['ImageName'] == row['ImageName']].index[0]
        data.loc[original_index, 'Inpainted'] = True

    # Save the updated DataFrame back to the CSV file
    data.to_csv(csv_file_path, index=False)


