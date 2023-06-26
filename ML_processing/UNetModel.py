# USAGE
# python predict.py
# import the necessary packages
import numpy as np
import torch
import cv2, os
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_DATASET_PATH = "./Complete Package/masks/"
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256



def genUNetMask(imagePath):
	"""
	Uses a UNet model to generate a mask for the image.

	Generated mask will always be of the size (256, 256)
	"""

	env = os.path.dirname(os.path.abspath(__file__))
	print(f"{env}/models/UNetModel.pth")
	unet = torch.load(f"{env}/models/UNetModel.pth").to(DEVICE)
	return np.array(makePrediction(unet, imagePath))

def makePrediction(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

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