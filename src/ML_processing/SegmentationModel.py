from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from numpy import asarray
import cv2, os

env = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = f'{env}/models/BurntInAnnotationDetectingModel.pth'

def prepareImage(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    return preprocess(img)


def predictAnnotation(imagedir):
    """
    Takes in an image directory and returns the predicted annotation.

    Returns: "clean" or "dirty"
    """
    img = Image.open(imagedir).convert('RGB')
    model = prepareModel()

    prediction = predictScore(model, img)
    
    return prediction[0]

def prepareModel():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.load_state_dict(torch.load(model_path))

    model_ft = model_ft.cpu()
    model_ft.eval()
    return model_ft

def predictScore(model, img):
    img = prepareImage(img)
    img = torch.unsqueeze(img, 0)
    img.to(device)
    pred = model(img)
  
    labels = ["clean", "dirty"]
    #
    # Find the index (tensor) corresponding to the maximum score in the out tensor.
    # Torch.max function can be used to find the information
    #
    _, index = torch.max(pred, 1)
    #
    # Find the score in terms of percentage by using torch.nn.functional.softmax function
    # which normalizes the output to range [0,1] and multiplying by 100
    #
    percentage = torch.nn.functional.softmax(pred, dim=1)[0] * 100

    return labels[index[0]], percentage[index[0]].item()


def maskImage(imageDir, maskDir):
    image = cv2.imread(imageDir, 0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 224:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    img = Image.fromarray(mask)
    img.save(maskDir)
    return maskDir