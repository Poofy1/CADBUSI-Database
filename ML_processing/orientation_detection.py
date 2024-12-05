from PIL import Image
import torch, os
import torchvision.models as models
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.utils.data import Dataset
from storage_adapter import *
warnings.filterwarnings('ignore')
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        image_names_set = set(db_to_process['ImageName'].values)
        self.images = sorted([img for img in os.listdir(root_dir) if img in image_names_set])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = read_image(img_name, use_pil=True)
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]  
    
class Net(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        
        # Load the pretrained MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Change the first layer because the images are black and white.
        # MobileNetV2's first layer is named "features" and its first item is the conv layer
        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Change the final layer because this is a multi-class classification problem.
        num_ftrs = self.model.classifier[1].in_features

        # Add dropout
        self.dropout = torch.nn.Dropout(0.25)
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, 3)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x


def Find_Orientation(images_dir, model_name, csv_input, image_size=375):
    print("Finding Missing Orientations")
    
    model = Net()
    model.load_state_dict(torch.load(f"{env}/models/{model_name}.pt"))
    model = model.to(device)
    model.eval()
    
    
    # Data loader
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    db_out = read_csv(csv_input)
    if 'reparsed_orientation' not in db_out.columns:
        db_out['reparsed_orientation'] = False
    else:
        db_out['reparsed_orientation'] = db_out['reparsed_orientation'].where(db_out['reparsed_orientation'], False)
        
    db_to_process = db_out[(db_out['orientation'] == 'unknown') 
                       & (db_out['RegionCount'] == 1) 
                       & (db_out['reparsed_orientation'] != True)]
    
    
    
    # Add new column 'reparsed_orientation' and set its value to True
    db_to_process['reparsed_orientation'] = True
    
    dataset = MyDataset(images_dir, db_to_process, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, num_workers = 3)

    results = []
    
    
    
    softmax = nn.Softmax(dim=1)
    label_dict = {0: "unknown", 1: "trans", 2: "long"}

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, total=len(dataloader)):  # Unpack filenames here
            images = images.to(device)
            ori_pred = model(images)  # Predict orientations

            # Apply softmax to the output to get probabilities
            ori_pred_prob = softmax(ori_pred)

            # Get the class indices with max probability
            _, predicted_indices = torch.max(ori_pred_prob, 1)
            
            # Get the class labels
            predicted_labels = [label_dict[i.item()] for i in predicted_indices]

            for i in range(len(filenames)):
                # Append a tuple of filename and predicted orientation to results
                results.append((filenames[i], predicted_labels[i]))


    results_dict = {filename: value for filename, value in results}
    results_series = pd.Series(results_dict)

    db_to_process['orientation'] = db_to_process['ImageName'].map(results_series)

    db_out.update(db_to_process, overwrite=True)

    save_data(db_out, csv_input)