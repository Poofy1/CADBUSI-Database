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
from tools.storage_adapter import *
from src.DB_processing.database import DatabaseManager
warnings.filterwarnings('ignore')
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all files from the directory
        all_files = list_files(root_dir)

        # Extract just the filenames from the full paths
        file_dict = {os.path.basename(img): img for img in all_files}

        # Filter by the database image names and store only the filenames (not full paths)
        self.images = sorted([os.path.basename(file_dict[img_name]) for img_name in db_to_process['image_name'].values 
                            if img_name in file_dict])

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


def Find_Orientation(CONFIG, image_size=375):
    print("Finding Missing Orientations")
    
    images_dir = f'{CONFIG["DATABASE_DIR"]}/images/'
    model_name = 'ori_model'
    
    with DatabaseManager() as db:
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
        
        # Read breast data to get patients with BILATERAL laterality
        breast_df = db.get_study_cases_dataframe()
        bilateral_patients = breast_df[breast_df['study_laterality'] == 'BILATERAL']['patient_id'].unique()
        
        # Read image data
        image_df = db.get_images_dataframe()
        
        if 'reparsed_orientation' not in image_df.columns:
            image_df['reparsed_orientation'] = False
        else:
            image_df['reparsed_orientation'] = image_df['reparsed_orientation'].where(image_df['reparsed_orientation'], False)
            
        # Add orientation_confidence column if it doesn't exist
        if 'orientation_confidence' not in image_df.columns:
            image_df['orientation_confidence'] = None
        
        # Filter to only process images for bilateral patients
        db_to_process = image_df[(image_df['orientation'] == 'unknown') 
                         & (image_df['region_count'] == 1) 
                         & (image_df['reparsed_orientation'] != True)
                         & (image_df['patient_id'].isin(bilateral_patients))]
        
        # If no images to process, return early
        if len(db_to_process) == 0:
            print("No bilateral patient images to process")
            return
            
        # Add new column 'reparsed_orientation' and set its value to True
        db_to_process['reparsed_orientation'] = True
        
        dataset = MyDataset(images_dir, db_to_process, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=3)

        orientation_results = []
        confidence_results = []
        
        softmax = nn.Softmax(dim=1)
        label_dict = {0: "unknown", 1: "trans", 2: "long"}

        with torch.no_grad():
            for images, filenames in tqdm(dataloader, total=len(dataloader)):
                images = images.to(device)
                ori_pred = model(images)

                # Apply softmax to the output to get probabilities
                ori_pred_prob = softmax(ori_pred)

                # Get the class indices with max probability
                max_probs, predicted_indices = torch.max(ori_pred_prob, 1)
                
                # Get the class labels
                predicted_labels = [label_dict[i.item()] for i in predicted_indices]

                for i in range(len(filenames)):
                    # Store filename with predicted orientation and confidence
                    orientation_results.append((filenames[i], predicted_labels[i]))
                    confidence_results.append((filenames[i], max_probs[i].item()))

        # Update database with results
        cursor = db.conn.cursor()
        
        for filename, orientation in orientation_results:
            cursor.execute("""
                UPDATE Images
                SET orientation = ?,
                    reparsed_orientation = 1
                WHERE image_name = ?
            """, (orientation, filename))
        
        for filename, confidence in confidence_results:
            cursor.execute("""
                UPDATE Images
                SET orientation_confidence = ?
                WHERE image_name = ?
            """, (confidence, filename))
        
        db.conn.commit()
        
        print(f"Processed {len(db_to_process)} images for {len(bilateral_patients)} bilateral patients")