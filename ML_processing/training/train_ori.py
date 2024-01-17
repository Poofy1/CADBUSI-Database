import pandas as pd
import timm
from PIL import Image
import torch, os
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchinfo import summary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from training_util import *
from torchvision import transforms
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
label_dict = {"other": 0, "trans": 1, "long": 2}



def load_image(image_path, image_size):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.35, 1.15), saturation=(0, 1.5), hue=(-0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        StaticNoise(intensity_min=0.01, intensity_max=0.02),
        transforms.Normalize([0.5], [0.5]),
    ])

    return preprocess(image)


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, image_size, label_dict):
        self.df = dataframe
        self.img_dir = img_dir
        self.image_size = image_size
        self.label_dict = label_dict  # add this line
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['patient_id']
        img = load_image(f"{self.img_dir}/{img_id}", self.image_size)
        
        """img_disp = img.clone().detach()  
        img_disp = img_disp * 0.5 + 0.5
        if img_disp.shape[0] == 1:
            img_disp = img_disp.squeeze(0)
        plt.imshow(img_disp.numpy(), cmap='gray')
        plt.show()"""
        
        ori = torch.tensor(self.label_dict[row['Orientation']], dtype=torch.long)
        return img, ori


class Mobile(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Mobile, self).__init__()
        
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



    
    
def test_images(images_dir, model_name, image_size):
    model = Mobile()
    model.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for image_file in image_files:
            image = load_image(os.path.join(images_dir, image_file), image_size)
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU

            ori_pred = model(image)
            ori_pred_prob = softmax(ori_pred)

            # Transfer predictions to CPU, remove batch dimension, convert to appropriate data type
            ori_pred_prob = ori_pred_prob.view(-1).tolist()

            print(f'{image_file}: {[f"{prob:2f}" for prob in ori_pred_prob]}')




def train_model(model, train_loader, val_loader, num_epochs, model_name):
    print("Training..")
    criterion_class = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        total_train = 0
        correct_train = 0
        train_losses = []

        for I, (img, ori) in enumerate(train_loader):
            img = img.to(device)
            ori = ori.to(device)

            optimizer.zero_grad()
            ori_pred = model(img)

            predicted_train = torch.argmax(ori_pred.data, dim=1)
            total_train += ori.size(0)
            correct_train += (predicted_train == ori).sum().item()

            caliper_loss = criterion_class(ori_pred, ori)
            train_losses.append(caliper_loss.item())
            caliper_loss.backward()
            optimizer.step()

        train_loss = sum(train_losses) / len(train_losses) # Average loss
        train_acc = 100 * correct_train / total_train  # Training accuracy

        # Start of validation
        model.eval()  # Set the model to evaluation mode
        val_losses_class = []
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for img, ori in val_loader:
                img = img.to(device)
                ori = ori.to(device)

                ori_pred = model(img)
                predicted_val = torch.argmax(ori_pred.data, dim=1)
                val_total += ori.size(0)
                val_correct += (predicted_val == ori).sum().item()

                loss_class2 = criterion_class(ori_pred, ori)
                val_losses_class.append(loss_class2.item())

        val_loss_class = sum(val_losses_class) / len(val_losses_class)
        val_acc = 100 * val_correct / val_total  # Validation accuracy

        print(f"\nEpoch {epoch+1} Train | Validation \nLoss:     {train_loss:.4f} / {val_loss_class:.4f} \nAccuracy: {train_acc:.1f}% / {val_acc:.1f}%")
        
    torch.save(model.state_dict(), f"{env}/model/{model_name}.pt")
    


def oversample_data():
    # load data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Orientation'])

    # First, split the data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # Get the count of the most frequent class in the training set
    max_class_count = train_df['Orientation'].value_counts().max()

    print(f'Target labels per class: {max_class_count}')

    # Resample each class in the training set
    resampled_dfs = [train_df[train_df['Orientation'] == class_id].sample(max_class_count, replace=True, random_state=42) 
                     for class_id in train_df['Orientation'].unique()]

    # Concatenate the resampled dataframes
    resampled_train_df = pd.concat(resampled_dfs, axis=0)

    # Shuffle the resulting dataframe
    resampled_train_df = resampled_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return resampled_train_df, val_df

def undersample_data():
    # load data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Orientation'])

    # First, split the data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # Get the count of the least frequent class in the training set
    min_class_count = train_df['Orientation'].value_counts().min()

    print(f'Target labels per class: {min_class_count}')

    # Resample each class in the training set
    resampled_dfs = [train_df[train_df['Orientation'] == class_id].sample(min_class_count, replace=False, random_state=42) 
                     for class_id in train_df['Orientation'].unique()]

    # Concatenate the resampled dataframes
    resampled_train_df = pd.concat(resampled_dfs, axis=0)

    # Shuffle the resulting dataframe
    resampled_train_df = resampled_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return resampled_train_df, val_df

def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No previous model found, starting training from scratch.")
    return model

if __name__ == "__main__":
    csv_path = f"{env}/dataset/ori_data.csv"
    img_dir = f"{env}/dataset/orientation_images"

    
    train_df, val_df = oversample_data()
    #train_df, val_df = undersample_data()

    
    image_size = 375
    batch_size = 16
    model_name = "ori_model"
    num_epochs = 5


    train_dataset = CustomDataset(train_df, img_dir, image_size, label_dict)
    val_dataset = CustomDataset(val_df, img_dir, image_size, label_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 2, persistent_workers=True)

    # Initialize the model and move it to the GPU
    model = Mobile().to(device)
    model = load_model(model, f"{env}/model/{model_name}.pt")
    summary(model, input_size=(batch_size, 1, image_size, image_size))

    


    if True:
        train_model(model, train_loader, val_loader, num_epochs, model_name)



    images_dir = f"{env}/test_images/"
    if False:
        test_images(images_dir, model_name, image_size)
    