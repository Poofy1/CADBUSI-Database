import pandas as pd
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



def load_image(image_path, image_size):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.35, 1.15), saturation=(0, 1.5), hue=(-0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.75, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        StaticNoise(intensity_min=0.0, intensity_max=0.025),
        transforms.Normalize([0.5], [0.5]),
    ])

    return preprocess(image)


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, image_size):
        self.df = dataframe
        self.img_dir = img_dir
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id = row['patient_id']
        img = load_image(f"{self.img_dir}/{img_id}", self.image_size)
        

        '''img_disp = img.clone().detach()  
        img_disp = img_disp * 0.5 + 0.5
        if img_disp.shape[0] == 1:
            img_disp = img_disp.squeeze(0)
        plt.imshow(img_disp.numpy(), cmap='gray')
        plt.show()'''
        
        has_calipers = torch.tensor(row['has_calipers'], dtype=torch.float32)
        return img, has_calipers
    
    
class Net(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)
        
        # Change the first layer because the images are black and white.
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Change the final layer because this is a binary classification problem.
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)
        
        # Apply sigmoid activation for binary classification
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x
    
    
def test_images(images_dir, model_name, image_size):
    model = Net()
    model.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    with torch.no_grad():
        for image_file in image_files:
            image = load_image(os.path.join(images_dir, image_file), image_size)
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU

            has_calipers_pred = model(image)

            # Transfer predictions to CPU, remove batch dimension, convert to appropriate data type
            has_calipers_pred = has_calipers_pred.view(-1).tolist()[0]

            print(f'{image_file}: {100 * has_calipers_pred:.5f}')




def train_model(model, train_loader, val_loader, num_epochs, model_name):
    print("Training..")
    criterion_class = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        total_train = 0
        correct_train = 0
        train_losses = []

        for I, (img, has_calipers) in enumerate(train_loader):
            img = img.to(device)
            has_calipers = has_calipers.to(device)

            optimizer.zero_grad()
            has_calipers_pred = model(img)

            predicted_train = (has_calipers_pred.data > 0.5).float() # calculate predicted labels
            total_train += has_calipers.size(0)
            correct_train += (predicted_train == has_calipers).sum().item()

            caliper_loss = criterion_class(has_calipers_pred, has_calipers)
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

        with torch.no_grad():  # No need to track the gradients
            for img, has_calipers in val_loader:
                img = img.to(device)
                has_calipers = has_calipers.to(device)

                has_calipers_pred = model(img)
                predicted_val = (has_calipers_pred.data > 0.5).float() # calculate predicted labels
                val_total += has_calipers.size(0)
                val_correct += (predicted_val == has_calipers).sum().item()

                loss_class2 = criterion_class(has_calipers_pred, has_calipers)
                val_losses_class.append(loss_class2.item())

        val_loss_class = sum(val_losses_class) / len(val_losses_class)
        val_acc = 100 * val_correct / val_total  # Validation accuracy

        print(f"\nEpoch {epoch+1} Train | Validation \nLoss:     {train_loss:.4f} / {val_loss_class:.4f} \nAccuracy: {train_acc:.1f}% / {val_acc:.1f}%")
        
    torch.save(model.state_dict(), f"{env}/model/{model_name}.pt")


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need to track the gradients
        for img, has_calipers in test_loader:
            img = img.to(device)
            has_calipers = has_calipers.to(device)

            outputs = model(img)
            predicted = (outputs.data > 0.5).float() # calculate predicted labels
            total += has_calipers.size(0)
            correct += (predicted == has_calipers).sum().item()

    return 100 * correct / total  # Test accuracy

def compare_models(model1, model2, csv_path, img_dir, image_size, batch_size):
    df = pd.read_csv(csv_path)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_dataset = CustomDataset(test_df.reset_index(drop=True), img_dir, image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model1 = model1.to(device)
    model2 = model2.to(device)

    model1_acc = evaluate_model(model1, test_loader)
    model2_acc = evaluate_model(model2, test_loader)

    print(f"Model 1 accuracy: {model1_acc:.2f}%")
    print(f"Model 2 accuracy: {model2_acc:.2f}%")
    
    
    
    

if __name__ == "__main__":
    csv_path = f"{env}/dataset/labeled_data.csv"
    img_dir = f"{env}/dataset/images"

    # load data
    df = pd.read_csv(csv_path)
    
    image_size = 256
    batch_size = 16
    model_name = "caliper_model"
    num_epochs = 20

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # create dataset and dataloader
    train_dataset = CustomDataset(train_df.reset_index(drop=True), img_dir, image_size)
    val_dataset = CustomDataset(val_df.reset_index(drop=True), img_dir, image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and move it to the GPU
    model = Net().to(device)
    summary(model, input_size=(batch_size, 1, image_size, image_size))



    if False:
        train_model(model, train_loader, val_loader, num_epochs, model_name)



    images_dir = f"{env}/test_images/"
    if True:
        test_images(images_dir, model_name, image_size)
        
        
        
    if False:
        # Initialize the models and move them to the GPU
        model1 = Net().to(device)
        model2 = Net().to(device)

        # Load the trained weights
        model1.load_state_dict(torch.load(f'F:\CODE\CASBUSI\CASBUSI-Database\ML_processing/models/caliper_model.pt'))
        model2.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))

        compare_models(model1, model2, csv_path, img_dir, image_size, batch_size)
