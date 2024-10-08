import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random
from segmentation_models_pytorch import Unet
from tqdm import tqdm
from pytorch_msssim import SSIM
from torchvision import transforms
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CaliperDataset(Dataset):
    def __init__(self, csv_file, img_dir, caliper_dir=None, val=False, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.caliper_dir = caliper_dir
        self.transform = transform
        self.save_dir = None
        
        if caliper_dir:
            self.caliper_shapes = [f for f in os.listdir(caliper_dir) if f.endswith('.png')]
        else:
            self.caliper_shapes = []
        
        # Filter the data based on the 'val' column
        if val:
            self.data = self.data[self.data['val'] == 1]
        else:
            self.data = self.data[self.data['val'] == 0]

    def __len__(self):
        return len(self.data)

    def add_calipers(self, image, num_calipers):
        if not self.caliper_dir or not self.caliper_shapes:
            return image  # Return original image if no calipers are to be added
        
        img_array = np.array(image)
        
        # Calculate the inner region (75% of the image)
        inner_width = int(image.width * 0.75)
        inner_height = int(image.height * 0.75)
        x_offset = (image.width - inner_width) // 2
        y_offset = (image.height - inner_height) // 2
        
        for _ in range(num_calipers):
            caliper_name = random.choice(self.caliper_shapes)
            caliper = Image.open(os.path.join(self.caliper_dir, caliper_name)).convert('RGBA')
            
            # Random position within the inner 75% of the image
            x = random.randint(x_offset, x_offset + inner_width - caliper.width)
            y = random.randint(y_offset, y_offset + inner_height - caliper.height)
            
            # Convert caliper to numpy array
            caliper_array = np.array(caliper)
            
            # Create a mask for non-transparent pixels
            mask = caliper_array[:, :, 3] > 0
            
            # Convert caliper to grayscale
            caliper_gray = np.dot(caliper_array[:, :, :3], [0.2989, 0.5870, 0.1140])
            
            # Add caliper to the image
            overlay = img_array[y:y+caliper.height, x:x+caliper.width]
            overlay[mask] = caliper_gray[mask]
            img_array[y:y+caliper.height, x:x+caliper.width] = overlay
        
        return Image.fromarray(img_array)

    def __getitem__(self, idx):
        caliper_img_name = self.data.iloc[idx]['Caliper_Image']
        caliper_img_path = os.path.join(self.img_dir, caliper_img_name)
        caliper_img = Image.open(caliper_img_path).convert('L')  # Convert to grayscale
        
        clean_img_name = self.data.iloc[idx]['Duplicate_Image']
        clean_img_path = os.path.join(self.img_dir, clean_img_name)
        clean_img = Image.open(clean_img_path).convert('L')  # Convert to grayscale
        
        
        if self.caliper_dir:
            num_calipers = random.randint(1, 6)
            synthetic_caliper_img2 = self.add_calipers(clean_img, num_calipers)
            synthetic_caliper_img = self.add_calipers(clean_img, num_calipers)
        else:
            synthetic_caliper_img2 = caliper_img
            synthetic_caliper_img = clean_img
        
        # Save the image with added calipers
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"synthetic_caliper_{idx}.png")
            synthetic_caliper_img.save(save_path)
        
        # Apply the same transform to both images
        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            synthetic_caliper_img2 = self.transform(synthetic_caliper_img2)
            torch.manual_seed(seed)
            synthetic_caliper_img = self.transform(synthetic_caliper_img)
        
        # Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        synthetic_caliper_img2 = normalize(to_tensor(synthetic_caliper_img2))
        synthetic_caliper_img = normalize(to_tensor(synthetic_caliper_img))
        
        return synthetic_caliper_img2, synthetic_caliper_img
    
def collate_fn(batch):
    caliper_images, original_images = zip(*batch)
    
    # Get the maximum dimensions in the batch
    max_height = max(max(img.shape[1] for img in caliper_images), 
                     max(img.shape[1] for img in original_images))
    max_width = max(max(img.shape[2] for img in caliper_images), 
                    max(img.shape[2] for img in original_images))
    
    # Round up to nearest multiple of 32
    max_height = ((max_height + 31) // 32) * 32
    max_width = ((max_width + 31) // 32) * 32
    
    # Pad images to the maximum size
    padded_caliper = []
    padded_original = []
    for caliper, original in zip(caliper_images, original_images):
        # Pad caliper image
        pad_height_c = max_height - caliper.shape[1]
        pad_width_c = max_width - caliper.shape[2]
        padding_c = (0, pad_width_c, 0, pad_height_c)  # left, right, top, bottom
        padded_caliper.append(torch.nn.functional.pad(caliper, padding_c, mode='constant', value=0))
        
        # Pad original image
        pad_height_o = max_height - original.shape[1]
        pad_width_o = max_width - original.shape[2]
        padding_o = (0, pad_width_o, 0, pad_height_o)  # left, right, top, bottom
        padded_original.append(torch.nn.functional.pad(original, padding_o, mode='constant', value=0))
    
    # Stack the padded images
    stacked_caliper = torch.stack(padded_caliper)
    stacked_original = torch.stack(padded_original)
    
    return stacked_caliper, stacked_original


from torchvision.transforms import functional as F

def train_model(csv_file, img_dir, caliper_dir, model_path, num_epochs=50):
    
    # Initialize U-Net model
    model = Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=1)
    
    # Check if a saved model exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print("Existing model loaded. Continuing training...")
    else:
        print("No existing model found. Initializing new model...")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2),
    ])

    # Create dataset
    train_dataset = CaliperDataset(csv_file, img_dir, caliper_dir, transform=None)
    val_dataset = CaliperDataset(csv_file, img_dir, caliper_dir, val=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=3, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)

    # Loss function and optimizer
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for caliper_images, original_images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            caliper_images = caliper_images.to(device)
            original_images = original_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(caliper_images)
            loss = criterion(outputs, original_images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for caliper_images, original_images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                caliper_images = caliper_images.to(device)
                original_images = original_images.to(device)
                
                outputs = model(caliper_images)
                loss = criterion(outputs, original_images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the trained model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved. Best validation loss: {best_val_loss:.4f}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    

# Function to remove padding and resize to original dimensions
def post_process(output, original_size):
    # Remove padding
    h, w = original_size
    output = output[:, :h, :w]
    
    # Denormalize
    output = output * 0.5 + 0.5
    
    # Clamp values to [0, 1] range
    output = torch.clamp(output, 0, 1)
    
    # Convert to PIL Image and resize
    output = transforms.ToPILImage()(output)
    output = output.resize((w, h), Image.BILINEAR)
    
    return output

# Example usage of trained model
def remove_calipers(model, image_path):
    image = Image.open(image_path).convert('L')
    original_size = image.size
    
    # Preprocess
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    image = normalize(to_tensor(image)).unsqueeze(0)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image.to(device))
    
    # Post-process
    output = output.squeeze(0).cpu()
    output = post_process(output, original_size)
    
    return output
    
import os
import numpy as np
from PIL import Image

def show_validation_examples(model, data_dir, csv_file, img_dir, num_examples=5):
    val_dataset = CaliperDataset(csv_file, img_dir, val=True)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create results directory
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (caliper_images, original_images) in enumerate(val_loader):
            if i >= num_examples:
                break

            caliper_images = caliper_images.to(device)
            original_images = original_images.to(device)

            outputs = model(caliper_images)

            # Convert tensors to images
            caliper_img = post_process(caliper_images.squeeze(0).cpu(), original_images.shape[2:])
            output_img = post_process(outputs.squeeze(0).cpu(), original_images.shape[2:])
            original_img = post_process(original_images.squeeze(0).cpu(), original_images.shape[2:])

            # Save images
            caliper_img.save(os.path.join(results_dir, f"example_{i+1}_1input.png"))
            output_img.save(os.path.join(results_dir, f"example_{i+1}_2output.png"))
            original_img.save(os.path.join(results_dir, f"example_{i+1}_3ground_truth.png"))

    print(f"Validation examples saved to {results_dir}")



if __name__ == "__main__":
    data_dir = "D:/DATA/CASBUSI/PairExport"
    csv_file = os.path.join(data_dir, "PairData.csv")
    img_dir = os.path.join(data_dir, "images")
    caliper_dir = "D:/DATA/CASBUSI/PairExport/unique_caliper_shapes"
    model_path = os.path.join(data_dir, "caliper_removal_unet_l1loss_N2N_3.pth")

    choice = input("Do you want to (1) train the model or (2) show validation examples? Enter 1 or 2: ")

    if choice == '1':
        train_model(csv_file, img_dir, caliper_dir, model_path)
    elif choice == '2':
        if os.path.exists(model_path):
            model = Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=1)
            model.load_state_dict(torch.load(model_path))
            show_validation_examples(model, data_dir, csv_file, img_dir)
        else:
            print("No pre-trained model found. Please train the model first.")
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")