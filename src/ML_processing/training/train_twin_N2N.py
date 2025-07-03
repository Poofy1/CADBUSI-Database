import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import random
from tqdm import tqdm
from torchvision import transforms
import numpy as np




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F

class N2N_Original_Used_UNet(nn.Module):
    """Custom U-Net architecture from original Noise2Noise Paper (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1):
        """Initializes U-Net."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(N2N_Original_Used_UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)
        
        """# Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)"""
        
        # Decoder
        upsample5 = self._block3(pool5)
        upsample5 = F.interpolate(upsample5, size=pool4.shape[2:], mode='bilinear', align_corners=False)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        
        upsample4 = self._block4(concat5)
        upsample4 = F.interpolate(upsample4, size=pool3.shape[2:], mode='bilinear', align_corners=False)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        
        upsample3 = self._block5(concat4)
        upsample3 = F.interpolate(upsample3, size=pool2.shape[2:], mode='bilinear', align_corners=False)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        
        upsample2 = self._block5(concat3)
        upsample2 = F.interpolate(upsample2, size=pool1.shape[2:], mode='bilinear', align_corners=False)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        
        upsample1 = self._block5(concat2)
        upsample1 = F.interpolate(upsample1, size=x.shape[2:], mode='bilinear', align_corners=False)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)
    
    
    
class CaliperDataset(Dataset):
    def __init__(self, csv_file, img_dir, caliper_dir=None, val=False, transform=None, subset_limit=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.caliper_dir = caliper_dir
        self.transform = transform
        self.save_dir = None #"D:\DATA\CASBUSI\PairExport/test/"
        
        if caliper_dir:
            self.caliper_shapes = [f for f in os.listdir(caliper_dir) if f.endswith('.png')]
        else:
            self.caliper_shapes = []
        
        # Filter the data based on the 'val' column
        if val:
            self.data = self.data[self.data['val'] == 1]
        else:
            self.data = self.data[self.data['val'] == 0]
            
        # Limit the dataset to a subset if specified
        if subset_limit is not None and subset_limit > 0:
            self.data = self.data.iloc[:subset_limit]

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
            
            # Randomly scale the caliper (50% to 150% of original size)
            scale_factor = random.uniform(0.5, 1.5)
            new_width = int(caliper.width * scale_factor)
            new_height = int(caliper.height * scale_factor)
            
            # Resize without antialiasing
            caliper = caliper.resize((new_width, new_height), Image.NEAREST)
            
            # 50% chance to apply rotation and skew
            if random.random() < 0.5:
                # Random rotation angle (-45 to 45 degrees)
                rotation_angle = random.uniform(-45, 45)
                
                # Random skew factors (-0.3 to 0.3)
                skew_factor_x = random.uniform(-0.3, 0.3)
                skew_factor_y = random.uniform(-0.3, 0.3)
                
                # Create an affine transform matrix for skewing
                skew_matrix = [1, skew_factor_x, 0,
                            skew_factor_y, 1, 0]
                
                # Apply transformations: rotate and skew
                caliper = caliper.rotate(rotation_angle, expand=True, resample=Image.NEAREST)
                caliper = caliper.transform(
                    caliper.size, 
                    Image.AFFINE, 
                    skew_matrix,
                    resample=Image.NEAREST
                )
                
            # Convert to numpy array and create mask
            caliper_array = np.array(caliper)
            mask = caliper_array[:, :, 3] > 128  # Use a threshold to determine solid pixels
            
            # Convert caliper to grayscale
            caliper_gray = np.dot(caliper_array[:, :, :3], [0.2989, 0.5870, 0.1140])
            
            # Get the dimensions after all transformations
            final_width, final_height = caliper.size
            
            # Ensure the transformed caliper fits within the inner region
            max_x = max(0, inner_width - final_width)
            max_y = max(0, inner_height - final_height)
            
            if max_x > 0 and max_y > 0:
                # Random position within the inner 75% of the image
                x = random.randint(x_offset, x_offset + max_x)
                y = random.randint(y_offset, y_offset + max_y)
                
                # Add caliper to the image
                try:
                    overlay = img_array[y:y+final_height, x:x+final_width]
                    if mask.shape == overlay.shape:
                        overlay[mask] = caliper_gray[mask]
                        img_array[y:y+final_height, x:x+final_width] = overlay
                except ValueError:
                    continue  # Skip if dimensions don't match
        
        # Add random box outline (can be anywhere on the image)
        # Random box dimensions (20% to 80% of image dimensions)
        box_width = random.randint(int(image.width * 0.2), int(image.width * 0.8))
        box_height = random.randint(int(image.height * 0.2), int(image.height * 0.8))
        
        # Random position (ensure box fits within image)
        max_x = image.width - box_width
        max_y = image.height - box_height
        box_x = random.randint(0, max_x)
        box_y = random.randint(0, max_y)
        
        # Random outline thickness (1 to 5 pixels)
        thickness = random.randint(1, 5)
        
        # Random grayscale color for the outline (0 to 255)
        outline_color = random.randint(0, 255)
        
        # Draw the box outline
        # Top edge
        img_array[box_y:box_y+thickness, box_x:box_x+box_width] = outline_color
        # Bottom edge
        img_array[box_y+box_height-thickness:box_y+box_height, box_x:box_x+box_width] = outline_color
        # Left edge
        img_array[box_y:box_y+box_height, box_x:box_x+thickness] = outline_color
        # Right edge
        img_array[box_y:box_y+box_height, box_x+box_width-thickness:box_x+box_width] = outline_color
        
        return Image.fromarray(img_array)

    def __getitem__(self, idx):
        caliper_img_name = self.data.iloc[idx]['Caliper_Image']
        caliper_img_path = os.path.join(self.img_dir, caliper_img_name)
        caliper_img = Image.open(caliper_img_path).convert('L')  # Convert to grayscale
        
        clean_img_name = self.data.iloc[idx]['Duplicate_Image']
        clean_img_path = os.path.join(self.img_dir, clean_img_name)
        clean_img = Image.open(clean_img_path).convert('L')  # Convert to grayscale
        
        
        if self.caliper_dir:
            num_calipers = random.randint(1, 14)
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

def train_model(csv_file, img_dir, caliper_dir, model_path, num_epochs=50):
    
    # Initialize U-Net model
    #model = Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=1)
    model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
    
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
    results_dir = os.path.join(data_dir, "results2")
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




def evaluate_metrics(model, data_dir, csv_file, img_dir):
    """
    Evaluate PSNR and SSIM metrics on the validation set.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    val_dataset = CaliperDataset(csv_file, img_dir, val=True, subset_limit = 1000)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    print("Calculating PSNR and SSIM metrics...")
    with torch.no_grad():
        for caliper_images, ground_truth_images in tqdm(val_loader, desc="Evaluating metrics"):
            # Move to device
            caliper_images = caliper_images.to(device)
            ground_truth_images = ground_truth_images.to(device)
            
            # Get model predictions
            outputs = model(caliper_images)
            
            # Move to CPU and convert to numpy for metric calculation
            output_np = outputs.squeeze(0).cpu().numpy() * 0.5 + 0.5  # Denormalize
            output_np = np.clip(output_np, 0, 1)
            
            gt_np = ground_truth_images.squeeze(0).cpu().numpy() * 0.5 + 0.5  # Denormalize
            gt_np = np.clip(gt_np, 0, 1)
            
            # Calculate metrics (convert to range [0, 255] for standard calculation)
            output_255 = (output_np[0] * 255).astype(np.uint8)
            gt_255 = (gt_np[0] * 255).astype(np.uint8)
            
            # Calculate PSNR
            curr_psnr = psnr(gt_255, output_255, data_range=255)
            
            # Calculate SSIM
            curr_ssim = ssim(gt_255, output_255, data_range=255)
            
            total_psnr += curr_psnr
            total_ssim += curr_ssim
            count += 1
    
    # Calculate averages
    average_psnr = total_psnr / count if count > 0 else 0
    average_ssim = total_ssim / count if count > 0 else 0
    
    print(f"Evaluation Results:")
    print(f"Average PSNR: {average_psnr:.2f} dB")
    print(f"Average SSIM: {average_ssim:.4f}")
    
    return average_psnr, average_ssim

def process_image_directory(model, input_dir, output_dir):
    """
    Process all images in a directory and save caliper-removed versions.
    
    Args:
        model: Trained U-Net model
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
    """
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Process each image
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            original_size = image.size  # (width, height)
            
            # Preprocess
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            image_tensor = normalize(to_tensor(image)).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(image_tensor.to(device))
            
            # Post-process - Fix the dimension issue
            output = output.squeeze(0).cpu()
            
            # Denormalize
            output = output * 0.5 + 0.5
            
            # Clamp values to [0, 1] range
            output = torch.clamp(output, 0, 1)
            
            # Convert to PIL Image
            output_img = transforms.ToPILImage()(output)
            
            # Resize to original size (PIL format: width, height)
            output_img = output_img.resize(original_size, Image.BILINEAR)
            
            # Save processed image
            input_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(input_filename)
            output_filename = f"{name}_caliper_removed{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            output_img.save(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"Processing complete! Output images saved to {output_dir}")
    
    
if __name__ == "__main__":
    data_dir = "D:/DATA/CASBUSI/PairExport"
    csv_file = os.path.join(data_dir, "PairData.csv")
    img_dir = os.path.join(data_dir, "box_images")
    caliper_dir = "D:/DATA/CASBUSI/PairExport/unique_caliper_shapes"
    model_path = os.path.join(data_dir, "caliper_removal_unet_l1loss_N2N_7.pth")

    print("Select an option:")
    print("1. Train the model")
    print("2. Show validation examples")
    print("3. Evaluate PSNR and SSIM")
    print("4. Process image directory (simple)")
    
    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        train_model(csv_file, img_dir, caliper_dir, model_path)
    elif choice == '2':
        if os.path.exists(model_path):
            model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(model_path))
            show_validation_examples(model, data_dir, csv_file, img_dir)
        else:
            print("No pre-trained model found. Please train the model first.")
    elif choice == '3':
        if os.path.exists(model_path):
            model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(model_path))
            evaluate_metrics(model, data_dir, csv_file, img_dir)
        else:
            print("No pre-trained model found. Please train the model first.")
    elif choice == '4':
        if os.path.exists(model_path):
            model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(model_path))
            
            img_dir = os.path.join(data_dir, "box_images")
            output_dir = os.path.join(data_dir, "box_images")
            
            if not os.path.exists(img_dir):
                print("Input directory does not exist!")
            else:
                process_image_directory(model, img_dir, output_dir)
        else:
            print("No pre-trained model found. Please train the model first.")

    else:
        print("Invalid choice. Please run the script again and enter 1-4.")