from PIL import Image
import torch, os
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from storage_adapter import *
from torch.utils.data import Dataset
from torch.amp import autocast
warnings.filterwarnings('ignore')
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on A100/T4
torch.backends.cudnn.allow_tf32 = True

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all files from the directory
        all_files = list_files(root_dir)

        # Extract just the filenames from the full paths
        file_dict = {os.path.basename(img): img for img in all_files}

        # Filter by the database image names and store only the filenames (not full paths)
        self.images = sorted([os.path.basename(file_dict[img_name]) for img_name in db_to_process['ImageName'].values 
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


def find_calipers(images_dir, model_name, db_to_process, image_size=256):
    # Separate RGB images from non-RGB images
    rgb_images = db_to_process[db_to_process['PhotometricInterpretation'] == 'RGB'].copy()
    non_rgb_images = db_to_process[db_to_process['PhotometricInterpretation'] != 'RGB'].copy()
    
    results = []
    
    # Add RGB images with False predictions
    for img_name in rgb_images['ImageName'].values:
        results.append((img_name, False, -1))
    
    # Only process non-RGB images if there are any
    if len(non_rgb_images) > 0:
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
        dataset = MyDataset(images_dir, non_rgb_images, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, total=len(dataloader), desc="Processing non-RGB images"):
                images = images.to(device)
                with autocast('cuda'):
                    has_calipers_pred = model(images)
                raw_predictions = has_calipers_pred.cpu().view(-1).tolist()
                boolean_predictions = (has_calipers_pred > 0.5).cpu().view(-1).tolist()
                
                # Pair each filename with both its boolean prediction and raw prediction value
                result_triplets = list(zip(filenames, boolean_predictions, raw_predictions))
                results.extend(result_triplets)

    return results