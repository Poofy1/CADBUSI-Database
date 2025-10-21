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
        self.db_to_process = db_to_process
        
        # Get all files from the directory
        all_files = list_files(root_dir)

        # Extract just the filenames from the full paths
        file_dict = {os.path.basename(img): img for img in all_files}

        # Filter by the database image names and store only the filenames (not full paths)
        self.images = sorted([os.path.basename(file_dict[img_name]) for img_name in db_to_process['image_name'].values 
                            if img_name in file_dict])
        
        # Create a mapping from image_name to crop parameters
        self.crop_info = {}
        for idx, row in db_to_process.iterrows():
            img_name = row['image_name']
            if img_name in file_dict:
                self.crop_info[os.path.basename(img_name)] = {
                    'crop_x': row.get('crop_x', None),
                    'crop_y': row.get('crop_y', None),
                    'crop_w': row.get('crop_w', None),
                    'crop_h': row.get('crop_h', None)
                }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_name = os.path.join(self.root_dir, img_filename)
        image = read_image(img_name, use_pil=True)
        
        # Apply crop if crop parameters are available
        crop_params = self.crop_info.get(img_filename, {})
        crop_x = crop_params.get('crop_x', None)
        crop_y = crop_params.get('crop_y', None)
        crop_w = crop_params.get('crop_w', None)
        crop_h = crop_params.get('crop_h', None)
        
        if crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None:
            # Convert to integers in case they're floats
            crop_x, crop_y, crop_w, crop_h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)
            # PIL crop uses (left, upper, right, lower)
            image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
        if self.transform:
            image = self.transform(image)
        return image, img_filename
    
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


def find_calipers(images_dir, db_to_process, image_size=256):
    # Separate RGB images from non-RGB images
    rgb_images = db_to_process[db_to_process['photometric_interpretation'] == 'RGB'].copy()
    non_rgb_images = db_to_process[db_to_process['photometric_interpretation'] != 'RGB'].copy()
    
    results = []
    
    # Add RGB images with False predictions
    for img_name in rgb_images['image_name'].values:
        results.append((img_name, False, -1))
    
    # Only process non-RGB images if there are any
    if len(non_rgb_images) > 0:
        model = Net()
        model.load_state_dict(torch.load(f"{env}/models/caliper_detect_10_7_25.pt"))
        model = model.to(device)
        model.eval()
        
        # Data loader - note: crop will be applied in MyDataset before transforms
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale first
            transforms.Resize((image_size, image_size)),   # Then resize
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MyDataset(images_dir, non_rgb_images, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, total=len(dataloader), desc="Finding Caliper Images"):
                images = images.to(device)
                with autocast('cuda'):
                    has_calipers_pred = model(images)
                raw_predictions = has_calipers_pred.cpu().view(-1).tolist()
                boolean_predictions = (has_calipers_pred > 0.4).cpu().view(-1).tolist()
                
                # Pair each filename with both its boolean prediction and raw prediction value
                result_triplets = list(zip(filenames, boolean_predictions, raw_predictions))
                results.extend(result_triplets)

    return results