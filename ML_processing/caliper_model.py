from PIL import Image
import torch, os
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([img for img in os.listdir(root_dir) if img in db_to_process['ImageName'].values])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
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


def find_calipers(images_dir, model_name, db_to_process, image_size=256, batch_size=4):
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
    dataset = MyDataset(images_dir, db_to_process, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    results = []
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, total=len(dataloader)):  # Unpack filenames here
            images = images.to(device)
            has_calipers_pred = model(images)
            prediction = (has_calipers_pred > 0.5).cpu() 
            # Pair each filename with its corresponding prediction
            result_pairs = list(zip(filenames, prediction.view(-1).tolist()))
            results.extend(result_pairs)  # Extend results with pairs

    return results