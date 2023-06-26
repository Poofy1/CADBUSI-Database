from PIL import Image
import torch, os
from torch.utils.data import Dataset
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")


def load_image(image_path, image_size):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    return preprocess(image)
    
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
    
    
def find_calipers(images_dir, model_name, image_size=256):
    model = Net()
    model.load_state_dict(torch.load(f"{env}/models/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    results = []
    
    with torch.no_grad():
        for image_file in tqdm(image_files):
            image = load_image(os.path.join(images_dir, image_file), image_size)
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU

            has_calipers_pred = model(image)

            # Transfer predictions to CPU, remove batch dimension, convert to appropriate data type
            prediction = has_calipers_pred.view(-1).tolist()[0]
            results.append(True if prediction > 0.5 else False)     

    
    return results