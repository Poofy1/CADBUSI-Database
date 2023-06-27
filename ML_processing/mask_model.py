import os, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

class MyDataset(Dataset):
    def __init__(self, root_dir, max_width, max_height, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
        self.max_width = max_width
        self.max_height = max_height

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        img_before_pad = preprocess(image)
        padding = transforms.Pad((0, 0, self.max_width - img_before_pad.shape[-1], self.max_height - img_before_pad.shape[-2]))
        img_after_pad = padding(img_before_pad)
        return img_after_pad
    
    
    
def find_masks(images_dir, model_name, max_width, max_height, batch_size=4):
    # Load a pre-trained model for classification
    backbone = torchvision.models.squeezenet1_1(pretrained=True).features
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    num_classes = 3
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    model.load_state_dict(torch.load(f"{env}/models/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    # Data loader
    dataset = MyDataset(images_dir, max_width, max_height)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    class1_results = []
    class2_results = []

    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            output = model(images)

            for i in range(len(output)):
                pred_boxes = output[i]['boxes']
                pred_scores = output[i]['scores']
                pred_labels = output[i]['labels']

                threshold = 0.5
                try:
                    mask = pred_scores > threshold
                    pred_boxes = pred_boxes[mask]
                    pred_scores = pred_scores[mask]
                    pred_labels = pred_labels[mask]

                    class1_boxes = pred_boxes[pred_labels == 1].cpu().numpy().astype(int)
                    class2_boxes = pred_boxes[pred_labels == 2].cpu().numpy().astype(int)
                except:
                    print("image failed to find correct data")
                    class1_boxes = None
                    class2_boxes = None

                class1_results.append(class1_boxes)
                class2_results.append(class2_boxes)

    class1_results = [arr.tolist() if arr is not None else [] for arr in class1_results]
    class2_results = [arr.tolist() if arr is not None else [] for arr in class2_results]

    return class1_results, class2_results
