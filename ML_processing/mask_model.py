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
        self.images = sorted(os.listdir(root_dir))
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
        return img_after_pad, self.images[idx] 
    
    
    
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
        for images, filenames in tqdm(dataloader):  # Unpack filenames here
            images = images.to(device)
            output = model(images)

            for i in range(len(output)):
                pred_boxes = output[i]['boxes']
                pred_scores = output[i]['scores']
                pred_labels = output[i]['labels']

                best_boxes = []
                for class_id in range(1, 3):  # Assuming classes are 1, 2, and 3
                    class_mask = pred_labels == class_id
                    class_scores = pred_scores[class_mask]

                    if len(class_scores) > 0:
                        best_score_index = class_scores.argmax()
                        best_box = pred_boxes[class_mask][best_score_index]
                        best_boxes.append(best_box.cpu().numpy().astype(int))
                    else:
                        best_boxes.append(None)

                # Pair each filename with its corresponding results
                filename = filenames[i]
                class1_results.append((filename, best_boxes[0].tolist() if best_boxes[0] is not None else []))
                class2_results.append((filename, best_boxes[1].tolist() if best_boxes[1] is not None else []))

    return class1_results, class2_results
