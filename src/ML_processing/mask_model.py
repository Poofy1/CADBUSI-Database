import os, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from storage_adapter import *
from torch.amp import autocast
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on A100/T4
torch.backends.cudnn.allow_tf32 = True

def get_first_image_in_each_folder(video_folder_path):
    first_images = []
    video_folder_path = os.path.normpath(video_folder_path)
    
    storage = StorageClient.get_instance()
    
    # For GCP: Get unique folder prefixes, then construct first image path
    prefix = video_folder_path.replace('\\', '/').rstrip('/') + '/'
    
    # Use delimiter to get folder structure
    iterator = storage._bucket.list_blobs(prefix=prefix, delimiter='/')
    blobs = list(iterator)  # Get the blobs
    prefixes = iterator.prefixes  # Get the folder prefixes
    
    # For each folder prefix, construct the first image path
    for folder_prefix in prefixes:
        # Extract folder name from prefix (remove trailing slash)
        folder_name = folder_prefix.rstrip('/').split('/')[-1]
        
        # Construct first image path using naming convention
        first_image_path = f"{folder_name}/{folder_name}_0.png"
        first_images.append(first_image_path)

    return first_images

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, max_width, max_height, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all files from the directory
        all_files = list_files(root_dir)

        # Extract just the filenames from the full paths
        file_dict = {os.path.basename(img): img for img in all_files}

        # Filter by the database image names and store only the filenames (not full paths)
        self.images = sorted([os.path.basename(file_dict[img_name]) for img_name in db_to_process['ImageName'].values 
                            if img_name in file_dict])
        
        self.max_width = max_width
        self.max_height = max_height
        
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = read_image(img_name, use_pil=True)
        
        img_before_pad = self.preprocess(image)
        padding = transforms.Pad((0, 0, self.max_width - img_before_pad.shape[-1], self.max_height - img_before_pad.shape[-2]))
        img_after_pad = padding(img_before_pad)
        return img_after_pad, self.images[idx] 
    
class MyDatasetVideo(Dataset):
    def __init__(self, root_dir, db_to_process, max_width, max_height, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        all_first_images = get_first_image_in_each_folder(root_dir) 
        # Filter to only include images from db_to_process
        images_to_process = set(db_to_process['ImagesPath'].tolist())
        
        # Extract folder name from image path (before the "/")
        self.images = [img for img in all_first_images 
                       if img.split('/')[0] in images_to_process]
        
        self.max_width = max_width
        self.max_height = max_height
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = read_image(img_name, use_pil=True)
        
        img_before_pad = self.preprocess(image)
        padding = transforms.Pad((0, 0, self.max_width - img_before_pad.shape[-1], self.max_height - img_before_pad.shape[-2]))
        img_after_pad = padding(img_before_pad)
        return img_after_pad, self.images[idx] 
    
    
    
def find_masks(images_dir, model_name, db_to_process, max_width, max_height, video_format=False, video_folders=None):
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
    if video_format:
        dataset = MyDatasetVideo(images_dir, db_to_process, max_width, max_height)
    else:
        dataset = MyDataset(images_dir, db_to_process, max_width, max_height)
        
    dataloader = DataLoader(
        dataset, 
        batch_size=64,
        num_workers=8,
        pin_memory=True,
    )

    class1_results = []
    class2_results = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, total=len(dataloader)):  # Unpack filenames here
            images = images.to(device, non_blocking=True)
            with autocast('cuda'):
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
