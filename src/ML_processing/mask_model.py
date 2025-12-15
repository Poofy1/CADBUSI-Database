import os, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from tools.storage_adapter import *
from torch.amp import autocast

env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_first_image_in_each_folder(video_folder_path):
    first_images = []
    video_folder_path = os.path.normpath(video_folder_path)
    
    storage = StorageClient.get_instance()
    
    prefix = video_folder_path.replace('\\', '/').rstrip('/') + '/'
    
    iterator = storage._bucket.list_blobs(prefix=prefix, delimiter='/')
    blobs = list(iterator)
    prefixes = iterator.prefixes
    
    for folder_prefix in prefixes:
        folder_name = folder_prefix.rstrip('/').split('/')[-1]
        first_image_path = f"{folder_name}/{folder_name}_0.png"
        first_images.append(first_image_path)

    return first_images

class MyDataset(Dataset):
    def __init__(self, root_dir, db_to_process, max_width, max_height, transform=None):
        """
        Args:
            root_dir: Root directory for images
            db_to_process: DataFrame with image_name column (already renamed from image_name)
            max_width: Maximum width for padding
            max_height: Maximum height for padding
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all files from the directory
        all_files = list_files(root_dir)
        file_dict = {os.path.basename(img): img for img in all_files}
        
        # Filter by the database image names
        self.images = sorted([os.path.basename(file_dict[img_name]) 
                            for img_name in db_to_process['image_name'].values 
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
        padding = transforms.Pad((0, 0, 
                                 self.max_width - img_before_pad.shape[-1], 
                                 self.max_height - img_before_pad.shape[-2]))
        img_after_pad = padding(img_before_pad)
        return img_after_pad, self.images[idx]

class MyDatasetVideo(Dataset):
    def __init__(self, root_dir, db_to_process, max_width, max_height, transform=None):
        """
        Args:
            root_dir: Root directory for video frames
            db_to_process: DataFrame with images_path column (already renamed from images_path)
            max_width: Maximum width for padding
            max_height: Maximum height for padding
        """
        self.root_dir = root_dir
        self.transform = transform
        
        all_first_images = get_first_image_in_each_folder(root_dir)
        
        # Filter to only include images from db_to_process
        images_to_process = set(db_to_process['images_path'].tolist())
        
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
        padding = transforms.Pad((0, 0, 
                                 self.max_width - img_before_pad.shape[-1], 
                                 self.max_height - img_before_pad.shape[-2]))
        img_after_pad = padding(img_before_pad)
        return img_after_pad, self.images[idx]

def find_masks(images_dir, model_name, db_to_process, max_width, max_height, 
               video_format=False):
    """
    Find masks/bounding boxes in images using object detection model.
    
    Args:
        images_dir: Directory containing images
        model_name: Name of the model file (without .pt extension)
        db_to_process: DataFrame with image data (with renamed columns: image_name or images_path)
        max_width: Maximum width for padding
        max_height: Maximum height for padding
        video_format: Whether processing video frames
    
    Returns:
        Tuple of (class1_results, class2_results) - lists of (filename, bbox) tuples
    """
    # Load model
    backbone = torchvision.models.squeezenet1_1(pretrained=True).features
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), 
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    num_classes = 3
    model = FasterRCNN(backbone, num_classes=num_classes, 
                      rpn_anchor_generator=anchor_generator)

    model.load_state_dict(torch.load(f"{env}/models/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
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

    # Run inference
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, total=len(dataloader), desc='Finding OCR Masks'):
            images = images.to(device, non_blocking=True)
            with autocast('cuda'):
                output = model(images)

            for i in range(len(output)):
                pred_boxes = output[i]['boxes']
                pred_scores = output[i]['scores']
                pred_labels = output[i]['labels']

                best_boxes = []
                for class_id in range(1, 3):
                    class_mask = pred_labels == class_id
                    class_scores = pred_scores[class_mask]

                    if len(class_scores) > 0:
                        best_score_index = class_scores.argmax()
                        best_box = pred_boxes[class_mask][best_score_index]
                        best_boxes.append(best_box.cpu().numpy().astype(int))
                    else:
                        best_boxes.append(None)

                filename = filenames[i]
                class1_box = best_boxes[0].tolist() if best_boxes[0] is not None else []
                class2_box = best_boxes[1].tolist() if best_boxes[1] is not None else []
                
                class1_results.append((filename, class1_box))
                class2_results.append((filename, class2_box))

    return class1_results, class2_results