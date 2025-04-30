import pandas as pd
from PIL import Image
import torch, os, ast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from training_util import *
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")


def load_image(image_path, max_width, max_height):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.35, 1.15), saturation=(0, 1.5), hue=(-0.1, 0.1)),
        transforms.ToTensor(),
        StaticNoise(intensity_min=0.0, intensity_max=0.03),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    
    
    img_before_pad = preprocess(image)

    # Now let's do the padding
    padding = transforms.Pad((0, 0, max_width - img_before_pad.shape[-1], max_height - img_before_pad.shape[-2]))
    
    img_after_pad = padding(img_before_pad)
    return img_after_pad

 
class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, max_width, max_height):
        self.df = dataframe
        self.img_dir = img_dir
        self.max_width = max_width
        self.max_height = max_height
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id = row['patient_id']
        img = load_image(f"{self.img_dir}/{img_id}", self.max_width, self.max_height)
        
        
        """img_disp = img.clone().detach()  
        img_disp = img_disp * 0.5 + 0.5
        if img_disp.shape[0] == 1:
            img_disp = img_disp.squeeze(0)
        plt.imshow(img_disp.numpy(), cmap='gray')
        plt.show()"""
        
        # Assume the two masks are bounding boxes in format [x1, y1, x2, y2]
        boxes = []
        for box in [row['image_mask'], row['description_mask']]:
            if pd.isnull(box): # check if the field is not empty
                continue
            box = ast.literal_eval(box) if isinstance(box, str) else box
            boxes.append(box)

        labels = torch.tensor([1, 2], dtype=torch.int64)[:len(boxes)] # adjust labels to the actual number of boxes
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = labels

        return (img, target)


    
    
    
    
    
# Load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.squeezenet1_1(pretrained=True).features

# FasterRCNN needs to know the number of output channels in a backbone. For squeezenet1_1, it's 512
# so we need to add it here
backbone.out_channels = 512

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios 
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# FasterRCNN also needs to know the number of output channels
# (i.e., classes) plus one for the background.
# For your case, it's 3 (object 1, object 3) + 1 (background) = 4
num_classes = 3

# Create the Faster RCNN model
model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator)
    
    
    

    

def test_images(model, images_dir, model_name, max_width, max_height):
    model.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    colors = ['blue', 'red', 'green']

    with torch.no_grad():
        for image_file in image_files:
            fig = plt.figure(figsize=(5, 5))
            gs = gridspec.GridSpec(1, 1, figure=fig)

            image = load_image(os.path.join(images_dir, image_file), max_width, max_height)

            # Move the input image to the GPU and perform the forward pass
            image = image.to(device).unsqueeze(0)
            output = model(image)

            # Extract predicted bounding boxes, scores, and labels
            pred_boxes = output[0]['boxes']
            pred_scores = output[0]['scores']
            pred_labels = output[0]['labels']

            best_crops = []  # Store the best crop for each class

            # Iterate through each unique class
            unique_labels = torch.unique(pred_labels)
            for label in unique_labels:
                # Find the highest scoring prediction for the current class
                class_indices = torch.where(pred_labels == label)[0]
                class_scores = pred_scores[class_indices]
                best_index = class_indices[class_scores.argmax()]
                best_crop = pred_boxes[best_index]
                best_crops.append(best_crop)

            # Convert the image back to the original format
            img_disp = image.cpu().numpy() * 0.5 + 0.5
            img_disp = img_disp.squeeze()

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(img_disp, cmap='gray')

            # Add the predicted bounding boxes with different colors
            for j, box in enumerate(best_crops):
                x1, y1, x2, y2 = box.cpu().numpy()
                color = colors[j % len(colors)]  # Cycle through the colors
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none'))

            # Show the current image
            plt.show()











def train_model(model, train_loader, val_loader, num_epochs, model_name):
    print("Training..")
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for I, data in enumerate(train_loader):
            img, targets = data
            img = img.to(device)
        
            batched_boxes = targets["boxes"].to(device)
            batched_labels = targets["labels"].to(device)

            batch_targets = []
            for bidx in range(img.size(0)):
                target_dict = {
                    "boxes": batched_boxes[bidx],
                    "labels": batched_labels[bidx],
                }
                batch_targets.append(target_dict)
            
            optimizer.zero_grad()
            loss_dict = model(img, batch_targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()
            losses.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.5f}")
        
    torch.save(model.state_dict(), f"{env}/model/{model_name}.pt")





if __name__ == "__main__":
    csv_path = f"{env}/dataset/labeled_data.csv"
    img_dir = f"{env}/dataset/images"

    # load data
    df = pd.read_csv(csv_path)
    
    # Get the maximum dimensions
    max_width, max_height = find_max_dimensions(df)
    print(f"Max dimensions: {max_width}, {max_height}")
    
    batch_size = 16
    model_name = "mask_model"
    num_epochs = 15

    train_df, val_df = train_test_split(df, test_size=0.025, random_state=42)
    
    # create dataset and dataloader
    train_dataset = CustomDataset(train_df.reset_index(drop=True), img_dir, max_width, max_height)
    val_dataset = CustomDataset(val_df.reset_index(drop=True), img_dir, max_width, max_height)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize the model and move it to the GPU
    model = model.to(device)
    
    # Load the model if it exists
    model_path = f"{env}/model/{model_name}.pt"
    model = load_model(model, model_path)

    if True:
        summary(model, input_size=(batch_size, 1, max_width, max_height))
        train_model(model, train_loader, val_loader, num_epochs, model_name)




    images_dir = f"{env}/test_images/"
    if True:
        test_images(model, images_dir, model_name, max_width, max_height)
