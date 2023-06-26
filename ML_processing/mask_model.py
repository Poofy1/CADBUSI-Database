import os, torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")





def load_image(image_path, max_width, max_height):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    
    
    img_before_pad = preprocess(image)

    # Now let's do the padding
    padding = transforms.Pad((0, 0, max_width - img_before_pad.shape[-1], max_height - img_before_pad.shape[-2]))
    
    img_after_pad = padding(img_before_pad)
    return img_after_pad
    
    
    
def find_masks(images_dir, model_name, max_width, max_height):
    # Load a pre-trained model for classification
    backbone = torchvision.models.squeezenet1_1(pretrained=True).features

    # FasterRCNN needs to know the number of output channels in a backbone. For squeezenet1_1, it's 512
    backbone.out_channels = 512

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))


    # (object 1, object 2) + 1 (background) = 3
    num_classes = 3

    # Create the Faster RCNN model
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)


    model.load_state_dict(torch.load(f"{env}/models/{model_name}.pt"))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    class1_results = []
    class2_results = []
    
    with torch.no_grad():
        for image_file in tqdm(image_files):
            image = load_image(os.path.join(images_dir, image_file), max_width, max_height)
            
            # Move the input image to the GPU and perform the forward pass
            image = image.to(device).unsqueeze(0)
            output = model(image)
            
            # Extract predicted bounding boxes and scores
            pred_boxes = output[0]['boxes']
            pred_scores = output[0]['scores']
            pred_labels = output[0]['labels']

            # Threshold the predictions based on the score
            threshold = 0.5
            try:
                pred_boxes = pred_boxes[pred_scores>threshold]
                pred_scores = pred_scores[pred_scores>threshold]
                pred_labels = pred_labels[pred_scores>threshold]
                
                # Separate boxes by class
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
