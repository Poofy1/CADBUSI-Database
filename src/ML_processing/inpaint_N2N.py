import torch
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from training.train_twin_N2N import N2N_Original_Used_UNet
from storage_adapter import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
env = os.path.dirname(os.path.abspath(__file__))

def Inpaint_Dataset_N2N(csv_file_path, input_folder):
    print("Inpainting and Evaluating Caliper Images")
    
    # Load the CSV file
    data = read_csv(csv_file_path)
    
    # Add 'Inpainted' column if not present
    if 'Inpainted' not in data.columns:
        data['Inpainted'] = False
    else:
        data['Inpainted'] = data['Inpainted'].where(data['Inpainted'], False)
    
    # Filter the data
    processed_data = data[
        (data['label'] == True) & 
        (data['has_calipers'] == True) & 
        (data['Inpainted'] == False)
    ]
    
    # Prepare transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    model_path = f'{env}/models/N2N_7.pth'
    model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        for index, row in tqdm(processed_data.iterrows(), total=len(processed_data)):
            # Load and process input image
            input_image_path = os.path.join(input_folder, row['ImageName'])
            input_image_path = os.path.normpath(input_image_path)
            
            # Load original image and determine if RGB
            original_image = read_image(input_image_path, use_pil=True)
            original_size = original_image.size
            is_rgb = original_image.mode == 'RGB'
            
            # Convert to grayscale for model processing
            grayscale_image = original_image.convert('L')
            
            # Preprocess for model
            image_tensor = normalize(to_tensor(grayscale_image)).unsqueeze(0)
            image_tensor = image_tensor.to(DEVICE)
            
            # Generate inpainted image
            output = model(image_tensor)
            
            # Post-process output
            output = output.squeeze(0).cpu()
            output = output * 0.5 + 0.5  # Denormalize
            output = torch.clamp(output, 0, 1)
            
            # Convert to PIL Image
            processed_grayscale = transforms.ToPILImage()(output)
            processed_grayscale = processed_grayscale.resize(original_size, Image.BILINEAR)
            
            # Handle RGB vs Grayscale differently
            if is_rgb:
                # Convert to numpy arrays
                original_gray_np = np.array(grayscale_image, dtype=np.float32) / 255.0
                processed_gray_np = np.array(processed_grayscale, dtype=np.float32) / 255.0
                original_rgb_np = np.array(original_image, dtype=np.float32) / 255.0
                
                # Create mask of significantly changed areas
                difference = np.abs(original_gray_np - processed_gray_np)
                threshold = 0.05  # Adjust this threshold as needed
                replacement_mask = difference > threshold
                
                # Start with original RGB image
                result_rgb = original_rgb_np.copy()
                
                # For pixels that changed significantly, replace ALL color channels 
                # with the processed grayscale value
                if np.any(replacement_mask):
                    grayscale_rgb_value = processed_gray_np[replacement_mask]
                    # Set R, G, B all to the same grayscale value
                    result_rgb[replacement_mask, 0] = grayscale_rgb_value  # R
                    result_rgb[replacement_mask, 1] = grayscale_rgb_value  # G  
                    result_rgb[replacement_mask, 2] = grayscale_rgb_value  # B
                
                # Convert back to PIL Image
                result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)
                final_image = Image.fromarray(result_rgb, mode='RGB')
            else:
                # For grayscale images, use processed result directly
                final_image = processed_grayscale
            
            # Save inpainted image
            delete_file(input_image_path)
            save_data(final_image, input_image_path)
            
            # Update CSV
            data.loc[index, 'Inpainted'] = True
    
    # Save updated CSV
    save_data(data, csv_file_path)