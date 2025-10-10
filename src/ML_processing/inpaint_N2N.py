import torch
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from training.train_twin_N2N import N2N_Original_Used_UNet
from storage_adapter import *
from src.DB_processing.database import DatabaseManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
env = os.path.dirname(os.path.abspath(__file__))

def Inpaint_Dataset_N2N(input_folder):
    print("Inpainting and Evaluating Caliper Images")
    
    with DatabaseManager() as db:
        # Load image data from database
        data = db.get_images_dataframe()
        
        # Add 'inpainted_from' column if not present
        if 'inpainted_from' not in data.columns:
            data['inpainted_from'] = None
        
        # Filter the data - only process rows that haven't been inpainted yet
        processed_data = data[
            (data['distance'] > 5) & # no clean duplicate available 
            ((data['has_calipers'] == True) | (data['photometric_interpretation'] == 'RGB')) & 
            (data['inpainted_from'].isna())
        ]
        
        # Prepare transforms
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        model_path = f'{env}/models/N2N_7.pth'
        model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        model.eval()
        
        # List to store new rows to be added
        new_rows = []
        
        with torch.no_grad():
            for index, row in tqdm(processed_data.iterrows(), total=len(processed_data)):
                # Load and process input image
                input_image_path = os.path.join(input_folder, row['image_name'])
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
                
                # Create new filename with _inpainted suffix
                original_filename = row['image_name']
                name, ext = os.path.splitext(original_filename)
                new_filename = f"{name}_inpainted{ext}"
                new_image_path = os.path.join(input_folder, new_filename)
                new_image_path = os.path.normpath(new_image_path)
                
                # Save inpainted image with new filename (keep original)
                save_data(final_image, new_image_path)
                
                # Create a copy of the current row for the inpainted version
                new_row = row.to_dict()
                new_row['image_name'] = new_filename
                new_row['inpainted_from'] = original_filename
                new_row['has_calipers'] = 0
                new_rows.append(new_row)
        
        # Insert new rows into database using batch insert
        if new_rows:
            rows_inserted = db.insert_images_batch(new_rows)
            print(f"Added {rows_inserted} inpainted images to database")