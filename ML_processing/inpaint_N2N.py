import torch
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
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
    
    model_path = f'{env}/models/N2N_5.pth'
    model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for index, row in tqdm(processed_data.iterrows(), total=len(processed_data)):
            # Load and process input image
            input_image_path = os.path.join(input_folder, row['ImageName'])
            original_image = read_image(input_image_path, use_pil=True).convert('L')
            
            # Preprocess image
            image_tensor = normalize(to_tensor(original_image)).unsqueeze(0)
            image_tensor = image_tensor.to(DEVICE)
            
            # Generate inpainted image
            output = model(image_tensor)
            
            # Post-process output
            output = output.squeeze(0).cpu()
            output = output * 0.5 + 0.5  # Denormalize
            output = torch.clamp(output, 0, 1)
            output_image = transforms.ToPILImage()(output)
            
            # Save inpainted image
            os.remove(input_image_path)
            save_data(output_image, input_image_path)
            
            # Update CSV
            data.loc[index, 'Inpainted'] = True
    
    # Save updated CSV
    save_data(data, csv_file_path)
