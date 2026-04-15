import torch
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from training.train_twin_N2N import N2N_Original_Used_UNet
from tools.storage_adapter import *
from src.DB_processing.database import DatabaseManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
env = os.path.dirname(os.path.abspath(__file__))


def process_single_image(args):
    """Process a single image: load, inference, save to inpainted/ folder."""
    row_dict, input_folder, inpainted_folder, model, device, to_tensor, normalize = args

    try:
        # Load image
        input_image_path = os.path.join(input_folder, row_dict['image_name'])
        input_image_path = os.path.normpath(input_image_path)

        original_image = read_image(input_image_path, use_pil=True)
        original_size = original_image.size
        is_rgb = original_image.mode == 'RGB'

        # Convert to grayscale for model
        grayscale_image = original_image.convert('L')

        # Preprocess for model
        image_tensor = normalize(to_tensor(grayscale_image)).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Generate inpainted image
        with torch.no_grad():
            output = model(image_tensor)

        # Post-process output
        output = output.squeeze(0).cpu()
        output = output * 0.5 + 0.5  # Denormalize
        output = torch.clamp(output, 0, 1)

        # Convert to PIL Image
        processed_grayscale = transforms.ToPILImage()(output)
        processed_grayscale = processed_grayscale.resize(original_size, Image.BILINEAR)

        # Handle RGB vs Grayscale
        if is_rgb:
            original_gray_np = np.array(grayscale_image, dtype=np.float32) / 255.0
            processed_gray_np = np.array(processed_grayscale, dtype=np.float32) / 255.0
            original_rgb_np = np.array(original_image, dtype=np.float32) / 255.0

            difference = np.abs(original_gray_np - processed_gray_np)
            replacement_mask = difference > 0.05

            result_rgb = original_rgb_np.copy()
            if np.any(replacement_mask):
                grayscale_rgb_value = processed_gray_np[replacement_mask]
                result_rgb[replacement_mask, 0] = grayscale_rgb_value
                result_rgb[replacement_mask, 1] = grayscale_rgb_value
                result_rgb[replacement_mask, 2] = grayscale_rgb_value

            result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)
            final_image = Image.fromarray(result_rgb, mode='RGB')
        else:
            final_image = processed_grayscale

        # Create new filename and save to inpainted/ folder
        original_filename = row_dict['image_name']
        name, ext = os.path.splitext(original_filename)
        new_filename = f"{name}_inpainted{ext}"
        new_image_path = os.path.join(inpainted_folder, new_filename)
        new_image_path = os.path.normpath(new_image_path)

        # Save inpainted image
        save_data(final_image, new_image_path)

        return {
            'image_name': original_filename,
            'inpainted_version': new_filename
        }

    except Exception as e:
        print(f"Error processing {row_dict['image_name']}: {e}")
        return None


def Inpaint_Dataset_N2N(input_folder, num_workers=8):

    # Derive inpainted/ folder as sibling of images/
    base_dir = os.path.dirname(input_folder.rstrip('/\\'))
    inpainted_folder = os.path.join(base_dir, 'inpainted')
    os.makedirs(inpainted_folder, exist_ok=True)

    with DatabaseManager() as db:
        # Load image data from database
        data = db.get_images_dataframe()

        # Filter the data
        processed_data = data[
            (data['distance'] > 5) &
            ((data['has_calipers'] == True))
        ]

        if len(processed_data) == 0:
            print("No images to process")
            return

        print(f"Processing {len(processed_data)} images with {num_workers} workers")

        # Prepare transforms
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        # Load model
        model_path = f'{env}/models/N2N_7.pth'
        model = N2N_Original_Used_UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        model.eval()

        # Share model across threads (thread-safe with eval() and no_grad())
        model.share_memory()  # Important for multi-threading with PyTorch

        # Prepare arguments for each image
        args_list = [
            (row.to_dict(), input_folder, inpainted_folder, model, DEVICE, to_tensor, normalize)
            for _, row in processed_data.iterrows()
        ]

        # Process in parallel
        new_rows = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_image, args_list),
                total=len(args_list),
                desc="Inpainting Images"
            ))
            new_rows = [r for r in results if r is not None]

        # Update original rows with inpainted_version and populate CaliperPairs
        if new_rows:
            rows_updated = db.insert_images_batch(new_rows, update_only=True)
            print(f"Updated {rows_updated} images with inpainted_version")

            caliper_pairs = [
                {
                    'caliper_image_name': row['image_name'],
                    'inpainted_image_name': row['inpainted_version']
                }
                for row in new_rows
            ]
            db.insert_caliper_pairs_batch(caliper_pairs)
            print(f"Saved {len(caliper_pairs)} caliper/inpainted pairs to CaliperPairs table")

