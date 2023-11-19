import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))



def compile_images(csv_file, output_dir, images_per_row=4):
    # Load CSV file
    failed_cases = pd.read_csv(csv_file)
    df = pd.read_csv(f"{env}/database/ImageData.csv")
    image_dir = f"{env}/database/images/"

    # Additional setup for tracking image placements
    image_placement_tracker = []

    # Filter df for rows where 'label' is True
    df_label_true = df[df['label'] == True]

    # Group by 'Accession_Number' and count
    acc_num_counts = df_label_true.groupby('Accession_Number').size()

    # Filter 'failed_cases' for Accession Numbers with 12 or fewer images
    failed_cases_filtered = failed_cases[failed_cases['Accession_Number'].isin(acc_num_counts[acc_num_counts <= 12].index)]

    # Sort 'failed_cases_filtered' by 'Loss' in descending order and select the top 20%
    sorted_failed_cases = failed_cases_filtered.sort_values(by='Loss', ascending=False)
    top_20_percent = sorted_failed_cases.head(int(len(sorted_failed_cases) * 0.20))

    # Iterate over each Accession_Number
    for acc_num in tqdm(top_20_percent['Accession_Number']):
        
        true_label_images = df_label_true[df_label_true['Accession_Number'] == acc_num]

        # Load images
        loaded_images = [Image.open(os.path.join(image_dir, img_name)) for img_name in true_label_images['ImageName']]

        # Determine the size of the compiled image
        max_width = max(img.size[0] for img in loaded_images)
        max_height = max(img.size[1] for img in loaded_images)
        total_rows = (len(loaded_images) - 1) // images_per_row + 1
        panel_width = max_width * images_per_row
        panel_height = max_height * total_rows

        # Create a new blank image
        panel = Image.new('RGB', (panel_width, panel_height))
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype("arialbd.ttf", size=100)  # Adjust font size as needed

        # Paste images into the panel and add grid coordinates
        for i, img in enumerate(loaded_images):
            x = (i % images_per_row) * max_width
            y = (i // images_per_row) * max_height
            panel.paste(img, (x, y))

            # Calculate coordinates for text
            text_x = x + 50
            text_y = y + 50  # Slightly down from the top
            grid_coord = f"[{i // images_per_row},{i % images_per_row}]"
            draw.text((text_x, text_y), grid_coord, fill=(255, 0, 0), font=font)

            # Update image placement tracker
            image_placement_tracker.append({
                'Accession_Number': acc_num,
                'ImageName': true_label_images['ImageName'].iloc[i],
                'Placement': grid_coord
            })

        # Save the compiled image
        panel.save(os.path.join(output_dir, f'ACC_{int(acc_num)}.png'))

    # Save the image placement information to a CSV file
    pd.DataFrame(image_placement_tracker).to_csv(f'{env}/database/LossLabelingReferences.csv', index=False)

# Usage
output_dir = f"{env}/database/LossLabeling/"
os.makedirs(output_dir, exist_ok=True)
compile_images(f"{env}/failed_cases.csv", output_dir)