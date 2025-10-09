import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from textwrap import wrap
from src.DB_processing.database import DatabaseManager


def Create_Labelbox_Data(failed_cases_csv, database_path, images_per_row=4):
    output_dir = f"{database_path}/LossLabeling/"
    os.makedirs(output_dir, exist_ok=True)
    
    with DatabaseManager() as db:
        # Load failed cases CSV (external input)
        failed_cases = pd.read_csv(failed_cases_csv)
        
        # Load data from database
        df = db.get_images_dataframe()
        case_df = db.get_study_cases_dataframe()
        image_dir = f"{database_path}/images/"

        # Additional setup for tracking image placements
        image_placement_tracker = []

        # Filter df for rows where 'is_labeled' is True
        df_label_true = df[df['is_labeled'] == 1]

        # Group by 'accession_number' and count
        acc_num_counts = df_label_true.groupby('accession_number').size()

        # Filter failed_cases as before
        failed_cases_filtered = failed_cases[failed_cases['Accession_Number'].isin(acc_num_counts[acc_num_counts <= 12].index)]

        # Sort and select the top 20%
        sorted_failed_cases = failed_cases_filtered.sort_values(by='Loss', ascending=False)
        top_20_percent = sorted_failed_cases.head(int(len(sorted_failed_cases) * 0.20))

        # Merge case_df with failed_cases to get findings/synoptic_report
        # Note: Adjust column name based on what you want to display
        case_df_for_merge = case_df[['accession_number', 'findings']].rename(columns={
            'accession_number': 'Accession_Number',
            'findings': 'Path_Desc'
        })
        merged_cases = pd.merge(left=failed_cases, right=case_df_for_merge, on='Accession_Number')

        for acc_num in tqdm(top_20_percent['Accession_Number']):
            true_label_images = df_label_true[df_label_true['accession_number'] == acc_num]

            # Extract case information, including Path_Desc
            case_info = merged_cases[merged_cases['Accession_Number'] == acc_num]
            prediction_label = "Malignant" if case_info['Prediction'].iloc[0] > 0.5 else "Benign"
            true_label = "Malignant" if case_info['True_Label'].iloc[0] > 0.5 else "Benign"
            path_desc = case_info['Path_Desc'].iloc[0]

            first_line = f"Prediction: {prediction_label} | True Label: {true_label}"
            path_desc_lines = wrap(path_desc, width=100)

            # Load images
            loaded_images = [Image.open(os.path.join(image_dir, img_name)) for img_name in true_label_images['image_name']]

            # Determine the size of the compiled image
            max_width = max(img.size[0] for img in loaded_images)
            max_height = max(img.size[1] for img in loaded_images)
            total_rows = (len(loaded_images) - 1) // images_per_row + 1
            panel_width = max_width * images_per_row
            panel_height = max_height * total_rows

            # Create a new blank image
            panel = Image.new('RGB', (panel_width, panel_height))
            draw = ImageDraw.Draw(panel)
            font = ImageFont.truetype("arialbd.ttf", size=60)

            # Calculate header height
            line_height = font.getsize('A')[1]
            header_height = 50 + line_height * (1 + len(path_desc_lines))

            # Adjust panel height to include header
            panel_height += header_height

            # Redraw the image with the new header height
            panel = Image.new('RGB', (panel_width, panel_height))
            draw = ImageDraw.Draw(panel)

            # Draw the first line of the header
            draw.rectangle([0, 0, panel_width, header_height], fill=(0, 0, 0))
            y_text = 10
            draw.text((10, y_text), first_line, fill=(255, 255, 255), font=font)
            y_text += line_height + 10

            # Draw the wrapped Path_Desc starting on a new line
            for line in path_desc_lines:
                draw.text((10, y_text), line, fill=(255, 255, 255), font=font)
                y_text += line_height

            # Paste images into the panel and add grid coordinates
            for i, img in enumerate(loaded_images):
                x = (i % images_per_row) * max_width
                y = (i // images_per_row) * max_height + header_height
                panel.paste(img, (x, y))

                # Calculate coordinates for text
                text_x = x + 50
                text_y = y + 50
                grid_coord = f"[{i // images_per_row},{i % images_per_row}]"
                draw.text((text_x, text_y), grid_coord, fill=(255, 0, 0), font=font)

                # Update image placement tracker
                image_placement_tracker.append({
                    'Accession_Number': acc_num,
                    'ImageName': true_label_images['image_name'].iloc[i],
                    'Placement': grid_coord
                })

            # Save the compiled image
            panel.save(os.path.join(output_dir, f'ACC_{int(acc_num)}.png'))

        # Save the image placement information to a CSV file
        pd.DataFrame(image_placement_tracker).to_csv(f'{database_path}/LossLabelingReferences.csv', index=False)