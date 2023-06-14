import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import csv
import tqdm




def Crop_and_save_images(csv_file_path, image_input_folder, output_csv, output_folder, images_per_row):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Filter rows where 'Area' column is 'breast'
    #data = data[data['area'].str.lower() == 'breast']
    
    # Exclude rows where 'size' column has any text
    #data = data[data['size'].isna()]
    
    data = data[data['label'] == True]

    # If there are no matching data, return
    if data.empty:
        print("No data for 'breast' area found.")
        return
    
    # Transform the 'orientation' column
    #data['orientation'] = data['orientation'].apply(lambda x: x if x in ['long', 'trans'] else 'other')
    
    # Create a new column to group the data
    #data['group'] = data.apply(lambda row: 'doppler' if row['PhotometricInterpretation'] == 'RGB' else row['orientation'], axis=1)
    data['group'] = data['label_cat']

    # Group the data by 'patient_id'
    grouped_patient = data.groupby('anonymized_accession_num')

    # Check if the output CSV file exists
    csv_exists = os.path.isfile(output_csv)

    # Open the output CSV file in append mode if it exists, otherwise in write mode
    with open(output_csv, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # If the CSV file doesn't exist, write the header row
        if not csv_exists:
            writer.writerow(['patient_id', 'group', 'image_filename', 'x', 'y', 'width', 'height', 'us_x0', 'us_y0', 'us_x1', 'us_y1'])
    
        for patient_id, patient_group in tqdm.tqdm(grouped_patient):
            images = []
            image_records = []
            total_height = 0
            total_width = 0 

            
            for group_val in ['long', 'trans', 'doppler', 'other']:
                group = patient_group[patient_group['group'] == group_val]
                group_images = []

                
                for index, row in group.iterrows():
                    image_path = os.path.join(image_input_folder, row['image_filename'])
                    if os.path.isfile(image_path):
                        img = Image.open(image_path)
                        # Crop the image
                        cropped_img = img.crop((row['RegionLocationMinX0'], row['RegionLocationMinY0'], row['RegionLocationMaxX1'], row['RegionLocationMaxY1']))
                        group_images.append(cropped_img)
                
                if group_images:
                    # Create a title bar
                    if group_val == 'long':
                        title_bar_color = (100, 100, 255) 
                    elif group_val == 'trans':
                        title_bar_color = (100, 255, 100) 
                    elif group_val == 'doppler':
                        title_bar_color = (255, 100, 100) 
                    else:
                        title_bar_color = (255, 255, 255) 
                        
                    title_bar = Image.new('RGB', (max(img.width for img in group_images)*images_per_row, 100), title_bar_color)
                    d = ImageDraw.Draw(title_bar)
                    fnt = ImageFont.truetype('C:\\Windows\\Fonts\\Arial.ttf', 100)
                    d.text((10,10), str(group_val).upper(), font=fnt, fill=(0, 0, 0))

                    images.append(title_bar)
                    image_records.append(None)
                    total_height += 100

                    for index, row in group.iterrows():
                        image_path = os.path.join(image_input_folder, row['image_filename'])
                        if os.path.isfile(image_path):
                            image_records.append({'group': group_val, 
                                                'image_filename': row['image_filename'], 
                                                'us_x0': row['RegionLocationMinX0'], 
                                                'us_y0': row['RegionLocationMinY0'], 
                                                'us_x1': row['RegionLocationMaxX1'], 
                                                'us_y1': row['RegionLocationMaxY1']})
                    
                    row_width = 0
                    for i in range(0, len(group_images), images_per_row):
                        img1 = group_images[i]
                        for j in range(images_per_row):
                            row_width += img1.width
                            try:
                                images.append(group_images[i+j])
                            except IndexError:  # If there is an odd number of images
                                img2 = Image.new('RGB', img1.size, (0, 0, 0))  # Create an empty image
                                image_records.append({'group': "empty_space", 'image_filename': ''})
                                images.append(img2)
                        
                        total_width = max(total_width, row_width)
                        row_width = 0
                        total_height += img1.height

            if images:
                # Join all images into one
                new_img = Image.new('RGB', (total_width, total_height))
                y_offset = 0
                x_offset = 0
                reset_row = 0
                # Add the images to the new image
                for index, img in enumerate(images):
                
                    record = image_records[index]
                    
                    if record is None:
                        new_img.paste(img, (0, y_offset))
                        reset_row = 0
                        x_offset = 0
                        y_offset += img.height
                    else:
                        if record['group'] != "empty_space": # Skip Empty Spaces
                            writer.writerow([patient_id, 
                                            record['group'], 
                                            record['image_filename'], 
                                            x_offset, 
                                            y_offset, 
                                            img.width, 
                                            img.height,
                                            record['us_x0'],
                                            record['us_y0'], 
                                            record['us_x1'], 
                                            record['us_y1']])
                        
                        new_img.paste(img, (x_offset, y_offset))
                        
                        reset_row += 1
                        if reset_row == images_per_row:
                            reset_row = 0
                            x_offset = 0
                            y_offset += img.height
                        else:
                            x_offset += img.width
                            
                            
                    # Save the new image
                    new_img.save(os.path.join(output_folder, f'{patient_id}.png'))
                            
        print("Image processing completed.")
    

# Useful Labels:
# image_id, patient_id, filename
# us_x0, us_y0, us_x1, us_y1  OR  crop_x, crop_y, crop_w, crop_h
# image_type (RGB = Doppler images)
# area (if not "breast" put in other)
# orientation (Long, Trans, anything else put in other)

"""env = os.path.dirname(os.path.abspath(__file__))
image_input = f"{env}/downloads/images/"
image_output = f"{env}/labelbox_data/labelbox_images/"
input_csv = f"{env}/database/temp.csv"
output_csv = f"{env}/labelbox_data/crop_data.csv"

Crop_and_save_images(input_csv, image_input, output_csv, image_output, 4)"""