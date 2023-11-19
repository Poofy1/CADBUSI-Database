import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
env = os.path.dirname(os.path.abspath(__file__))


def process_group(Patient_ID, laterality, patient_group, images_per_row, output_folder, image_input_folder):
    
    # Define the output image path
    output_image_path = os.path.join(output_folder, f'{int(Patient_ID)}_{laterality}.png')

    # Check if the image already exists
    #if os.path.isfile(output_image_path):
    #    return pd.DataFrame()
    
    images = []
    image_records = []
    total_height = 0
    total_width = 0
    
    # Track groups that we have found
    found_groups = set()

    for group_val in ['long', 'trans', 'doppler', 'other']:
        group = patient_group[patient_group['group'] == group_val]
        group_images = []

        for index, row in group.iterrows():
            image_path = os.path.join(image_input_folder, row['ImageName'])

            if os.path.isfile(image_path):
                img = Image.open(image_path)
                # Crop the image
                cropped_img = img.crop((row['RegionLocationMinX0'], row['RegionLocationMinY0'], row['RegionLocationMaxX1'], row['RegionLocationMaxY1']))
                group_images.append(cropped_img)

        if group_images:
            # This group has images, so add it to the found_groups set
            found_groups.add(group_val)
            
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
                image_name = row['ImageName']
                

                image_records.append({'group': group_val, 
                                    'ImageName': image_name, 
                                    'us_x0': row['RegionLocationMinX0'], 
                                    'us_y0': row['RegionLocationMinY0'], 
                                    'us_x1': row['RegionLocationMaxX1'], 
                                    'us_y1': row['RegionLocationMaxY1'],})
            
            row_width = 0
            for i in range(0, len(group_images), images_per_row):
                img1 = group_images[i]
                for j in range(images_per_row):
                    row_width += img1.width
                    try:
                        images.append(group_images[i+j])
                    except IndexError:  # If there is an odd number of images
                        img2 = Image.new('RGB', img1.size, (0, 0, 0))  # Create an empty image
                        image_records.append({'group': "empty_space", 'ImageName': ''})
                        images.append(img2)
                
                total_width = max(total_width, row_width)
                row_width = 0
                total_height += img1.height

    new_data = pd.DataFrame(columns=['Patient_ID', 'group', 'ImageName', 'x', 'y', 'width', 'height', 'us_x0', 'us_y0', 'us_x1', 'us_y1'])
    
    if set(['long', 'trans', 'doppler']).issubset(found_groups):
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
                        # If the image filename exists
                        new_row = pd.DataFrame([{
                            'Patient_ID': int(Patient_ID),
                            'group': record['group'],
                            'ImageName': record['ImageName'],
                            'x': x_offset,
                            'y': y_offset,
                            'width': img.width,
                            'height': img.height,
                            'us_x0': int(record['us_x0']),
                            'us_y0': int(record['us_y0']),
                            'us_x1': int(record['us_x1']),
                            'us_y1': int(record['us_y1'])}])
                        # New code: append the new row to new_data
                        new_data = new_data.append(new_row, ignore_index=True)
                        
                    new_img.paste(img, (x_offset, y_offset))
                    
                    reset_row += 1
                    if reset_row == images_per_row:
                        reset_row = 0
                        x_offset = 0
                        y_offset += img.height
                    else:
                        x_offset += img.width
                        
                # Save the new image
                new_img.save(output_image_path)

    return new_data





def Crop_and_save_images(images_per_row):
    print("Transforming Images for Labelbox")

    image_input_folder = f"{env}/database/images/"
    output_csv = f"{env}/database/CropData.csv"
    output_folder = f"{env}/database/labelbox_images/"
    labeled_data_file = f"{env}/database/LabeledData.csv"
    case_data = pd.read_csv(f"{env}/database/CaseStudyData.csv")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    data = pd.read_csv(f"{env}/database/ImageData.csv")
    
    data = data[data['label'] == True]

    # If there are no matching data, return
    if data.empty:
        print("No data for 'breast' area found.")
        return
    
    data['group'] = data['label_cat']
    
    # Merge the image data with the trustworthiness data on Patient_ID field
    case_data = case_data.drop_duplicates(subset='Patient_ID')
    merged_data = pd.merge(data, case_data, on='Patient_ID', how='left')

    # Filter out data with trustworthiness score above the threshold
    trust_threshold = 1
    data = merged_data[merged_data['trustworthiness'] <= trust_threshold]
    
    #Debug Data range
    #data = data[(data['Patient_ID'] >= 1100) & (data['Patient_ID'] <= 1200)]
    
    # Group the data by 'Patient_ID'
    grouped_patient = data.groupby(['Patient_ID', 'laterality'])

    # Check if the output CSV file exists
    csv_exists = os.path.isfile(output_csv)

    # Open the output CSV file in append mode if it exists, otherwise in write mode
    if csv_exists:
        existing_data = pd.read_csv(output_csv)
        if 'inpainted' not in existing_data.columns:
            existing_data['inpainted'] = False
    else:
        existing_data = pd.DataFrame(columns=['Patient_ID', 'group', 'ImageName', 'x', 'y', 'width', 'height', 'us_x0', 'us_y0', 'us_x1', 'us_y1'])
        
    #Skip formated data
    labeled_data = pd.read_csv(labeled_data_file) if os.path.isfile(labeled_data_file) else pd.DataFrame(columns=['Patient_ID'])

    # Create a ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit the process_group function to the executor with the trust threshold
        futures = {executor.submit(process_group, Patient_ID, laterality, patient_group, images_per_row, output_folder, image_input_folder) 
                for (Patient_ID, laterality), patient_group in grouped_patient if int(Patient_ID) not in labeled_data['Patient_ID'].values}
        pbar = tqdm(total=len(futures), desc="", unit="patient")
        for future in as_completed(futures):
            df_records = future.result()
            results.append(df_records)
            pbar.update()
        pbar.close()
        
        
    new_data = pd.concat(results, ignore_index=True)
    existing_data = existing_data.set_index(['Patient_ID', 'ImageName'])
    new_data = new_data.set_index(['Patient_ID', 'ImageName'])

    # Concatenate existing_data and new_data, and drop duplicates
    all_data = pd.concat([existing_data, new_data])
    all_data = all_data.loc[~all_data.index.duplicated(keep='last')]
    all_data.reset_index(inplace=True)
    
    # Apply the filtering to 'all_data' instead of 'existing_data'
    all_data = all_data[all_data['group'] != "empty_space"]
                        
    all_data.to_csv(output_csv, index=False)