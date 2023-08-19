import labelbox, requests, os, re, shutil, time, PIL, glob
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))


# Need retry method becauyse labelbox servers are unreliable
def get_with_retry(url, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
            else:
                print(f"\nFailed to download mask image, retrying {retries}/{max_retries}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        
        # Sleep for a bit before retrying
        time.sleep(2 * retries)
        retries += 1
    
    # If we've exhausted all retries, return None
    print(f"Failed to download mask image, status code: {response.status_code}")
    return None


# Returns df with all labels in respective columns
# Also Downloads mask images
def Get_Labels(response):

    # Create Mask Folder
    os.makedirs(f"{env}/database/temp_labelbox_data/", exist_ok=True)
    
    # Ensure we received a valid response
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} Data Rows")
        
        # Flatten json into a dataframe
        df = pd.json_normalize(data)  
        # Create empty df to build off of
        joined_labels_df = pd.DataFrame()

        # Iterate over each row in df
        for _, row_data in tqdm(df.iterrows(), total=df.shape[0]):
            # Get segmentation data
            row_data = pd.DataFrame([row_data])
            segmentation_df = pd.json_normalize(row_data["Label.objects"]).add_prefix("Label.objects.")
            external_id_digits = str(int("".join(re.findall("\d+", row_data['External ID'].iloc[0]))))
            mask_paths = []  # Initialize an empty list to store mask paths
            bad_images = [] 
            doppler_image = []
            cyst_image = []
            normal_image = []
            
            for _, row in segmentation_df.iterrows():
                row = pd.json_normalize(row)

                for index in range(len(row)):
                    single_segmentation_row = row.loc[index]
                    
                    data_type = single_segmentation_row['value']
                    
                    if data_type == "doppler_lesion":
                        doppler_image.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    elif data_type == "bad_images":
                        bad_images.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    elif data_type == "cyst":
                        cyst_image.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    elif data_type == "normal_tissue":
                        normal_image.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    else:
                        # Get the mask image
                        mask_response = get_with_retry(single_segmentation_row['instanceURI'])
                        if mask_response is None:
                            print(f"Failed to download image after multiple retries: {single_segmentation_row['instanceURI']}")
                        else:
                            try:
                                mask_img = Image.open(BytesIO(mask_response.content))
                            except PIL.UnidentifiedImageError:
                                print(f"Cannot identify image file: {single_segmentation_row['instanceURI']}")
                                
                        mask_img = Image.open(BytesIO(mask_response.content))
                        mask_path = f"{external_id_digits}_{data_type}.png"
                        mask_img.save(f"{env}/database/temp_labelbox_data/{mask_path}")
                        mask_paths.append(mask_path)
            
            # Get Label Data
            label_df = pd.json_normalize(row_data["Label.classifications"])
            data_dict = {}

            for column in label_df.columns:
                normalized_df = pd.json_normalize(label_df[column])

                # List of preferred column names in order
                preferred_cols = ['answer', 'answers', 'answer.title', 'answer.value']

                # Select the first column that is found in normalized_df.columns
                answer_column = next((col for col in preferred_cols if col in normalized_df.columns), None)

                if answer_column is None:
                    print("No preferred column found in normalized_df")
                    continue  # Skip the rest of the loop for this row

                # Create a dictionary for each record with title as key and answer as value
                for i in range(normalized_df.shape[0]):
                    title = normalized_df.loc[i, 'title']
                    answer = normalized_df.loc[i, answer_column]
                    if answer_column == 'answers':
                        # If answer is a list of dictionaries, extract the 'value' from each dictionary
                        answer = [ans.get('value') for ans in answer]
                    data_dict[title] = str(answer)

            data_dict['mask_names'] = str(', '.join(mask_paths))
            data_dict['bad_images'] = bad_images
            data_dict['doppler_image'] = doppler_image
            data_dict['cyst_images'] = cyst_image
            data_dict['normal_images'] = normal_image
            data_dict['Patient_ID'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Move Patient_ID to the left
        cols = ['Patient_ID', 'mask_names', 'bad_images', 'doppler_image', 'cyst_images', 'normal_images'] + [col for col in joined_labels_df.columns if col not in ['Patient_ID', 'mask_names', 'bad_images', 'doppler_image', 'cyst_images', 'normal_images']]
        joined_labels_df = joined_labels_df.reindex(columns=cols)
        
        return joined_labels_df
    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')


def Find_Images(df, crop_data, df_cords_col, df_image_names_col):
    df['Patient_ID'] = df['Patient_ID'].astype(int)
    crop_data['Patient_ID'] = crop_data['Patient_ID'].astype(int)
            
    cords = df[df_cords_col].apply(lambda x: np.array(x) if isinstance(x, str) else x)
    
    df[df_image_names_col] = ''

    # Iterate over each row in df
    for i, row in df.iterrows():
        Patient_ID = row['Patient_ID']
        images = []
        
        for sublist in cords[i]:
            if isinstance(sublist, np.ndarray):
                sublist = sublist.tolist()  # Convert numpy array to a regular list
            
            x_point, y_point = sublist

            # Find the corresponding row in crop_data
            for j, crop_row in crop_data[crop_data['Patient_ID'] == Patient_ID].iterrows():
                
                # Check if the coordinates fall within the bounds of the image
                if (crop_row['x'] <= x_point < crop_row['x'] + crop_row['width']) and \
                    (crop_row['y'] <= y_point < crop_row['y'] + crop_row['height']):
                    # If so, add the image filename to the images list
                    images.append(crop_row['ImageName'])
                    
        # Convert the list of filenames to a comma-separated string
        df.at[i, df_image_names_col] = ', '.join(images)
                    
    return df

def Find_Masks(df, crop_data, df_mask_names_col, df_image_names_col, original_images):
    # Create Mask Folder
    os.makedirs(f"{env}/database/masks/", exist_ok=True)
    
    
    df['Patient_ID'] = df['Patient_ID'].astype(int)
    crop_data['Patient_ID'] = crop_data['Patient_ID'].astype(int)

    df[df_image_names_col] = ''

    # Iterate over each row in df
    for i, row in df.iterrows():
        Patient_ID = row['Patient_ID']
        mask_names = row[df_mask_names_col].split(', ')

        # list to store all related image file names for each mask_name
        all_images_for_mask = []
        mask_paths = []
        
        for mask_name in mask_names:
            if mask_name:
                mask_img = Image.open(f"{env}/database/temp_labelbox_data/{mask_name}")
                mask_np = np.array(mask_img)

                # Find all non-zero points in the mask
                mask_point = np.transpose(np.nonzero(mask_np))[:, :2][0]
                images = []
                

                # Find the corresponding row in crop_data
                for _, crop_row in crop_data[crop_data['Patient_ID'] == Patient_ID].iterrows():

                    # Check if any of the mask points fall within the bounds of the image
                    y_point, x_point = mask_point

                    if (crop_row['x'] <= x_point < crop_row['x'] + crop_row['width']) and \
                        (crop_row['y'] <= y_point < crop_row['y'] + crop_row['height']):
                        # If so, add the image filename to the images list
                        images.append(crop_row['ImageName'])
                        
                        # Create and save a cropped version of the mask
                        left = crop_row['x']
                        upper = crop_row['y']
                        right = crop_row['x'] + crop_row['width']
                        lower = crop_row['y'] + crop_row['height']
                        cropped_mask = mask_img.crop((left, upper, right, lower))
                        
                        # Open original image to grab the size
                        original_img = Image.open(f"{original_images}{crop_row['ImageName']}")
                        original_img = Image.new(original_img.mode, original_img.size) # Make original image blank
                        
                        # Box dimensions need to match those of the cropped image
                        box = (crop_row['us_x0'], crop_row['us_y0'], crop_row['us_x1'], crop_row['us_y1'])
                        mask_path = f"mask_{crop_row['ImageName']}"
                        mask_paths.append(mask_path)
                        
                        original_img.paste(cropped_mask, box)
                        original_img.save(f"{env}/database/masks/{mask_path}")
                        
                        
                        break  # We found an image that the mask belongs to, no need to check other points
                
                # Add list of image filenames to all_images_for_mask
                all_images_for_mask.extend(images)

        # Convert the list of filenames to a comma-separated string and store it in the df
        df.at[i, df_mask_names_col] = str(', '.join(mask_paths))
        df.at[i, df_image_names_col] = ', '.join(all_images_for_mask)

    #Delete old masks
    shutil.rmtree(f"{env}/database/temp_labelbox_data/")
    
    return df
        
        
        
def move_used_images(df, original_images_dir, used_images_dir):
    # Create directory for used images
    os.makedirs(used_images_dir, exist_ok=True)

    # Get all unique patient IDs from the df
    patient_ids = df['Patient_ID'].unique()

    # Iterate over each patient ID
    for patient_id in patient_ids:
        # Find the image that contains the patient ID in its name
        for image_path in glob.glob(f"{original_images_dir}{patient_id}_*.png"):
            # Move the image to the used_images_dir
            shutil.move(image_path, f"{used_images_dir}{os.path.basename(image_path)}")


def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images):
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)

    print("Contacting Labelbox")
    export_url = project.export_labels()

    # Download the export file from the provided URL
    response = requests.get(export_url)

    # Parse Data from labelbox
    print("Parsing Labelbox Data")
    df = Get_Labels(response)

    print("Refrencing Original Images")
    crop_data = pd.read_csv(f'{env}/database/CropData.csv')
    df = Find_Images(df, crop_data, 'doppler_image', 'doppler_image_names')
    df = Find_Images(df, crop_data, 'bad_images', 'bad_image_names')
    df = Find_Images(df, crop_data, 'cyst_images', 'cyst_image_names')
    df = Find_Images(df, crop_data, 'normal_images', 'normal_image_names')
    df = Find_Masks(df, crop_data, 'mask_names', 'masked_original_names', original_images)
    
    labelbox_images = f"{env}/database/labelbox_images/"
    used_images_dir = f"{env}/database/used_images/"
    move_used_images(df, labelbox_images, used_images_dir)

    # Reorder Columns
    ordering = ['Patient_ID', 'mask_names', 'masked_original_names', 'bad_image_names', 'doppler_image_names', 'cyst_image_names', 'normal_image_names']
    cols = ordering + [col for col in df.columns if col not in ordering]
    df = df.reindex(columns=cols)

    # List of columns to be dropped
    columns_to_drop = ['bad_images', 'doppler_image', 'cyst_images', 'normal_images', '==== SECTION 1 ====', '==== SECTION 2 ====', '==== SECTION 3 ====', '==== SECTION 4 ====', '==== SECTION 5 ====']

    # Check if the columns exist in the DataFrame before dropping them
    columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]

    # Drop the existing columns
    df = df.drop(columns_to_drop_existing, axis=1)

    # Write final csv to disk
    df.to_csv(f'{env}/database/LabeledData.csv', index=False)