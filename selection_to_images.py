import labelbox, requests, os, re, shutil
import pandas as pd
from PIL import Image
from io import BytesIO
from PIL import ImageChops
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))


# Returns df with all labels in respective columns
# Also Downloads mask images
def Get_Labels(response):

    # Create Mask Folder
    os.makedirs(f"{env}/temp/", exist_ok=True)
    
    # Ensure we received a valid response
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} Data Rows")
        
        # Flatten json into a dataframe
        df = pd.json_normalize(data)  
        # Create empty df to build off of
        joined_labels_df = pd.DataFrame()

        # Iterate over each row in df
        for _, row_data in df.iterrows():
            # Get segmentation data
            row_data = pd.DataFrame([row_data])
            segmentation_df = pd.json_normalize(row_data["Label.objects"]).add_prefix("Label.objects.")
            external_id_digits = str(int("".join(re.findall("\d+", row_data['External ID'].iloc[0]))))
            mask_paths = []  # Initialize an empty list to store mask paths
            bad_images = [] 
            doppler_image = []
            
            for _, row in segmentation_df.iterrows():
                row = pd.json_normalize(row)

                for index in range(len(row)):
                    single_segmentation_row = row.loc[index]
                    
                    data_type = single_segmentation_row['value']
                    
                    if data_type == "doppler_lesion":
                        doppler_image.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    elif data_type == "bad_images":
                        bad_images.append([single_segmentation_row['point.x'], single_segmentation_row['point.y']])
                    else:
                        # Get the mask image
                        mask_response = requests.get(single_segmentation_row['instanceURI'])
                        mask_img = Image.open(BytesIO(mask_response.content))
                        mask_path = f"{external_id_digits}_{data_type}.png"
                        mask_img.save(f"{env}/temp/{mask_path}")
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
            data_dict['patient_id'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Move patient_id to the left
        cols = ['patient_id', 'mask_names', 'bad_images', 'doppler_image'] + [col for col in joined_labels_df.columns if col not in ['patient_id', 'mask_names', 'bad_images', 'doppler_image']]
        joined_labels_df = joined_labels_df.reindex(columns=cols)
        
        return joined_labels_df
    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')


def Find_Images(df, crop_data, df_cords_col, df_image_names_col):
    df['patient_id'] = df['patient_id'].astype(int)
    crop_data['patient_id'] = crop_data['patient_id'].astype(int)
            
    cords = df[df_cords_col].apply(lambda x: np.array(x) if isinstance(x, str) else x)
    
    df[df_image_names_col] = ''

    # Iterate over each row in df
    for i, row in df.iterrows():
        patient_id = row['patient_id']
        images = []
        
        for sublist in cords[i]:
            if isinstance(sublist, np.ndarray):
                sublist = sublist.tolist()  # Convert numpy array to a regular list
            
            x_point, y_point = sublist

            # Find the corresponding row in crop_data
            for j, crop_row in crop_data[crop_data['patient_id'] == patient_id].iterrows():
                
                # Check if the coordinates fall within the bounds of the image
                if (crop_row['x'] <= x_point < crop_row['x'] + crop_row['width']) and \
                    (crop_row['y'] <= y_point < crop_row['y'] + crop_row['height']):
                    # If so, add the image filename to the images list
                    images.append(crop_row['image_filename'])
                    
        # Convert the list of filenames to a comma-separated string
        df.at[i, df_image_names_col] = ', '.join(images)
                    
    return df

def Find_Masks(df, crop_data, df_mask_names_col, df_image_names_col):
    # Create Mask Folder
    os.makedirs(f"{env}/database/masks/", exist_ok=True)
    
    
    df['patient_id'] = df['patient_id'].astype(int)
    crop_data['patient_id'] = crop_data['patient_id'].astype(int)

    df[df_image_names_col] = ''

    # Iterate over each row in df
    for i, row in df.iterrows():
        patient_id = row['patient_id']
        mask_names = row[df_mask_names_col].split(', ')

        # list to store all related image file names for each mask_name
        all_images_for_mask = []
        mask_paths = []

        for mask_name in mask_names:
            mask_img = Image.open(f"{env}/temp/{mask_name}")
            mask_np = np.array(mask_img)

            # Find all non-zero points in the mask
            mask_point = np.transpose(np.nonzero(mask_np))[:, :2][0]
            images = []
            

            # Find the corresponding row in crop_data
            for _, crop_row in crop_data[crop_data['patient_id'] == patient_id].iterrows():

                # Check if any of the mask points fall within the bounds of the image
                y_point, x_point = mask_point

                if (crop_row['x'] <= x_point < crop_row['x'] + crop_row['width']) and \
                    (crop_row['y'] <= y_point < crop_row['y'] + crop_row['height']):
                    # If so, add the image filename to the images list
                    images.append(crop_row['image_filename'])
                    
                    # Create and save a cropped version of the mask
                    left = crop_row['x']
                    upper = crop_row['y']
                    right = crop_row['x'] + crop_row['width']
                    lower = crop_row['y'] + crop_row['height']
                    cropped_mask = mask_img.crop((left, upper, right, lower))
                    
                    # Open original image to grab the size
                    original_img = Image.open(f"{original_images}{crop_row['image_filename']}")
                    original_img = Image.new(original_img.mode, original_img.size) # Make original image blank
                    
                    # Box dimensions need to match those of the cropped image
                    box = (crop_row['us_x0'], crop_row['us_y0'], crop_row['us_x1'], crop_row['us_y1'])
                    mask_path = f"mask_{crop_row['image_filename']}"
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
    shutil.rmtree(f"{env}/temp/")
    
    return df


print("Newly created data in labelbox will take time to update and appear here")

# Path Config
selection_images = f'{env}/selection_images'
original_images = 'F:/CODE/CASBUSI/Datasets/bus_2018-19/image/'

# Find Label Box
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGc5emFjOTIyMDZzMDcyM2E2MW0xbnpuIiwib3JnYW5pemF0aW9uSWQiOiJja290NnVvMWgxcXR0MHlhczNuNmlmZnRjIiwiYXBpS2V5SWQiOiJjbGh1dm5rMTAwYnV2MDcybjlpZ3g4NGdzIiwic2VjcmV0IjoiZmRhZjcxYzBhNDM3MmNkYWNkNWIxODU5MzUyNjc1ODMiLCJpYXQiOjE2ODQ1MTk4OTgsImV4cCI6MjMxNTY3MTg5OH0.DMecSgJDDZrX1qw2T4HLs5Sv62lLLT-ePcMjyxpn0aE'
PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu'
client = labelbox.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)
export_url = project.export_labels()

# Download the export file from the provided URL
response = requests.get(export_url)

# Parse Data from labelbox
print("Parsing Labelbox Data")
df = Get_Labels(response)

print("Locating Images")
crop_data = pd.read_csv(f'{env}/crop_data.csv')
df = Find_Images(df, crop_data, 'doppler_image', 'doppler_image_names')
df = Find_Images(df, crop_data, 'bad_images', 'bad_image_names')
df = Find_Masks(df, crop_data, 'mask_names', 'masked_original_names')

# Reorder Columns
ordering = ['patient_id', 'mask_names', 'masked_original_names', 'bad_images', 'bad_image_names', 'doppler_image', 'doppler_image_names']
cols = ordering + [col for col in df.columns if col not in ordering]
df = df.reindex(columns=cols)


# Write final csv to disk
df.to_csv(f'{env}/labels.csv', index=False)

