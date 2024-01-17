import os
import labelbox, requests, os
import pandas as pd
from training_util import *
import json
env = os.path.dirname(os.path.abspath(__file__))


# Returns df with all labels in respective columns
# Also Downloads mask images
def Get_Labels(response):
    
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
            
            images = []
            inner_crop = []
            descriptions = []
            
            segmentation_df = pd.json_normalize(row_data["Label.objects"]).add_prefix("Label.objects.")
            external_id_digits = str(row_data['External ID'].iloc[0])
            
            for _, row in segmentation_df.iterrows():
                row = pd.json_normalize(row)

                for index in range(len(row)):
                    single_segmentation_row = row.loc[index]
                    data_type = single_segmentation_row['value']
                    
                    if data_type == "image":
                        # Extract the coordinates and append them in [x1, y1, x2, y2] format
                        x1 = single_segmentation_row['bbox.left']
                        y1 = single_segmentation_row['bbox.top']
                        x2 = x1 + single_segmentation_row['bbox.width']
                        y2 = y1 + single_segmentation_row['bbox.height']
                        images.append([x1, y1, x2, y2])
                    elif data_type == "largest_inner":
                        # Extract the coordinates and append them in [x1, y1, x2, y2] format
                        x1 = single_segmentation_row['bbox.left']
                        y1 = single_segmentation_row['bbox.top']
                        x2 = x1 + single_segmentation_row['bbox.width']
                        y2 = y1 + single_segmentation_row['bbox.height']
                        inner_crop.append([x1, y1, x2, y2])
                    elif data_type == "description":
                        # Extract the coordinates and append them in [x1, y1, x2, y2] format
                        x1 = single_segmentation_row['bbox.left']
                        y1 = single_segmentation_row['bbox.top']
                        x2 = x1 + single_segmentation_row['bbox.width']
                        y2 = y1 + single_segmentation_row['bbox.height']
                        descriptions.append([x1, y1, x2, y2])
                        
   
                        
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

            data_dict['image_mask'] = images
            data_dict['inner_mask'] = inner_crop
            data_dict['description_mask'] = descriptions
            data_dict['patient_id'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Move patient_id to the left
        cols = ['patient_id'] + [col for col in joined_labels_df.columns if col not in ['patient_id']]
        joined_labels_df = joined_labels_df.reindex(columns=cols)
        
        # Converting the list objects to custom string representation
        joined_labels_df['image_mask'] = joined_labels_df['image_mask'].apply(lambda x: x[0] if x else [])
        joined_labels_df['inner_mask'] = joined_labels_df['inner_mask'].apply(lambda x: x[0] if x else [])
        joined_labels_df['description_mask'] = joined_labels_df['description_mask'].apply(lambda x: x[0] if x else [])

        
        return joined_labels_df
    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')






def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images):
    print("(Newly created data in labelbox will take time to update!)")
    
    # Create Dataset Folder
    os.makedirs(f"{env}/dataset/", exist_ok=True)
    
    client = labelbox.Client(api_key = LB_API_KEY)
    project = client.get_project(PROJECT_ID)
    print("Contacting Labelbox")
    export_url = project.export_labels()
    
    # Download the export file from the provided URL
    print("Downloading data")
    response = requests.get(export_url)

    # Parse Data from labelbox
    print("Parsing Labelbox Data")
    df = Get_Labels(response)
    
    # Add image size to the dataframe
    df['image_size'] = df['patient_id'].apply(
        lambda image_filename: get_image_size(os.path.join(original_images, image_filename))
    )

    

    # Translate labels to 0 or 1
    df.rename(columns={'booleans': 'has_calipers'}, inplace=True)
    df['has_calipers'] = df['has_calipers'].apply(lambda x: 0 if pd.isnull(x) else 1)  
     

    # Write final csv to disk
    df.to_csv(f'{env}/dataset/labeled_data.csv', index=False)
    
    
    
    
    
    
    
    
# Config
with open('F:/CODE/CASBUSI/CASBUSI-Database/config.json', 'r') as config_file:
    config = json.load(config_file)

LB_API_KEY = config['LABELBOX_API_KEY']
PROJECT_ID = 'clj7jkqkm042907xlbm8s3xd0'
original_images = f"{env}/dataset/images/"


Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images)