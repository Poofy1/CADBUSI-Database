import os, json
import labelbox, requests, os
import pandas as pd
from PIL import Image
from training_util import *
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

            orientations = []

            segmentation_df = pd.json_normalize(row_data["Label.objects"]).add_prefix("Label.objects.")
            external_id_digits = str(row_data['External ID'].iloc[0])
            

                        
            # Get Label Data
            label_df = pd.json_normalize(row_data["Label.classifications"])
            data_dict = {}
            
            for column in label_df.columns:

                normalized_df = pd.json_normalize(label_df[column])

                # List of preferred column names in order
                preferred_cols = ['answer.value']

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

            data_dict['patient_id'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Move patient_id to the left
        cols = ['patient_id'] + [col for col in joined_labels_df.columns if col not in ['patient_id']]
        joined_labels_df = joined_labels_df.reindex(columns=cols)
        
        
        return joined_labels_df
    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')


def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images):
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


    # Write final csv to disk
    df.to_csv(f'{env}/dataset/ori_data.csv', index=False)
    
    
    
    
    
    
# Config

with open('F:/CODE/CASBUSI/CASBUSI-Database/config.json', 'r') as config_file:
    config = json.load(config_file)

LB_API_KEY = config['LABELBOX_API_KEY']
PROJECT_ID = 'clkcfzgyo07nl07121qd41nqv'
original_images = f"{env}/dataset/orientation_images/"


print("(Newly created data in labelbox will take time to update!)")
Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images)