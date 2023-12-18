import labelbox, requests, os, re, shutil, time, PIL, glob, ast
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import datetime, glob
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
            external_id_digits = str(int("".join(re.findall("\d+", row_data['External ID'].iloc[0]))))

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

            data_dict['Accession_Number'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Reorder columns to ensure Accession_Number is first
        columns = ['Accession_Number'] + [col for col in joined_labels_df.columns if col != 'Accession_Number']
        joined_labels_df = joined_labels_df[columns]

        return joined_labels_df

    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')
    
    
def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, database_path):
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)
    
    loss_refrences = pd.read_csv(f'{database_path}/LossLabelingReferences.csv')
    loss_refrences['Accession_Number'] = loss_refrences['Accession_Number'].astype(str)
    try:
        previous_df = pd.read_csv(f'{database_path}/InstanceLabels.csv')
        previous_df['Accession_Number'] = previous_df['Accession_Number'].astype(str)
    except FileNotFoundError:
        previous_df = pd.DataFrame(columns=['Accession_Number'])  # Include 'Accession_Number' column
    
    print("Contacting Labelbox")
    export_url = project.export_labels()

    # Download the export file from the provided URL
    response = requests.get(export_url)

    # Parse Data from labelbox
    print("Parsing Labelbox Data")
    instanceLabels = Get_Labels(response)
    instanceLabels['Accession_Number'] = instanceLabels['Accession_Number'].astype(str)

    # Get the Accession_Number that are in the old DataFrame
    if not previous_df.empty:
        old_patient_ids = set(previous_df['Accession_Number'])
    else:
        old_patient_ids = set()

    # Create a new DataFrame that contains only the new data
    new_data = instanceLabels[~instanceLabels['Accession_Number'].isin(old_patient_ids)]
    
    instanceLabels = pd.concat([previous_df, new_data])
    
    # Step 1: Transform instanceLabels to long format
    instanceLabels_long = pd.melt(instanceLabels, id_vars=['Accession_Number'], var_name='Placement', value_name='Label')
    instanceLabels_long['Accession_Number'] = instanceLabels_long['Accession_Number'].astype(str)

    # Step 2: Merge with loss_refrences
    merged_df = pd.merge(loss_refrences, instanceLabels_long, on=['Accession_Number', 'Placement'], how='left')

    # Step 3: Create Boolean Columns
    merged_df['Reject'] = merged_df['Label'] == 'Reject'
    merged_df['Malignancy Present'] = merged_df['Label'] == 'Malignancy Present'
    merged_df['Malignancy Absent'] = merged_df['Label'] == 'Malignancy Absent'
    merged_df['Both Benign/Malignancy Present'] = merged_df['Label'] == 'Both Benign/Malignancy Present'

    # Filter out rows where all three columns are False
    condition = (merged_df['Reject'] | merged_df['Malignancy Present'] | merged_df['Malignancy Absent'] | merged_df['Both Benign/Malignancy Present'])
    filtered_df = merged_df[condition]

    # Select only required columns
    final_df = filtered_df[['Accession_Number', 'ImageName', 'Reject', 'Malignancy Present', 'Malignancy Absent', 'Both Benign/Malignancy Present']]

    # Write final csv to disk
    final_df.to_csv(f'{database_path}/InstanceLabels.csv', index=False)