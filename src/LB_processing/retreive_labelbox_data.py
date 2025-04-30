import labelbox, requests, re, time
import pandas as pd
from tqdm import tqdm


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
                #label_df[column].to_csv(f'{"D:/DATA/CASBUSI/labelbox_data/"}/wtf{column}.csv', index=False)
                
                # List of preferred column names in order
                preferred_cols = ['answer', 'answers', 'answer.title', 'answer.value']

                # Select the first column that is found in normalized_df.columns
                answer_column = next((col for col in preferred_cols if col in normalized_df.columns), None)

                if answer_column is None:
                    print("No preferred column found in normalized_df")
                    continue  # Skip the rest of the loop for this row

                for i in range(normalized_df.shape[0]):
                    title = normalized_df.loc[i, 'title'].replace('_old', '')
                    answer = normalized_df.loc[i, answer_column]
                    
                    # Handle multiple answers within a list
                    if isinstance(answer, list):
                        answer = '; '.join([ans['title'] for ans in answer if 'title' in ans])
                    
                    data_dict[title] = answer
                                        

            data_dict['Accession_Number'] = external_id_digits

            # Add data to dataframe
            joined_labels_df = pd.concat([joined_labels_df, pd.DataFrame([data_dict])], ignore_index=True)

        # Reorder columns to ensure Accession_Number is first
        columns = ['Accession_Number'] + [col for col in joined_labels_df.columns if col != 'Accession_Number']
        joined_labels_df = joined_labels_df[columns]

        return joined_labels_df

    else:
        print(f'Failed to download file: {response.status_code}, {response.text}')
    
    
def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, database_path, labelbox_path):
    print("(Newly created data in labelbox will take time to update!)")
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)
    
    image_df = pd.read_csv(f'{database_path}/ImageData.csv')
    
    loss_refrences = pd.read_csv(f'{database_path}/LossLabelingReferences.csv')
    loss_refrences['Accession_Number'] = loss_refrences['Accession_Number'].astype(str)
    try:
        previous_df = pd.read_csv(f'{labelbox_path}/InstanceLabels.csv')
        previous_df['Accession_Number'] = previous_df['Accession_Number'].astype(str)
    except FileNotFoundError:
        previous_df = pd.DataFrame(columns=['Accession_Number'])  # Include 'Accession_Number' column
    
    print("Contacting Labelbox")
    export_url = project.export_labels()

    # Download the export file from the provided URL
    response = requests.get(export_url)

    # Parsing Labelbox Data
    print("Parsing Labelbox Data")
    instanceLabels = Get_Labels(response)
    instanceLabels['Accession_Number'] = instanceLabels['Accession_Number'].astype(str)
    
    #instanceLabels.to_csv(f'{labelbox_path}/InstanceLabels2.csv', index=False)

    # Get the Accession_Number that are in the old DataFrame
    if not previous_df.empty:
        old_patient_ids = set(previous_df['Accession_Number'])
    else:
        old_patient_ids = set()

    # Create a new DataFrame that contains only the new data
    new_data = instanceLabels[~instanceLabels['Accession_Number'].isin(old_patient_ids)]
    
    instanceLabels = pd.concat([previous_df, new_data])
    
    #instanceLabels.to_csv(f'{labelbox_path}/InstanceLabels3.csv', index=False)
    
    # Step 1: Transform instanceLabels to long format
    instanceLabels_long = pd.melt(instanceLabels, id_vars=['Accession_Number'], var_name='Placement', value_name='Label')
    instanceLabels_long['Accession_Number'] = instanceLabels_long['Accession_Number'].astype(str)
    
    # Identify Accession Numbers to Reject
    discard_exam_accessions = instanceLabels_long[
        (instanceLabels_long['Placement'] == 'Exam Options') & 
        (instanceLabels_long['Label'] == 'Discard Entire Exam')
    ]['Accession_Number'].unique()

    # Set 'Reject Image' for all instances with these accession numbers
    for acc_num in discard_exam_accessions:
        instanceLabels_long.loc[instanceLabels_long['Accession_Number'] == acc_num, 'Label'] = 'Reject Image'

    # Step 2: Merge with loss_refrences
    merged_df = pd.merge(loss_refrences, instanceLabels_long, on=['Accession_Number', 'Placement'], how='left')
    
    #merged_df.to_csv(f'{labelbox_path}/InstanceLabels4.csv', index=False)

    def is_label_present(labels, label_name):
        # Check if labels is NaN or otherwise not a string
        if pd.isna(labels) or not isinstance(labels, str):
            return False
        # Split the labels string into a list if it's a string
        labels = labels.split('; ')
        # Check if label_name is in the list of labels
        return label_name in labels

    # Apply the function to create the Boolean columns
    merged_df['Reject Image'] = merged_df['Label'].apply(lambda labels: is_label_present(labels, 'Reject Image'))
    merged_df['Only Normal Tissue'] = merged_df['Label'].apply(lambda labels: is_label_present(labels, 'Only Normal Tissue'))
    merged_df['Cyst Lesion Present'] = merged_df['Label'].apply(lambda labels: is_label_present(labels, 'Cyst Lesion Present'))
    merged_df['Benign Lesion Present'] = merged_df['Label'].apply(lambda labels: is_label_present(labels, 'Benign Lesion Present'))
    merged_df['Malignant Lesion Present'] = merged_df['Label'].apply(lambda labels: is_label_present(labels, 'Malignant Lesion Present'))
    

    # Filter out rows where all three columns are False
    condition = (merged_df['Reject Image'] | merged_df['Only Normal Tissue'] | merged_df['Cyst Lesion Present'] | merged_df['Benign Lesion Present'] | merged_df['Malignant Lesion Present'])
    filtered_df = merged_df[condition]
    
    #filtered_df.to_csv(f'{labelbox_path}/InstanceLabels5.csv', index=False)
    
    # Select only required columns
    final_df = pd.merge(filtered_df, image_df[['ImageName', 'FileName']], on='ImageName', how='left')


    # Select only required columns, including the new FileName column
    final_df = final_df[['Accession_Number', 'FileName', 'Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']]

    # Write final csv to disk
    final_df.to_csv(f'{labelbox_path}/InstanceLabels.csv', index=False)