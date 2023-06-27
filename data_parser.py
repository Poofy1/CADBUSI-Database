import pandas as pd
import json, shutil, os, io
from tqdm import tqdm
import warnings

# Initialization
warnings.simplefilter(action='ignore', category=FutureWarning)
env = os.path.dirname(os.path.abspath(__file__))

# Static
database_CSV = f"{env}/database/unlabeled_data.csv"


def MergeSess(combined_df, merged_df):
    # Merge the current file's DataFrame to the combined DataFrame using 'id' column
    if combined_df.empty:
        return merged_df
    elif 'id' in merged_df.columns:
        return pd.merge(combined_df, merged_df, on='id', how='outer')
    else:
        # Find the new columns in merged_df that are not present in combined_df
        new_columns = [col for col in merged_df.columns if col not in combined_df.columns]
        
        # Add the new columns to combined_df with NaN values
        for col in new_columns:
            combined_df[col] = ''

        updated_rows = []

        # Iterate through unique 'anonymized_accession_num' values in merged_df
        for unique_num in merged_df['anonymized_accession_num'].unique():
            # Filter rows from both dataframes with matching 'anonymized_accession_num'
            combined_rows = combined_df[combined_df['anonymized_accession_num'] == unique_num].copy()
            merged_rows = merged_df[merged_df['anonymized_accession_num'] == unique_num]

            # Merge the filtered rows on 'anonymized_accession_num' and update the new columns
            combined_rows = combined_rows.merge(merged_rows, on='anonymized_accession_num', suffixes=('', '_y'))

            # Copy the new column values to the original columns in combined_rows
            for col in new_columns:
                combined_rows[col] = combined_rows[col + '_y']
                combined_rows.drop(col + '_y', axis=1, inplace=True)

            updated_rows.append(combined_rows)

        # Concatenate updated rows and drop the original rows from combined_df
        updated_rows_df = pd.concat(updated_rows, ignore_index=True)
        combined_df = pd.concat([combined_df, updated_rows_df]).drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
        return combined_df     

def merge_without_overwrite(combined_df, merged_df):
    if combined_df.empty:
        return merged_df
    elif 'id' in merged_df.columns: # Check if the 'id' column is present merged_df

        # Iterate through the columns in merged_df
        for col in merged_df.columns:
            # Check if the column is not in combined_df
            if col not in combined_df.columns:
                # Add the new column to combined_df with empty values
                combined_df[col] = ''

            # Check if the column is empty in any row of combined_df
            empty_rows = combined_df[combined_df[col] == '']
            if not empty_rows.empty:
                # Update the empty rows with the values from merged_df
                merged_rows = merged_df.loc[empty_rows.index][col]
                combined_df.loc[empty_rows.index, col] = merged_rows

        # Drop duplicates based on the 'id' column and keep the last occurrence
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
    else: 
        # Find the new columns in merged_df that are not present in combined_df
        new_columns = [col for col in merged_df.columns if col not in combined_df.columns]
        
        # Add the new columns to combined_df with NaN values
        for col in new_columns:
            combined_df[col] = ''

        updated_rows = []

        # Iterate through unique 'anonymized_accession_num' values in merged_df
        for unique_num in merged_df['anonymized_accession_num'].unique():
            # Filter rows from both dataframes with matching 'anonymized_accession_num'
            combined_rows = combined_df[combined_df['anonymized_accession_num'] == unique_num].copy()
            merged_rows = merged_df[merged_df['anonymized_accession_num'] == unique_num]

            # Merge the filtered rows on 'anonymized_accession_num' and update the new columns
            combined_rows = combined_rows.merge(merged_rows, on='anonymized_accession_num', suffixes=('', '_y'))

            # Copy the new column values to the original columns in combined_rows only if the slot is empty
            for col in new_columns:
                empty_rows = combined_rows[combined_rows[col].isna()]
                merged_rows = merged_rows.loc[empty_rows.index, col]
                combined_rows.loc[empty_rows.index, col] = merged_rows

            # Drop the '_y' suffix from merged columns
            combined_rows = combined_rows.drop([col for col in combined_rows.columns if '_y' in col], axis=1)

            updated_rows.append(combined_rows)

        # Concatenate updated rows and drop the original rows from combined_df
        updated_rows_df = pd.concat(updated_rows, ignore_index=True)
        combined_df = pd.concat([combined_df, updated_rows_df]).drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)

    
    #add new rows
    updated_existing_df = pd.concat([combined_df, merged_df]).drop_duplicates(subset=['id'], keep='first')

    # Reset the index
    updated_existing_df.reset_index(drop=True, inplace=True)

    return updated_existing_df


def TransformJSON(data_labels, folder):
    combined_df = pd.DataFrame()
    
    # Iterate through the files and their labels
    for file_name, labels in data_labels.items():

        if not os.path.exists(os.path.join(env, folder, file_name)):
            print(f"File {file_name} not found. Skipping this file.")
            continue
        
        if os.path.splitext(file_name)[1] == ".json":
            
            with open(os.path.join(env, folder, file_name), 'r') as f:
                data = json.load(f)
            
            # Process the JSON data, similar to what you did earlier
            dicom_items = []
            for obj in data:
                for item in obj['dicoms']:
                    new_item = item.copy()

                    for key, value in obj.items():
                        if key != 'dicoms':
                            new_item[key] = value

                    dicom_items.append(new_item)

            dicoms_df = pd.json_normalize(dicom_items)
            regions_df = pd.json_normalize(dicom_items, record_path=["metadata", "SequenceOfUltrasoundRegions"])
            merged_df = pd.merge(dicoms_df, regions_df, left_index=True, right_index=True)
            merged_df = merged_df.rename(columns=lambda x: x.replace("metadata.", ""))
            
        elif os.path.splitext(file_name)[1] == ".csv":
            # Read the CSV file using pandas
            merged_df = pd.read_csv(os.path.join(env, folder, file_name))
        


        # Convert the 'id' columns in both DataFrames to strings
        merged_df = merged_df.astype(str)

        # Filter the DataFrame to keep only the desired columns
        merged_df = merged_df[labels]
        
        combined_df = MergeSess(combined_df, merged_df)
        
        #debug
        #combined_df['id'] = combined_df['id'].astype(int)
        #combined_df = combined_df[combined_df['id'] >= 10]
        
        
    # Convert the combined DataFrame to CSV and read it back
    csv_string = io.StringIO()
    combined_df.to_csv(csv_string, index=False)
    
    
    
    return pd.read_csv(io.StringIO(csv_string.getvalue())).astype(str).replace('nan', '')





def PerformEntry(folder, data_labels, enable_overwritting, data_range=None):
    
    
    # Check Dirs
    if not os.path.exists(f"{env}/database/"):
        os.makedirs(f"{env}/database/", exist_ok=True)
    if not os.path.exists(f"{env}/database/images/"):
        os.makedirs(f"{env}/database/images/", exist_ok=True)
    if not os.path.exists(f"{env}/raw_data/"):
        os.makedirs(f"{env}/raw_data/", exist_ok=True)
    
    
    print("Transforming JSON/CSV")
    df = TransformJSON(data_labels, folder)
    
    # Read the existing data.csv file into a DataFrame, if it exists
    if os.path.exists(database_CSV):
        existing_df = pd.read_csv(database_CSV, dtype=str)
    else:
        existing_df = pd.DataFrame(columns=df.columns)
    
    
    # if Stage 1
    new_images = f"{env}/{folder}/images/"
    if os.path.exists(new_images):
        
        # Initialize an empty list to store the extracted text and filenames
        text_list = []
        files = os.listdir(new_images)
        
        # Loop through all the files in the folder
        for file_name in files[data_range[0]:data_range[1]]:
            # Check if the file is an image
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Extract the ID from the filename
                id = file_name.split("_")[0].lstrip('0')

                # Append the extracted text and filename to the list
                text_list.append((id, file_name))


        # Create a pandas DataFrame from the text list
        image_df = pd.DataFrame(text_list, columns=['id', 'image_filename'])
        df = pd.merge(image_df, df, on='id', how='inner')

    
        print("Copying Images")
        #Copy all images into database
        for filename in tqdm(os.listdir(new_images)[data_range[0]:data_range[1]]):
            source_file = os.path.join(new_images, filename)
            
                
            destination_file = os.path.join(f"{env}/database/images/", filename)
            
            # Only copy the file if it doesn't already exist in the destination folder
            if not os.path.exists(destination_file):
                shutil.copyfile(source_file, destination_file)
        
        
        #cols_to_remove = set(existing_df.columns) - set(df.columns)
        #existing_df = existing_df.drop(columns=cols_to_remove)
        cols_to_add = set(df.columns) - set(existing_df.columns)
        
        # Add the columns to df
        for col in cols_to_add:
            existing_df[col] = ''




        if enable_overwritting:
            # Append the new dataframe (df) to the existing dataframe (existing_df)
            combined_df = existing_df.append(df, ignore_index=True)

            # Remove duplicate rows
            combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        else:
            combined_df = merge_without_overwrite(existing_df, df)
             
    else: #Stage 2
        combined_df = MergeSess(existing_df, df)
        
            
        
        
    # Save the updated DataFrame back to the CSV file
    combined_df = combined_df.astype(str)
    combined_df['id'] = combined_df['id'].astype(int)
    combined_df = combined_df.sort_values(by='id')
        

    combined_df.to_csv(database_CSV, index=False, na_rep='')




def Store_Raw_Data():
    print("Storing Raw Data...")  
    #Finding index
    entry_index = 0
    while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
        entry_index += 1
    
    #Move data
    raw_folder = f"{env}/raw_data/entry_{entry_index}"

    shutil.copytree(os.path.join(env, "downloads"), raw_folder)

    # Remove the "downloads" folder and its contents
    shutil.rmtree(os.path.join(env, "downloads"))

    # Recreate the "downloads" folder as an empty folder
    os.mkdir(os.path.join(env, "downloads"))