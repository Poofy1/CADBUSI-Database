import pandas as pd
import labelbox, requests, os, glob, sys

# Get the directory of env
current_dir = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(env)

# Now you can import your script
import images_to_labelbox

def add_new_columns(input_csv, csv_files):
    # Read the input CSV
    input_df = pd.read_csv(input_csv)

    # Get the column names of the input CSV
    input_columns = input_df.columns

    for file in csv_files:
        df = pd.read_csv(file)

        # Get the column names of the current CSV file
        file_columns = df.columns

        # Find the new columns in the input CSV
        new_columns = input_columns.difference(file_columns)

        # Merge the input dataframe with the current dataframe based on 'Patient_ID'
        df = pd.merge(df, input_df[list(new_columns) + ['Patient_ID']], on='Patient_ID', how='left')

        # Save the updated CSV file
        df.to_csv(file, index=False)
        
        
def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID):
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)

    print("Contacting Labelbox")
    export_url = project.export_labels()

    # Download the export file from the provided URL
    response = requests.get(export_url)

    # Parse Data from labelbox
    print("Parsing Labelbox Data")
    return images_to_labelbox.Get_Labels(response)


# Get current Labelbox data
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGc5emFjOTIyMDZzMDcyM2E2MW0xbnpuIiwib3JnYW5pemF0aW9uSWQiOiJja290NnVvMWgxcXR0MHlhczNuNmlmZnRjIiwiYXBpS2V5SWQiOiJjbGh1dm5rMTAwYnV2MDcybjlpZ3g4NGdzIiwic2VjcmV0IjoiZmRhZjcxYzBhNDM3MmNkYWNkNWIxODU5MzUyNjc1ODMiLCJpYXQiOjE2ODQ1MTk4OTgsImV4cCI6MjMxNTY3MTg5OH0.DMecSgJDDZrX1qw2T4HLs5Sv62lLLT-ePcMjyxpn0aE'
PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu'
input_csv = Read_Labelbox_Data(LB_API_KEY, PROJECT_ID)

# Add any new columns from the input CSV to each of the CSV files
path = f'{env}/labeled_data_archive/*.csv'
csv_files = glob.glob(path)
add_new_columns(input_csv, csv_files)
