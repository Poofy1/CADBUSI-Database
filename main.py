import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os, json
import pandas as pd
#from LB_processing.create_labelbox_data import Create_Labelbox_Data
#rom LB_processing.retreive_labelbox_data import Read_Labelbox_Data
from DB_processing.OCR import Perform_OCR
from DB_processing.trustworthiness import Find_Trust
from DB_processing.data_selection import Parse_Data, Rename_Images, Remove_Duplicate_Data, Remove_Green_Images
from DB_processing.export import Export_Database
from DB_processing.dcm_parser import Parse_Dicom_Files
from DB_processing.video_processing import ProcessVideoData, Video_Cleanup
from ML_processing.inpaint import Inpaint_Dataset
from ML_processing.inpaint_N2N import Inpaint_Dataset_N2N
from ML_processing.orientation_detection import Find_Orientation
env = os.path.dirname(os.path.abspath(__file__))
from storage_adapter import * 



def load_config():
    config_path = f'{env}/config.json'
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print("Config does not exist, creating file...")
        
        # Create default config with empty strings and default values
        default_config = {
            "BUCKET": "",
            "WINDIR": "",
            "UNZIPPED_DICOMS": "",
            "ANON_FILE": "",
            "DATABASE_DIR": "",
            
            "LABELBOX_API_KEY": "",
            "LABELBOX_LABELS": "",
            "TARGET_CASES": "",
            
            "EXPORT_DIR": "",
            "VAL_SPLIT": 0.2,
            
            "DEBUG_DATA_RANGE": [],
            "RESET_PROCESSED_FEILD": False,
            "REPROCESS_DATA_FILTERS": False
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config to file
        with open(config_path, 'w') as config_file:
            json.dump(default_config, config_file, indent=4)
    
    # Load config (either existing or newly created)
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        return config
    
CONFIG = load_config()

# Define the tasks
TASKS = {
    1: "Develop Database",
    2: "Develop Labelbox Data",
    3: "Retrieve Labelbox Data",
    4: "Develop Export"
}

# Start Operations
if __name__ == '__main__':
    
    # Determine storage client
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    if CONFIG["DEBUG_DATA_RANGE"] is None:
        CONFIG["DEBUG_DATA_RANGE"] = [0, 999999999999]

    if CONFIG["RESET_PROCESSED_FEILD"]:
        input_file = f'{CONFIG["DATABASE_DIR"]}/ImageData.csv'
        df = pd.read_csv(input_file)
        df['processed'] = False
        df.to_csv(input_file, index=False)

    # Present the user with the list of tasks
    print("Available Tasks:")
    for task_num, task_name in TASKS.items():
        print(f"{task_num}. {task_name}")

    # Get user input for the task number
    task_num = int(input("Enter the task number to start: "))

    # Execute the selected task
    if task_num == 1:  # Develop Database
        user_input = input("Continue with DCM Parsing step? (y/n): ")
        if user_input.lower() == "y":
            Parse_Dicom_Files(CONFIG["DATABASE_DIR"], CONFIG["ANON_FILE"], CONFIG["UNZIPPED_DICOMS"], CONFIG["DEBUG_DATA_RANGE"])
            Find_Trust(CONFIG["DATABASE_DIR"])
        
        user_input = input("Continue with OCR step? (y/n): ")
        if user_input.lower() == "y":
            Perform_OCR(CONFIG["DATABASE_DIR"])
        
        user_input = input("Continue with Data Cleaning step (Part 1/2)? (y/n): ")
        if user_input.lower() == "y":
            Remove_Green_Images(CONFIG["DATABASE_DIR"])
            Remove_Duplicate_Data(CONFIG["DATABASE_DIR"])
            Find_Orientation(f'{CONFIG["DATABASE_DIR"]}/images/', 'ori_model', f'{CONFIG["DATABASE_DIR"]}/ImageData.csv')
            
        user_input = input("Continue with Data Cleaning step (Part 2/2)? (y/n): ")
        if user_input.lower() == "y":
            Parse_Data(CONFIG["DATABASE_DIR"], only_labels = False)
            #Inpaint_Dataset(f'{CONFIG["DATABASE_DIR"]}/ImageData.csv', f'{CONFIG["DATABASE_DIR"]}/images/') # OLD and SLOW
            Inpaint_Dataset_N2N(f'{CONFIG["DATABASE_DIR"]}/ImageData.csv', f'{CONFIG["DATABASE_DIR"]}/images/')
            Rename_Images(CONFIG["DATABASE_DIR"])
            
        user_input = input("Process Video Data? (y/n): ")
        if user_input.lower() == "y":
            ProcessVideoData(CONFIG["DATABASE_DIR"])
            Video_Cleanup(CONFIG["DATABASE_DIR"])

        """
    elif task_num == 2:  # Develop Labelbox Data
        Create_Labelbox_Data(CONFIG["TARGET_CASES"], CONFIG["DATABASE_DIR"])

    elif task_num == 3:  # Retrieve Labelbox Data
        Read_Labelbox_Data(CONFIG["LABELBOX_API_KEY"], CONFIG["PROJECT_ID"], CONFIG["DATABASE_DIR"], CONFIG["LABELBOX_LABELS"])
        """
    
    elif task_num == 4:  # Develop Export
        Export_Database(CONFIG, reparse_images = False)

    if CONFIG["REPROCESS_DATA_FILTERS"]:
        Parse_Data(CONFIG["DATABASE_DIR"], only_labels = True)