import os, json
import pandas as pd
#from images_to_labelbox import Crop_and_save_images
from compile_labelbox import Read_Labelbox_Data
from OCR import Perform_OCR
from trustworthiness import Find_Trust
from data_selection import Parse_Data, Rename_Images, Remove_Duplicate_Data, Remove_Bad_Images
#from get_labelbox_data import Read_Labelbox_Data
from export import Export_Database
from ML_processing.inpaint import Inpaint_Dataset
from ML_processing.orientation_detection import Find_Orientation
from dcm_parser import Parse_Zip_Files
from video_processing import ProcessVideoData, Video_Cleanup
env = os.path.dirname(os.path.abspath(__file__))

def load_config():
    with open(f'{env}/config.json', 'r') as config_file:
        config = json.load(config_file)
        return config
    
config = load_config()

########### Config ###########

# General Settings
val_split = .2

# Labelbox Settings
images_per_row = 4
LB_API_KEY = config['LABELBOX_API_KEY']
#PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu' # Image Annotation
PROJECT_ID = 'clp39jn0f07ub070meh4fbozh' # Instance Labeling

# Select Mode (Only one true at a time!)
only_append_to_database = False
only_retreive_labelbox_data = False
only_export = True

# Paths
zip_input = f'D:/DATA/CASBUSI/zip_files/'
raw_storage_database = f'D:/DATA/CASBUSI/dicoms/'
anon_location = "D:/DATA/CASBUSI/cases_anon/total_cases_anon.csv"
export_dir = f'D:/DATA/CASBUSI/exports/'
labelbox_path = f'D:/DATA/CASBUSI/labelbox_data/'
database_path = f'D:/DATA/CASBUSI/database_1_2_2024/'
#database_path = f'D:/DATA/CASBUSI/database/'

# Debug Settings 
data_range = None #[0,100] # Set to None to use everything
reseted_processed = False
only_determine_labeling = False

#############################



# Start Opterations
if __name__ == '__main__':
    if data_range is None:
        data_range = [0, 999999999999]

    
    if reseted_processed:
        input_file = f'{database_path}/ImageData.csv'
        df = pd.read_csv(input_file)
        df['processed'] = False
        df.to_csv(input_file, index=False)


    # Main Data Appender
    if only_append_to_database:
        
        user_input = input("Continue with DCM Parsing step? (y/n): ")
        if user_input.lower() == "y":
            Parse_Zip_Files(database_path, zip_input, anon_location, raw_storage_database, data_range)
            Find_Trust(database_path)
            
        
        user_input = input("Continue with OCR step? (y/n): ")
        if user_input.lower() == "y":
            Perform_OCR(database_path)
        
        user_input = input("Continue with Data Cleaning step? (y/n): ")
        if user_input.lower() == "y":
            Remove_Bad_Images(database_path)
            Remove_Duplicate_Data(database_path)
            Find_Orientation(f'{database_path}/images/', 'ori_model', f'{database_path}/ImageData.csv')
            Parse_Data(database_path, only_labels = False)
            Inpaint_Dataset(f'{database_path}/ImageData.csv', f'{database_path}/images/')
            Rename_Images(database_path)
        
        # Deprecated  
        """user_input = input("Continue with Labelbox_Tranform step? (y/n): ")
        if user_input.lower() == "y":
            Crop_and_save_images(images_per_row)"""
            
        user_input = input("Process Video Data? (y/n): ")
        if user_input.lower() == "y":
            ProcessVideoData(database_path)
            Video_Cleanup(database_path)

        
        
    # Export Database
    if only_export:
        Export_Database(export_dir, val_split, database_path, labelbox_path, reparse_images = False)
        
        
    if only_retreive_labelbox_data:
        print("(Newly created data in labelbox will take time to update!)")
        
        """
        # Path Config
        original_images = f"{database_path}/images/"

        Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images)"""
        
        Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, database_path, labelbox_path)
        
        
        
    if only_determine_labeling:
        Parse_Data(True)
        