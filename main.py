import os
import pandas as pd
from images_to_labelbox import Crop_and_save_images
from OCR import Perform_OCR
from trustworthiness import Find_Trust
from data_selection import Parse_Data, Rename_Images
from labelbox_to_images import Read_Labelbox_Data
from export import Export_Database
from ML_processing.inpaint import Inpaint_Dataset
from ML_processing.orientation_detection import Find_Orientation
from dcm_parser import Parse_Zip_Files
from video_processing import ProcessVideoData, Video_Cleanup
env = os.path.dirname(os.path.abspath(__file__))


########### Config ###########

# General Settings
val_split = .2
export_trust_ceiling = 2 #inclusive

# Labelbox Settings
images_per_row = 4
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGc5emFjOTIyMDZzMDcyM2E2MW0xbnpuIiwib3JnYW5pemF0aW9uSWQiOiJja290NnVvMWgxcXR0MHlhczNuNmlmZnRjIiwiYXBpS2V5SWQiOiJjbGh1dm5rMTAwYnV2MDcybjlpZ3g4NGdzIiwic2VjcmV0IjoiZmRhZjcxYzBhNDM3MmNkYWNkNWIxODU5MzUyNjc1ODMiLCJpYXQiOjE2ODQ1MTk4OTgsImV4cCI6MjMxNTY3MTg5OH0.DMecSgJDDZrX1qw2T4HLs5Sv62lLLT-ePcMjyxpn0aE'
PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu'

# Select Mode (Only one true at a time!)
only_append_to_database = False
only_retreive_labelbox_data = False
only_export = True

# Paths
zip_input = f'D:/DATA/CASBUSI/zip_files/'
raw_storage_database = f'D:/DATA/CASBUSI/dicoms/'
export_dir = f'D:/DATA/CASBUSI/exports/export_8_18/'

# Debug Settings 
data_range = None #[0,10000] # Set to None to use everything
reseted_processed = False

#############################



# Start Opterations
if __name__ == '__main__':
    if data_range is None:
        data_range = [0, 999999999999]

    
    if reseted_processed:
        input_file = f'{env}/database/ImageData.csv'
        df = pd.read_csv(input_file)
        df['processed'] = False
        df.to_csv(input_file, index=False)


    # Main Data Appender
    if only_append_to_database:
        
        user_input = input("Continue with DCM Parsing step? (y/n): ")
        if user_input.lower() == "y":
            Parse_Zip_Files(zip_input, raw_storage_database, data_range)
            Find_Trust()
            
        
        user_input = input("Continue with OCR step? (y/n): ")
        if user_input.lower() == "y":
            Perform_OCR()
        
        user_input = input("Continue with Data Cleaning step? (y/n): ")
        if user_input.lower() == "y":
            Find_Orientation(f'{env}/database/images/', 'ori_model', f'{env}/database/ImageData.csv')
            Parse_Data()
            Inpaint_Dataset(f'{env}/database/ImageData.csv', f'{env}/database/images/')
            Rename_Images()
        
        user_input = input("Continue with Labelbox_Tranform step? (y/n): ")
        if user_input.lower() == "y":
            Crop_and_save_images(images_per_row)
            
        user_input = input("Process Video Data? (y/n): ")
        if user_input.lower() == "y":
            ProcessVideoData()
            Video_Cleanup()

        
        
    # Export Database
    if only_export:
        Export_Database(export_trust_ceiling, export_dir, val_split)
        
        
    if only_retreive_labelbox_data:
        print("(Newly created data in labelbox will take time to update!)")

        # Path Config
        original_images = f"{env}/database/images/"

        Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images)
        