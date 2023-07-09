import os
from val_split import PerformVal
from images_to_selection import Crop_and_save_images
from pre_image_processing import Pre_Process, Perform_OCR
from selection_to_images import Read_Labelbox_Data
from ML_processing.inpaint import Inpaint_Dataset
from dcm_parser import Parse_Zip_Files, Transfer_Laterality
env = os.path.dirname(os.path.abspath(__file__))


########### Config ###########

# General Settings
enable_overwritting = True 
val_split = .2

# Labelbox Settings
images_per_row = 4
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGc5emFjOTIyMDZzMDcyM2E2MW0xbnpuIiwib3JnYW5pemF0aW9uSWQiOiJja290NnVvMWgxcXR0MHlhczNuNmlmZnRjIiwiYXBpS2V5SWQiOiJjbGh1dm5rMTAwYnV2MDcybjlpZ3g4NGdzIiwic2VjcmV0IjoiZmRhZjcxYzBhNDM3MmNkYWNkNWIxODU5MzUyNjc1ODMiLCJpYXQiOjE2ODQ1MTk4OTgsImV4cCI6MjMxNTY3MTg5OH0.DMecSgJDDZrX1qw2T4HLs5Sv62lLLT-ePcMjyxpn0aE'
PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu'

# Select Mode (Only one true at a time!)
only_append_to_database = True
only_retreive_labelbox_data = False
only_update_val = False

# Paths
zip_input = f'{env}/zip_files/'
raw_storage_database = f'D:/DATA/CASBUSI/dicoms/'

# Debug Settings 
data_range = None #[0, 2000] # Set to None to use everything

#############################





# Start Opterations
# Static vars
if __name__ == '__main__':
    if data_range is None:
        data_range = [0, 999999999999]

    
    # WIP Inpaint feature
    #Inpaint_Dataset(f'{env}/database/unlabeled_data.csv', f'{env}/database/images/', f'{env}/database/inpainted/')



    # Main Data Appender
    if only_append_to_database:
        
        user_input = input("Continue with DCM Parsing step? (y/n): ")
        if user_input.lower() == "y":
            Parse_Zip_Files(zip_input, raw_storage_database, data_range)
            
        
        
        
        user_input = input("Continue with OCR step? (y/n): ")
        if user_input.lower() == "y":
            Perform_OCR()
            # Transfer Laterality to CaseStudyData
            #Transfer_Laterality()
        
        
        
        user_input = input("Continue with Data_Selection step? (y/n): ")
        if user_input.lower() == "y":
            Pre_Process()
        
        
        
        user_input = input("Continue with Labelbox_Tranform step? (y/n): ")
        if user_input.lower() == "y":
            print("Transforming Images for Labelbox")
            Crop_and_save_images(images_per_row)
        
        
        
        # Update val split amount
        PerformVal(val_split)
        
        
        


    # Update val split amount
    if only_update_val:
        PerformVal(val_split)
        
        
    if only_retreive_labelbox_data:
        print("(Newly created data in labelbox will take time to update!)")

        # Path Config
        original_images = f"{env}/database/images/"

        Read_Labelbox_Data(LB_API_KEY, PROJECT_ID, original_images)
        