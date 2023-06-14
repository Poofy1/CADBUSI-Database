import os
from data_parser import set_tesseract_path, PerformEntry, Store_Raw_Data
from val_split import PerformVal
from images_to_selection import Crop_and_save_images
from pre_image_processing import Pre_Process, Perform_OCR
import shutil
from selection_to_images import Read_Labelbox_Data

########### Config ###########
set_tesseract_path("C:/Users/Tristan/AppData/Local/Tesseract-OCR/tesseract.exe")

#List of all labels to use:
data_labels = {
    "database.json": [
                    "id",
                    "filename", 
                    "dicom_hash", 
                    "image_hash", 
                    "anonymized_accession_num", 
                    "biopsy", 
                    "birads",
                    "RegionLocationMinX0", 
                    "RegionLocationMinY0", 
                    "RegionLocationMaxX1", 
                    "RegionLocationMaxY1", 
                    "StudyDate", 
                    "StudyDescription",
                    "PhysicalDeltaX", 
                    "PhysicalDeltaY", 
                    "PatientAge", 
                    "ImageType",
                    "PhotometricInterpretation",
                    ],
}


# General Settings
enable_overwritting = True 
val_split = .2
images_per_row = 4  # LabelBox images row width
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGc5emFjOTIyMDZzMDcyM2E2MW0xbnpuIiwib3JnYW5pemF0aW9uSWQiOiJja290NnVvMWgxcXR0MHlhczNuNmlmZnRjIiwiYXBpS2V5SWQiOiJjbGh1dm5rMTAwYnV2MDcybjlpZ3g4NGdzIiwic2VjcmV0IjoiZmRhZjcxYzBhNDM3MmNkYWNkNWIxODU5MzUyNjc1ODMiLCJpYXQiOjE2ODQ1MTk4OTgsImV4cCI6MjMxNTY3MTg5OH0.DMecSgJDDZrX1qw2T4HLs5Sv62lLLT-ePcMjyxpn0aE'
PROJECT_ID = 'clgr3eeyn00tr071n6tjgatsu'


# Select Mode (Only one true at a time!)
only_append_to_database = True
only_reparse_raw_data = False
only_update_val = False

only_retreive_labelbox_data = False

# Add progressive steps




#############################

# Start Opterations
# Static vars
env = os.path.dirname(os.path.abspath(__file__))
image_input = f"{env}/downloads/images/"
image_output = f"{env}/database/labelbox_images/"
input_csv = f"{env}/database/unlabeled_data.csv"
output_csv = f"{env}/database/crop_data.csv"

if only_append_to_database:
    #Finding index
    entry_index = 0
    while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
        entry_index += 1
    print(f"Adding Entry {entry_index} to Database")
    
    
    
    user_input = input("Continue with Tranform_JSON step? (y/n): ")
    if user_input.lower() == "n":
        print("Skipping PerformEntry step")
    else:
        PerformEntry('downloads', data_labels, only_reparse_raw_data, enable_overwritting)
    
    
    
    user_input = input("Continue with OCR step? (y/n): ")
    if user_input.lower() == "n":
        print("Skipping OCR step")
    else:
        print("Generating Image OCR Data")
        Perform_OCR()
    
    
    
    user_input = input("Continue with Similar_Images step? (y/n): ")
    if user_input.lower() == "n":
        print("Skipping Similar_Images step")
    else:
        print("Finding Similar Images")
        Pre_Process()
    
    
    
    user_input = input("Continue with Labelbox_Tranform step? (y/n): ")
    if user_input.lower() == "n":
        print("Skipping Labelbox_Tranform step")
    else:
        print("Transforming Images for Labelbox")
        Crop_and_save_images(input_csv, image_input, output_csv, image_output, images_per_row)
    
    # Move data
    #Store_Raw_Data()
    
    # Update val split amount
    PerformVal(val_split)
    
    
if only_reparse_raw_data:
    if os.path.exists(f"{env}/database/"):
        shutil.rmtree(f"{env}/database/")
    for index, entry in enumerate(os.listdir(f"{env}/raw_data/"), start=0):
        print(f"\nAdding Entry {index}")
        entry_path = f'raw_data/{entry}'
        PerformEntry(entry_path, data_labels, only_reparse_raw_data, enable_overwritting)
        
        print("Transforming CSV for Image Processing")
        Pre_Process()
    
        #Label box prep
        print(f"Transforming Images For Labeling")
        entry_path = f'{env}/{entry_path}/images/'
        Crop_and_save_images(input_csv, entry_path, output_csv, image_output, images_per_row)
    
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
    