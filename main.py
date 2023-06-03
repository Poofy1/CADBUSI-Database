import os
from data_parser import set_tesseract_path, PerformEntry
from val_split import PerformVal
from images_to_selection import Crop_and_save_images
import shutil

########### Config ###########
set_tesseract_path("C:/Users/Tristan/AppData/Local/Tesseract-OCR/tesseract.exe")

#List of all labels to use:
data_labels = {
    "database.json": [
                    "id",
                    "filename", 
                    #"dicom_hash", 
                    #"image_hash", 
                    "anonymized_accession_num", 
                    #"biopsy", 
                    #"birads",
                    #"RegionLocationMinX0", 
                    #"RegionLocationMinY0", 
                    #"RegionLocationMaxX1", 
                    #"RegionLocationMaxY1", 
                    #"StudyDate", 
                    #"StudyDescription",
                    #"PhysicalDeltaX", 
                    #"PhysicalDeltaY", 
                    #"PatientAge", 
                    #"ImageType",
                    #"PhotometricInterpretation",
                    ],
    "labels.csv": [
                    "anonymized_accession_num", 
                    "Biopsy", 
                    "birads",
                    #"RegionLocationMinX0", 
                    #"RegionLocationMinY0", 
                    #"RegionLocationMaxX1", 
                    #"RegionLocationMaxY1", 
                    ],
}


reparse_data = True
enable_overwritting = True 
val_split = 0.2


#LabelBox images row width:
images_per_row = 4

#############################


#Orgainize image/Csv Data
#Creating labelbox images
#Extracting labelbox images



# Start Opterations
# Static vars
env = os.path.dirname(os.path.abspath(__file__))
image_input = f"{env}/downloads/images/"
image_output = f"{env}/labelbox_data/labelbox_images/"
input_csv = f"{env}/image_input/database_total_v4.csv"
output_csv = f"{env}/labelbox_data/crop_data.csv"

if not reparse_data:
    #Finding index
    entry_index = 0
    while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
        entry_index += 1
    print(f"Adding Entry {entry_index} to Database")
    
    PerformEntry('downloads', data_labels, reparse_data, enable_overwritting)
    
    #Label box prep
    print("\nTransforming Images for Labeling")
    Crop_and_save_images(input_csv, image_input, output_csv, image_output, images_per_row)
else:
    if os.path.exists(f"{env}/database/"):
        shutil.rmtree(f"{env}/database/")
    for index, entry in enumerate(os.listdir(f"{env}/raw_data/"), start=0):
        print(f"\nAdding Entry {index}")
        entry_path = f'raw_data/{entry}'
        PerformEntry(entry_path, data_labels, reparse_data, enable_overwritting)
        
        #Label box prep
        print(f"Transforming For Labeling")
        entry_path = f'{env}/{entry_path}/images/'
        Crop_and_save_images(input_csv, entry_path, output_csv, image_output, images_per_row)


#Create duplicate database with validation split
PerformVal(val_split)