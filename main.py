import os
from data_parser import set_tesseract_path, PerformEntry
from val_split import PerformVal
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


reparse_data = False
enable_overwritting = True 
val_split = .2


#############################




# Start Opterations

env = os.path.dirname(os.path.abspath(__file__))

if not reparse_data:
    #Finding index
    entry_index = 0
    while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
        entry_index += 1
    print(f"Adding Entry {entry_index} to Database")
    
    PerformEntry('downloads', data_labels, reparse_data, enable_overwritting)
else:
    if os.path.exists(f"{env}/database/"):
        shutil.rmtree(f"{env}/database/")
    for index, entry in enumerate(os.listdir(f"{env}/raw_data/"), start=0):
        print(f"\nAdding Entry {index}")
        entry = f'raw_data/{entry}'
        PerformEntry(entry, data_labels, reparse_data, enable_overwritting)


#Create duplicate database with validation split
PerformVal(val_split)
