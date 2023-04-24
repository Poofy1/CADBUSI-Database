import os
from data_parser import set_tesseract_path, PerformEntry
from val_split import PerformVal


########### Config ###########
set_tesseract_path("C:/Users/Tristan/AppData/Local/Tesseract-OCR/tesseract.exe")

val_split = .2

#List of all labels to use:
data_labels = {
    "database.json": [
                    "id",
                    "filename", 
                    "dicom_hash", 
                    "image_hash", 
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


# Experimental 
reparse_data = False


#############################




# Start Opterations

env = os.path.dirname(os.path.abspath(__file__))

if not reparse_data:
    #Finding index
    entry_index = 0
    while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
        entry_index += 1
    print(f"Adding Entry {entry_index} to Database")
    
    PerformEntry('downloads', data_labels, reparse_data)
else:
    for index, entry in enumerate(os.listdir(f"{env}/raw_data/"), start=1):
        print(f"\nAdding Entry {index}")
        entry = f'raw_data/{entry}'
        PerformEntry(entry, data_labels, reparse_data)


#Create duplicate database with validation split
PerformVal(val_split)
