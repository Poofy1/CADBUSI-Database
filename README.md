# CASBUSI Database

This program is designed to process Breast Ultrasound data from multiple sources and store it in a structured format, making it easy to manipulate, analyze, and prepare for machine learning training.

## Features

- Import breast ultrasound images with OCR capabilites
- Add or replace data based on the "id" or "anonymized_accession_num" fields
- Find images with similar features
- Prepare iamges for labelbox labeling
- Extract data from labelbox
- Extract and store requested data labels from different sources
- Perform validation splitting for machine learning training
- Supports re-parsing of existing data

## Installation

### Prerequisites

- Install Python 3.8
- Install required Python packages with pip:

```
pip install -r requirements.txt
```


## Configuration

### Parameters 
All input data will be controlled from the `main` script, you will need to configure the following parameters:

- `data_labels`: Define the labels you want to extract and store from the imported data. The labels must be specified in a dictionary where the keys represent the input file names and the values are lists of the desired labels. This will only accept `.json` files with the expected Mayo Clinic format and/or `.csv` files with any format. Each entry must include either `id` or `anonymized_accession_num`.
- `enable_overwritting`: Set this flag to `True` if you want and data to overwrite old data if they have the same `id` or `anonymized_accession_num` otherwise set to `False`
- `val_split`: Set the validation split ratio. A value of 0.2 means 20% of the accession studies will be allocated for validation.
- `images_per_row`: Determines how many images will be placed in each row in the cropped LabelBox images
- `LB_API_KEY`: LabelBox API Key
- `PROJECT_ID`: LabelBox Project ID

### Mode Select
- `only_append_to_database`: This mode allows you to add new or replace old data within the database.
- `only_reparse_raw_data`: This mode will reparse all old data with your current configuration.
- `only_update_val`: This mode will only update the validation column in the database.
- `only_retreive_labelbox_data`: This mode will retreive and add labeled LabelBox data to the database.
- Only one mode can be `True` at a time.

### Data Locations
- All input files must be held in the `downloads` folder, images must be stored in `downloads/images/`.
- Only new data rows will be added if the path `downloads/images/` exists and there is an image refrencing the data row
- The database will be held in the `database` folder
    - Original images: `database/images/`
    - LabelBox images: `database/labelbox_images/`
    - LabelBox image structure data: `database/crop_data.csv`
    - Pre-LabelBox data: `database/unlabeled_data.csv`
    - Final Labeled data: `database/label_data.csv`




### Configuration Example

```
data_labels = {
    "database.json": [
                    "id",
                    "anonymized_accession_num", 
                    "filename", 
                    "dicom_hash", 
                    "image_hash", 
                    "StudyDate", 
                    "StudyDescription",
                    "PhysicalDeltaX", 
                    "PhysicalDeltaY", 
                    "PatientAge", 
                    "ImageType",
                    "PhotometricInterpretation",
                    ],
    "labels.csv": [
                    "anonymized_accession_num", 
                    "Biopsy", 
                    "birads",
                    "RegionLocationMinX0", 
                    "RegionLocationMinY0", 
                    "RegionLocationMaxX1", 
                    "RegionLocationMaxY1", 
                    ],
}

# General Settings
enable_overwritting = True 
val_split = .2
images_per_row = 4  # LabelBox images row width
LB_API_KEY = ''
PROJECT_ID = ''


# Select Mode (Only one true at a time!)
only_append_to_database = True
only_reparse_raw_data = False
only_update_val = False

only_retreive_labelbox_data = False
```



## Usage

1. Run the `main` script to start the program:

```
python main.py
```
2. If you are adding to the database, the program will ask you if you want to continue at each step of the process. This allows you to skip steps if the process was previously interupted. The data steps are as follows:

    - `Tranform_JSON`: Process the input data files, convert them to csv, and add them to the database accordingly.
    - `OCR`: Read image text with OCR and organize the data.
    - `Similar_Images`: Find similar images and determine what images to exclude from training.
    - `Labelbox_Tranform`: Transform and crop images into clean LabelBox data.
    - `Move_Data`: Archives download folder into `Raw_Data`.
    - `Perform_Val`: Adds or edits the validation split column to the database.

![CASBUSI WORKFLOW](https://github.com/Poofy1/CASBUSI-Database/assets/70146048/9dcf3062-77df-42b7-a990-b239d7f175ab)



## License

This program is released under the MIT License. See the [LICENSE](LICENSE) file for more details.



