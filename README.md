# CASBUSI Database

This program is designed to process Breast Ultrasound data from multiple sources and store it in a structured format, making it easy to manipulate, analyze, and prepare for machine learning training.

## Features

- Import breast ultrasound images with OCR capabilites
- Add or replace data based on the `id` or `anonymized_accession_num` fields
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
- 6GB Nvidia GPU (Recommended)

```
pip install -r requirements.txt
```


## Configuration

### Parameters 
All input data will be controlled from the `main` script, you will need to configure the following parameters:

- `enable_overwritting`: Set this flag to `True` if you want data to overwrite old data if they have the same `id` or `anonymized_accession_num` otherwise set to `False` (Currently not functioning)
- `val_split`: Set the validation split ratio. A value of 0.2 means 20% of the accession studies will be allocated for validation.
- `images_per_row`: Determines how many images will be placed in each row in the cropped LabelBox images
- `LB_API_KEY`: LabelBox API Key
- `PROJECT_ID`: LabelBox Project ID
- `zip_input`: This is where you will place the input zip files for new data.
- `raw_storage_database`: This is where all the raw data will be stored. This will require a lot of drive space at this location.

### Mode Select
- `only_append_to_database`: This mode allows you to add new or replace old data within the database.
- `only_update_val`: This mode will only update the validation column in the database.
- `only_retreive_labelbox_data`: This mode will retrieve and add labeled LabelBox data to the database.
- Only one mode can be `True` at a time.

### Data Locations
- All input files must be in the `zip_input` folder.
- The output database will be held in the `database` folder
    - Original images: `database/images/`
    - LabelBox images: `database/labelbox_images/`
    - LabelBox image structure data: `database/CropData.csv`
    - Study Based data: `database/CaseStudyData.csv`
    - Image Based data: `database/ImageData.csv`
    - Image Counter Tracker: `database/IndexCounter.txt`
    - Processed Data Tracker: `database/ParsedFiles.txt`



## Usage

1. Run the `main` script to start the program:

```
python main.py
```
2. If you are adding to the database, the program will ask you if you want to continue at each step of the process. This allows you to skip steps if the process was previously interupted. The data steps are as follows:

    - `DCM Parsing`: Process the input data files, convert them to csv, and add them to the database accordingly.
    - `OCR`: Read image text with OCR and organize the data.
    - `Similar_Images`: Find similar images and determine what images to exclude from training.
    - `Labelbox_Tranform`: Transform and crop images into clean LabelBox data.
    - `Perform_Val`: Adds or edits the validation split column to the database.

![CASBUSI WORKFLOW](https://github.com/Poofy1/CASBUSI-Database/assets/70146048/70594e4b-026e-4a0b-b544-7e1edb003ce1)



## License

This program is released under the MIT License. See the [LICENSE](LICENSE) file for more details.



