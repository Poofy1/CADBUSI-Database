# CASBUSI Database

This database manager is designed to process breast ultrasound data from the Mayo Clinic and store it in a structured format, making it easy to manipulate, label, analyze, and prepare for machine learning training.

## Features

- Append new data to the database
- Edit or replace data within the database
- Detailed data selection
- Prepare images for Labelbox labeling
- Extract data from Labelbox
- Package export data for machine learning training

## Installation

### Prerequisites

- Python 3.8
- 6GB Nvidia GPU (Recommended)
- Install required Python packages with pip:

```
pip install -r requirements.txt
```


## Configuration

### Parameters 
All input data will be controlled from the `main` script, you will need to configure the following parameters:

- `export_trust_ceiling`: This is a threshold that will only export the studies that have a high enough trust rating.
- `val_split`: Set the validation split ratio. A value of 0.2 means 20% of the accession studies will be allocated for validation.
- `images_per_row`: Determines how many images will be placed in each row in the cropped LabelBox images
- `LB_API_KEY`: LabelBox API Key
- `PROJECT_ID`: LabelBox Project ID
- `zip_input`: This is where you will place the input zip files for new data.
- `anon_location`: This is where you will place the additional input `total_cases_anon.csv` file.
- `raw_storage_database`: This is where all the raw data will be stored. This will require a lot of drive space at this location.
- `export_dir`: This is where all export data will be placed.

### Mode Select
- `only_append_to_database`: This mode allows you to add new or replace old data within the database.
- `only_retreive_labelbox_data`: This mode will retrieve and add labeled LabelBox data to the database.
- `only_export`: This will export all relevant database and labeled data into the specified output folder (`export_dir`).
- Only one mode can be `True` at a time.

### Data Input
- All input zipped files must be in the `zip_input` folder variable.
- An additional CSV file is required for extra metadata. The location should be specified in the `anon_location` variable.

### Database Architecture
- The output database will be held in the `database` folder.
    - Images: `database/images/`
    - Videos: `database/videos/`
    - LabelBox images: `database/labelbox_images/`
    - LabelBox image structure data: `database/CropData.csv`
    - Study Based data: `database/CaseStudyData.csv`
    - Image Based data: `database/ImageData.csv`
    - Video Based data: `database/VideoData.csv`
    - Image Counter Tracker: `database/IndexCounter.txt`
    - Processed Data Tracker: `database/ParsedFiles.txt`
- The labeled data will be held in the `labeled_data_archive` folder.
    - This will include a `masks` folder from segmentation.
    - Retrieving data from Labelbox will create a new labeled data entry into the `labeled_data_archive` directory.
  
### Data Output
- Exporting the data will organize and copy all relevant data to the specified directory (`export_dir`).
    - The format will be similar to the original database architecture but will only include the filtered data.
    - Additional `masks` folder.
    - Format and prepare data for training.


## Usage

1. After configuring the `main.py` file, run the script to start the program:

```
python main.py
```
2. While adding new data to the database, the program will ask you if you want to continue at each step of the process. This allows you to skip steps if the process was previously interrupted. The data steps are as follows:

    - `DCM Parsing`: Process the input data files, convert them to csv, and add them to the database accordingly.
    - `OCR`: Read image text with OCR and organize the data.
    - `Similar_Images`: Find similar images and determine what images to exclude from training.
    - `Labelbox_Tranform`: Transform and crop images into clean LabelBox data.
    - `Perform_Val`: Adds or edits the validation split column to the database.

![CASBUSI WORKFLOW](https://github.com/Poofy1/CASBUSI-Database/assets/70146048/70594e4b-026e-4a0b-b544-7e1edb003ce1)




## Current Pipeline

1. Get Datamart file of studies to be downloaded
2. Manually import Notion files
    - Download studies as dicom files
    - Remove pixel level patient info
    - Number patients starting from previous batch
    - Save key file in secure location
3. (This Software) Data Processing
    - Parse data into CSV files
    - Remove bad images
    - Prepare data for labeling
4. Upload data to Labelbox and label data
5. (This Software) Retrieve label and masks from Labelbox



## License

This program is released under the MIT License. See the [LICENSE](LICENSE) file for more details.



