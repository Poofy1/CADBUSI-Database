# CASBUSI Database
This database manager is designed to process breast ultrasound data from the Mayo Clinic and store it in a structured format, making it easy to manipulate, label, analyze, and prepare for machine learning training.

## Requirements
- Python 3.8
- 6GB Nvidia GPU (Recommended)
- At least 4 TB of hard drive space (Recommended)
- Install required Python packages with pip:
```
pip install -r requirements.txt
```


## Configuration
All user parameters will be controlled from a `config.json` file, you will need to configure the following parameters:

- `ZIPPED_DICOMS`: Directory of the zipped input data from Mayo Clinic.
- `UNZIPPED_DICOMS`: Directory of the raw unzipped data. 
- `ANON_FILE`: Location of the additional input `total_cases_anon.csv` file.
- `DATABASE_DIR`: Final location of the database.

- `LABELBOX_API_KEY`: Label Box API key for uploading and retrieving Label Box data.
- `PROJECT_ID`: LabelBox project ID.
- `LABELBOX_LABELS`: Directory of processed labels from Label Box.

- `EXPORT_DIR`: Output directory of all processed export data.
- `VAL_SPLIT`: Validation split ratio for splitting up training data.

- `DEBUG_DATA_RANGE`: (Default: `null`) Processed a reduced set of dicom files.
- `RESET_PROCESSED_FEILD`: (Default: `false`) Sets all images as 'unprocessed' withing the `ImageData.csv`.
- `REPROCESS_DATA_FILTERS`: (Default: `false`) Re-filters what will be included in the final export. 



## Usage / Modes
- Within the `main.py` script you must select ONE of the four tasks to complete. Each mode will conduct a specific task.

- `DEVELOP_DATABASE`: This process involves many steps and may take a significant amount of time to complete. In case of errors, checkpoints have been added to incrementally prompt the user which steps they need to process. The steps are as follows:
    - DCM Parsing: Processes the input dicom files by converting metadata to csv and export the images.
    - OCR: Reads the test description in the images with OCR and organizes the extracted data.
    - Data Cleaning (Part 1/2): Finds and removes corrupted images. Removes duplicate data. Uses machine learning to find orientations of unlabeled images.
    - Data Cleaning (Part 2/2): Filters what data will be used in the final export. Uses machine learning to inpaint calipers out of images. Renames all images to a specific format. 
    - Process Videos: Performs many of the operations we completed with image data with the video data instead.
- `DEVELOP_LABELBOX_DATA`: This process will prepare data to be uploaded to Label Box. 
- `RETREIVE_LABELBOX_DATA`: This process will retrieve and organize Label Box data to a directory.
- `DEVELOP_EXPORT`: This process will export all relevant database data and labeled data into the specified output directory.

After configuring the `main.py` file, run the script to start the program:
`python main.py`




## Database Architecture
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
  
### Database Output
- Exporting the data will organize and copy all relevant data to the specified directory (`export_dir`).
    - The format will be similar to the original database architecture but will only include the filtered data.
    - Additional `masks` folder.
    - Format and prepare data for training.

## Current Data Pipeline

1. Get the Datamart file of studies to be downloaded.
2. Manually import Notion files
    - Download studies as Dicom files.
    - Remove pixel-level patient info.
    - Number of patients starting from the previous batch.
    - Save the key file in a secure location.
3. (This Software) Data Processing
    - Parse data into CSV files.
    - Remove bad images.
    - Prepare data for labeling.
4. Upload data to Labelbox and label data.
5. (This Software) Retrieve labels and masks from Labelbox.


![CASBUSI WORKFLOW](https://github.com/Poofy1/CASBUSI-Database/assets/70146048/70594e4b-026e-4a0b-b544-7e1edb003ce1)
