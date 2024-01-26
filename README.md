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

- `DEBUG_DATA_RANGE`: (Default: `null`) Process a reduced set of dicom files (Ex: [0, 1000]).
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



## Data Architecture
### Database
- The final database will be held in the specified `DATABASE_DIR` folder with this internal layout:
    - `/database/images/`: Raw image storage
        - Any caliper image that qualified to be used in an export has been replaced with an inpainted version of itself.
    - `/database/videos/`: Contains a separate folder for each video, each one with the first and middle frame of the video. 
    - `/database/LossLabeling/`: Contains all images for Label Box labeling. (Labeling instance labels)
    - `/database/LossLabelingReferences.csv`: LabelBox image data structure for retrieving and cross referencing data.
        - This is database specific! You must build and retrieve labels from Label Box using the same database. 
    - `/database/CaseStudyData.csv`: Study Based data.
    - `/database/ImageData.csv`: Image Based data.
    - `/database/VideoData.csv`: Video Based data.
    - `/database/IndexCounter.txt`: Index tracker for reading and appending new dicom data to the database.
    - `/database/ParsedFiles.txt`: List of dicom files that were already processed.

### Labeled Data
- The labeled data will be held in the specified `LABELBOX_LABELS` folder with this internal layout:
    - `/labelbox_data/InstanceLabels.csv`: Recorded instance labels from Label Box. This data is universal across databases as it includes the dicom `FileName` for each instance.

### Exports
- Exporting will create a new folder in the specified directory `EXPORT_DIR`.
    - The format will be similar to the original database architecture but will only include relevant data.
    - `/export_12_26_2023/TrainData.csv`: This file contains refrences to the data formatted into bags for the [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training) project to easily interpret.



## Current Data Pipeline

![CASBUSI Pipeline](CADBUSI-Pipeline.png)
