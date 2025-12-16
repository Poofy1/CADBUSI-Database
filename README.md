# CADBUSI-Database

SQLite database manager that is designed to process breast ultrasound data from Mayo Clinic and store it in a structured format, making it easy to anonymize, manipulate, label, analyze, and prepare for ML research.

## Setup

- Create a GCP AI Factory instance with a T4 GPU
- Clone repository: `git clone https://github.com/Poofy1/CADBUSI-Database.git`
- Go into dir: `cd CADBUSI-Database`
- Install requirements: `pip install -r requirements.txt`
- Create/Configure: `./config.py`
- Obtain certificate `./src/dicom_downloader/CertEmulationCA.crt`
- Obtain user privileges: `gcloud auth application-default login --no-launch-browser`

## Configuration
All user parameters will be controlled from a `./config.py` file, you will need to configure the following parameters:
```
CONFIG = {
    # Environment configuration
    "env": {
        "project_id": "your-project-id",
        "region": "us-central1",
        "topic_name": "dicom-processing-topic",
        "subscription_name": "dicom-processing-subscription",
        "my_service_account": "your-service-id",
        "service_account_identity": "your-service-account@your-project-id.iam.gserviceaccount.com"
    },
    
    # Cloud Run configuration
    "cloud_run": {
        "service": "pubsub-push-cloudrun",
        "version": "1.0",
        "ar": "your-artifact-registry",
        "ar_name": "pubsub-push-cloudrun",
        "target_tag": "us-central1-docker.pkg.dev/your-project-id/your-artifact-registry/pubsub-push-cloudrun:1.0",
        "vpc_shared": "your-shared-vpc-id",
        "vpc_name": "your-vpc-name"
    },
    
    # Storage configuration
    "storage": {
        "gcs_log": "gs://your-bucket-name/cloudbuild_log",
        "gcs_stage": "gs://your-bucket-name/cloudbuild_stage",
        "bucket_name": "your-bucket-name",
        "download_path": "Downloads",
    },
    
    "BUCKET": "your-bucket-name",
    "WINDIR": "D:/DATA/YOUR_PROJECT/",
    "DATABASE_DIR": "Databases/database_YYYY_MM_DD/", # Final location of the database.

    "LABELBOX_API_KEY": "your-labelbox-api-key",
    "PROJECT_ID": "your-labelbox-project-id",
    "LABELBOX_LABELS": "labelbox_data/",
    "TARGET_CASES": "/failed_cases.csv", # Directory of worst performing cases from training. Prepares these cases for instance labeling on Label Box.
    "VIDEO_SAMPLING": 0, # every nth frame, 0 turns off videos
    "DEBUG_DATA_RANGE": [], # Process a reduced set of dicom files (Ex: [0, 1000]).
}
```
## Usage
The pipeline is operated through a single command-line interface in main.py, which provides several functions. For general purpose, you should perform these commands in this order: 

```
python main.py --query [optional: --limit=N]
python main.py --deploy
python main.py --cleanup
python main.py --database
python main.py --export
```



### Querying Data

To query breast imaging data:
`python main.py --query [optional: --limit=N]`

This will:
1. Run a query to retrieve breast imaging records
2. Filter and clean the radiology and pathology data
3. Create a final dataset for processing
4. Save results to `query_data/endpoint_data.csv`

Example with a limit:`python main.py --query --limit=100`

#### Query Diagram `--query [optional: limit=N]`
![CADBUSI Query](/demo/CADBUSI_Query.png)

### Downloading DICOM Files

The tool offers Cloud Run deployment for efficient DICOM downloads. Dicoms will appear in specified GCP bucket storage:
```
# Deploy the FastAPI service to Cloud Run and start dicom data download (REQUIRED)
python main.py --deploy

# Resend the download requests to the pre-deployed service (OPTIONAL)
python main.py --rerun 

# Clean up Cloud Run resources when finished (REQUIRED)
python main.py --cleanup
```

IMPORTANT: After `python main.py --deploy` finishes execution, that does not mean the data transfer is complete. The download requests have been sent to Cloud Run. Check the bucket storage to see when population is finished. Only then should you run `python main.py --cleanup`

### Processing DICOM Files

To process the downloaded DICOM files into a complete database:

`python main.py --database [optional: --skip-inpaint]`

This will:
1. Generate encryption keys for safely anonymizing patient IDs
2. Deidentify DICOM files from `CONFIG['storage']['download_path']`
3. Process image files in the specified output directory in the destination bucket
4. `[optional: --skip-inpaint]` will skip the caliper removal process

Example:

`python main.py --database`

### Export Database

To process the downloaded DICOM files into a complete database:

`python main.py --export`

This will:
1. Crop all relevent images / videos into output dir
2. Create a consolidated label system



## Data Architecture
### Database
- The final database will be held in the specified `DATABASE_DIR` folder with this internal layout:
    - `/database/images/`: Raw image storage
        - Any caliper image that qualified to be used in an export has been replaced with an inpainted version of itself.
    - `/database/videos/`: Contains a separate folder for each video, each one with the first and middle frame of the video. 
    - `/database/LossLabeling/`: Contains all images for Label Box labeling. (Labeling instance labels)
    - `/database/LossLabelingReferences.csv`: LabelBox image data structure for retrieving and cross referencing data.
        - This is database specific! You must build and retrieve labels from Label Box using the same database. 
    - `/database/database.db`: SQLite metadata database

### Labeled Data
- The labeled data will be held in the specified `LABELBOX_LABELS` folder with this internal layout:
    - `/labelbox_data/InstanceLabels.csv`: Recorded instance labels from Label Box. This data is universal across databases as it includes the dicom `FileName` for each instance.
    - If there exists a boolean column named `Reject Image`, this will be used to ignore the specified image when exporting the database. This column will be excluded on export. 
    - Labeled instance data will referenced and stored using the dicom data hash ID `'DicomHash'`, generated from the unanonymized raw dicom file.

### Exports
- Exporting will create a new folder in the appropriate database directory `DATABASE_DIR`, with todays date on it so that it does not overwrite previous exports.
    - The format will be similar to the original database architecture but will only include relevant data.
    - If there is labeled instance data inside the `LABELBOX_LABELS` dir then these will be added in the export as well.
    - `/export_12_26_2023/TrainData.csv`: This file contains refrences to the data formatted into bags for the [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training) project to easily interpret.



## Data Pipeline
- [CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database)
- [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training)
