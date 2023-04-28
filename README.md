# CASBUSI Data Parser

This program is designed to process Breast Ultrasound data from multiple sources and store it in a structured format, making it easy to manipulate, analyze the data, and prepare for machine learning training.

## Features

- Importing Breast Ultrasound images with OCR capabilites
- Adds or replaces data based on the "id" or "anonymized_accession_num" fields
- Extracts and stores requested data labels from different sources
- Performs validation splitting for machine learning training
- Supports re-parsing of existing data

## Installation

### Prerequisites

- Python 3.8
- Tesseract OCR (Download from [here](https://github.com/UB-Mannheim/tesseract/wiki))
- Install required Python packages with pip:

```
pip install -r requirements.txt
```

In the `main` script, you need to configure the following parameter:
- `set_tesseract_path`: Set the path to the Tesseract-OCR executable file. (Found in `./AppData/Local/Tesseract-OCR/`)




## Input Data Requirements

### Configuration 

All input data will be controlled from the `main` script, you will need to configure the following parameters:

- `val_split`: Set the validation split ratio. A value of 0.2 means 20% of the accession studies will be allocated for validation.
- `data_labels`: Define the labels you want to extract and store from the imported data. The labels must be specified in a dictionary where the keys represent the input file names and the values are lists of the desired labels. This will only accept `.json` files with the expected Mayo Clinic format and/or `.csv` files with any format. Each entry must include either `id` or `anonymized_accession_num`.
- `reparse_data`: Set this flag to `True` if you want to re-parse existing data held in `raw_data`, otherwise to add data set it to `False`. Re-parsing will conform to the provided `data_labels` entry. Include all file formats, any missing files will be ignored while cycling through the `raw_data` entries.
- `enable_overwritting`: Set this flag to `True` if you want and data to overwrite old data if they have the same `id` or `anonymized_accession_num` otherwise set to `False`

All input files must be held in the `downloads` folder, images must be stored in `downloads/images/`.
Only new data rows will be added if the path `downloads/images/` exists and there is an image refrencing the data row

### Example Inputs

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
```




## Output Data Format

After processing, the program will generate the following outputs:

- `database/data.csv`: A CSV file containing the processed data with the specified labels.
- `database/train.csv`: A duplicate of the `database.csv` file with an additional `valid` column indicating the validation split.
- `database/images/`: A folder which will hold unmodified images that were used extract OCR information.


## Usage

1. Run the `main` script to start the program:

```
python main.py
```
2. The program will create or add to an existing database, according to the specified `data_labels`.

3. The program will move all data present in the `downloads` folder into the `raw_data` folder. The entry will be named `entry_x`, where `x` is the index of the new entry.

4. The program will create a duplicate database file (`train.csv`) with the specified validation split column `valid`. Marking `0` as training data and `1` as validation data


## License

This program is released under the MIT License. See the [LICENSE](LICENSE) file for more details.



