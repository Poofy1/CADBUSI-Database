# CASBUSI Data Parser

This program is designed to process Breast Ultrasound data from multiple sources and store it in a structured format, making it easy to manipulate and analyze the data. The program also supports validation splitting.

## Features

- Imports Breast Ultrasound data from different sources
- Refrence data from "id" or "anonymized_accession_num"
- Extracts and stores requested data labels
- Performs validation splitting for machine learning tasks
- Supports re-parsing of existing data

## Installation

### Prerequisites

- Python 3.8
- Tesseract OCR (Download from [here](https://github.com/UB-Mannheim/tesseract/wiki))
- Install required Python packages with pip:

```
pip install -r requirements.txt
```

### Configuration

In the `main` script, you need to configure the following parameters:

- `set_tesseract_path`: Set the path to the Tesseract-OCR executable file.
- `val_split`: Set the validation split ratio. A value of 0.2 means 20% of the data will be allocated for validation.
- `data_labels`: Define the labels you want to extract and store from the imported data. The labels must be specified in a dictionary where the keys represent the input file names and the values are lists of the desired labels.
- `reparse_data`: Set this flag to `True` if you want to re-parse existing data, otherwise set it to `False`.

## Usage

1. Run the `main` script to start the program:

```
python main.py
```

2. The program will process the data and create a new entry in the `raw_data` folder. The entry will be named `entry_x`, where `x` is the index of the new entry.

3. The program will create a duplicate database with the specified validation split in the `val_data` folder.


## License

This program is released under the MIT License. See the [LICENSE](LICENSE) file for more details.



