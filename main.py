import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.dicom_download import *
from src.query import *
from src.encrypt_keys import *
from src.query_clean_path import filter_path_data
from src.query_clean_rad import filter_rad_data
from src.filter_data import create_final_dataset

from src.DB_processing.image_processing import analyze_images
from src.DB_processing.data_selection import Select_Data, Remove_Duplicate_Data, Remove_Green_Images
from src.DB_processing.export import Export_Database
from src.DB_processing.dcm_parser import Parse_Dicom_Files
from src.DB_processing.video_processing import ProcessVideoData
from src.ML_processing.inpaint import Inpaint_Dataset
from src.ML_processing.inpaint_N2N import Inpaint_Dataset_N2N
from src.ML_processing.orientation_detection import Find_Orientation

from storage_adapter import * 
from config import CONFIG
import argparse
import os
import sys

env = os.path.dirname(os.path.abspath(__file__))



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DICOM processing pipeline')
    
    # Query arguments
    parser.add_argument('--query', action='store_true', help='Run breast imaging query')
    parser.add_argument('--limit', type=int, help='Optional limit for the query (e.g., 10)')
    
    # Download arguments
    parser.add_argument('--deploy', action='store_true', help='Deploy FastAPI to Cloud Run')
    parser.add_argument('--rerun', action='store_true', help='Send message to pre-deployed FastAPI on Cloud Run')
    parser.add_argument('--cleanup', action='store_true', help='Clean up resources')
    
    # Anonymize arguments
    parser.add_argument('--database', type=str, help='Directory name for anonymized DICOM output')
    parser.add_argument('--skip-inpaint', action='store_true', help='Skip the inpainting step')
    
    # Export arguments
    parser.add_argument('--export', action='store_true', help='Export current databse')
    
    return parser.parse_args()

def main():
    # Determine storage client
    StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    # Main entry point for the script
    args = parse_arguments()
    
    dicom_query_file = f'{env}/raw_data/endpoint_data.csv'
    key_output = f'{env}/encryption_key.pkl'
    output_path = os.path.join(env, "raw_data")
    
    # Handle query command
    if args.query:
        limit = args.limit
        
        # If no limit is specified, ask for confirmation
        if limit is None:
            confirmation = input("No limit specified. Are you sure you want to query without a limit? (y/n): ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Query cancelled.")
                sys.exit(0)
            print("Proceeding with unlimited query.")
        else:
            print(f"Setting query limit to {limit}")
        
        # Run the query with the specified limit
        rad_df, path_df = run_breast_imaging_query(limit=limit)

        # Parse that data
        rad_df = filter_rad_data(rad_df, output_path)
        path_df = filter_path_data(path_df, output_path)
        
        # Filter data
        create_final_dataset(rad_df, path_df, output_path)
    
    elif args.deploy or args.cleanup or args.rerun:
        dicom_download_remote_start(dicom_query_file, args.deploy, args.cleanup)
        
    elif args.database:
        anon_file = f'{env}/raw_data/anon_data.csv'
        BUCKET_PATH = f'{CONFIG["storage"]["download_path"]}/{args.database}'
        
        print(f"Starting database processing for {args.database}...")

        # Step 1: Encrypt IDs
        print("Step 1/5: Encrypting IDs...")
        key = encrypt_ids(dicom_query_file, anon_file, key_output)
        
        # Step 2: Parse DICOM files
        print("Step 2/5: Parsing and anonymizing DICOM files...")
        Parse_Dicom_Files(CONFIG["DATABASE_DIR"], 
                        anon_file, 
                        BUCKET_PATH, 
                        CONFIG["DEBUG_DATA_RANGE"],
                        encryption_key=key)
        
        # Step 3: Run OCR
        print("Step 3/5: Running OCR analysis...")
        analyze_images(CONFIG["DATABASE_DIR"])
        
        # Step 4: Clean data
        print("Step 4/5: Cleaning and processing image data...")
        Remove_Duplicate_Data(CONFIG["DATABASE_DIR"])
        Find_Orientation(CONFIG)
        Select_Data(CONFIG["DATABASE_DIR"], only_labels=False)
        
        # Make inpainting optional
        if not args.skip_inpaint:
            print("Running inpainting (can be skipped with --skip-inpaint)...")
            Inpaint_Dataset_N2N(f'{CONFIG["DATABASE_DIR"]}/ImageData.csv', 
                            f'{CONFIG["DATABASE_DIR"]}/images/')
        
        # Step 5: Process video
        print("Step 5/5: Processing video data...")
        ProcessVideoData(CONFIG["DATABASE_DIR"])

        
    elif args.export:
        Export_Database(CONFIG)
    
    else:
        print("No action specified. Use --help for available options.")

if __name__ == "__main__":
    main()