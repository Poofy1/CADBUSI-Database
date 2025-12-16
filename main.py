import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src.encrypt_keys import encrypt_ids

from tools.storage_adapter import * 
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
    parser.add_argument('--limit', type=int, help='Optional limit for the query/export')
    
    # Download arguments
    parser.add_argument('--deploy', action='store_true', help='Deploy FastAPI to Cloud Run')
    parser.add_argument('--rerun', action='store_true', help='Send message to pre-deployed FastAPI on Cloud Run')
    parser.add_argument('--cleanup', action='store_true', help='Clean up resources')
    
    # Anonymize arguments
    parser.add_argument('--database', action='store_true', help='Process database')
    parser.add_argument('--skip-inpaint', action='store_true', help='Skip the inpainting step')

    # Export arguments
    parser.add_argument('--export', action='store_true', help='Export current databse')
    
    return parser.parse_args()

def main():
    # Determine storage client
    storage = StorageClient.get_instance(CONFIG["WINDIR"], CONFIG["BUCKET"])
    
    # Main entry point for the script
    args = parse_arguments()
    OUTPUT_PATH = os.path.join(env, "data")
    DICOM_QUERY_PATH = os.path.join(OUTPUT_PATH, 'endpoint_data.csv')
    DATABASE_LOCAL_PATH = os.path.join(OUTPUT_PATH, 'cadbusi.db')
    DATABASE_GCP_PATH = f'{CONFIG["DATABASE_DIR"]}/cadbusi.db'
    
    
    # Handle query command
    if args.query:
        from src.data_ingest.query import run_breast_imaging_query
        from src.data_ingest.clean_pathology import filter_path_data
        from src.data_ingest.clean_radiology import filter_rad_data
        from src.data_ingest.filter_data import create_final_dataset
        limit = args.limit
        
        # Run the query with the specified limit
        rad_df, path_df = run_breast_imaging_query(limit=limit)

        # Parse that data
        rad_df = filter_rad_data(rad_df, OUTPUT_PATH)
        path_df = filter_path_data(path_df, OUTPUT_PATH)
        
        # Filter data
        create_final_dataset(rad_df, path_df, OUTPUT_PATH)
    
    elif args.deploy or args.cleanup or args.rerun:
        from src.dicom_downloader.dicom_download import dicom_download_remote_start
        dicom_download_remote_start(DICOM_QUERY_PATH, args.deploy, args.cleanup)
        
    elif args.database:
        from src.DB_processing.image_processing import analyze_images
        from src.DB_processing.data_selection import Select_Data
        from src.DB_processing.dcm_parser import Parse_Dicom_Files
        from src.DB_processing.video_processing import ProcessVideoData
        from src.ML_processing.lesion_detection import Locate_Lesions
        from src.ML_processing.inpaint_N2N import Inpaint_Dataset_N2N
        from src.ML_processing.orientation_detection import Find_Orientation
        from src.ML_processing.caliper_coordinates import Locate_Calipers
        from src.ML_processing.download_models import download_models
        
        lesion_pathology = f'{env}/data/lesion_pathology.csv'
        lesion_anon_file = f'{env}/data/lesion_anon_data.csv'
        anon_file = f'{env}/data/anon_data.csv'
        key_output = f'{env}/encryption_key.pkl'
        BUCKET_PATH = f'{CONFIG["storage"]["download_path"]}/'
        
        if os.path.exists(DATABASE_LOCAL_PATH):
            print("WARNING: Previous database still exists, this can cause issues! Recommend rebuilding from scratch")
        
        print(f"Starting database processing for {BUCKET_PATH}...")
        download_models() # Download all models

        # Step 1: Encrypt IDs
        print("Step 1/5: Encrypting IDs...")
        key = encrypt_ids(DICOM_QUERY_PATH, anon_file, key_output)
        key = encrypt_ids(lesion_pathology, lesion_anon_file, key_output)
        
        # Step 2: Parse DICOM files
        Parse_Dicom_Files(CONFIG, anon_file, lesion_anon_file, BUCKET_PATH, encryption_key=key)
        
        # Step 3: Run OCR
        print("Step 3/5: Processing image data...")
        analyze_images(CONFIG["DATABASE_DIR"])
        
        # Step 4: Clean data
        print("Step 4/5: Cleaning image data...")
        #Find_Orientation(CONFIG) # Unnessesary and unreliable 
        Select_Data(CONFIG["DATABASE_DIR"])
        
        # Make inpainting optional
        if not args.skip_inpaint:
            print("Running inpainting (can be skipped with --skip-inpaint)...")
            Inpaint_Dataset_N2N( f'{CONFIG["DATABASE_DIR"]}/images/')
            
        Locate_Lesions(f'{CONFIG["DATABASE_DIR"]}/images/')
        Locate_Calipers(f'{CONFIG["DATABASE_DIR"]}/images/')
        
        # Step 5: Process video
        print("Step 5/5: Processing video data...")
        ProcessVideoData(CONFIG["DATABASE_DIR"])
        
        #Upload Database
        if storage.is_gcp:
            blob = storage._bucket.blob(DATABASE_GCP_PATH.replace('//', '/').rstrip('/'))
            local_full_path = os.path.join(storage.windir, DATABASE_LOCAL_PATH) if storage.windir else DATABASE_LOCAL_PATH
            blob.upload_from_filename(local_full_path)
            print('Database uploaded')

    elif args.export:
        from src.DB_export.export import Export_Database
        #Download Database
        if storage.is_gcp:
            local_full_path = os.path.join(storage.windir, DATABASE_LOCAL_PATH) if storage.windir else DATABASE_LOCAL_PATH
            # Delete local database if it exists
            if os.path.exists(DATABASE_LOCAL_PATH):
                os.remove(local_full_path)
            
            # Download from GCP
            blob = storage._bucket.blob(DATABASE_GCP_PATH.replace('//', '/').rstrip('/'))
            os.makedirs(os.path.dirname(local_full_path), exist_ok=True)
            blob.download_to_filename(local_full_path)
            print(f'Database downloaded to {local_full_path}')
            
        if args.limit:
            print(f"Exporting with limit: {args.limit}")
        Export_Database(CONFIG, limit=args.limit)
    
    else:
        print("No action specified. Use --help for available options.")

if __name__ == "__main__":
    main()