from src.DB_processing.image_processing import *
from src.DB_processing.database import DatabaseManager


def modify_keys(dictionary):
    # Create a new dictionary with modified keys
    modified_dictionary = {}
    for key, value in dictionary.items():
        modified_key = os.path.basename(key).rsplit('_', 1)[0]
        modified_dictionary[modified_key] = value
    return modified_dictionary


def single_video_region(base_path, image_path):
    target_path = os.path.join(base_path, image_path)
    # Read image and convert to grayscale
    image = read_image(target_path)
    if image is None:
        return None, None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x, y, w, h = process_crop_region(image)
    
    return image_path, (x, y, w, h)

def get_video_ultrasound_region(image_folder_path, first_images):

    # Collect image data in list
    image_data = []
    
    # Thread pool and TQDM
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        futures.update({executor.submit(single_video_region, image_folder_path, img_path): img_path 
                       for img_path in first_images})
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_data.append(result)
                pbar.update()
                
    image_masks_dict = {filename: mask for filename, mask in image_data}

    return image_masks_dict


def ProcessVideoData(database_path):

    with DatabaseManager(database_path) as db:
        video_folder_path = f"{database_path}/videos/"

        # Load data from database
        video_df = db.get_videos_dataframe()
        breast_df = db.get_study_cases_dataframe()

        # Check if there are no videos
        if len(video_df) == 0:
            print("No videos to process")
            return

        # Prepare column mapping for database field names
        video_df = video_df.rename(columns={
            'images_path': 'ImagesPath',
            'accession_number': 'Accession_Number',
            'patient_id': 'Patient_ID'
        })

        breast_df = breast_df.rename(columns={
            'accession_number': 'Accession_Number',
            'study_laterality': 'Study_Laterality'
        })

        db_to_process = video_df
        append_audit("video_processing.input_videos", len(db_to_process))

        print("Finding OCR Masks")
        _, description_masks = find_masks(video_folder_path, 'mask_model', db_to_process, 1920, 1080, video_format=True)
        append_audit("video_processing.extracted_description_masks", len(description_masks))

        print("Performing OCR")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(ocr_image, image_file, description_mask, video_folder_path): image_file for image_file, description_mask in description_masks}
            progress = tqdm(total=len(futures), desc='')

            # Initialize dictionary to store descriptions
            descriptions = {}

            for future in as_completed(futures):
                result = future.result()  # result is now a list with the filename and the description
                descriptions[result[0]] = result[1]  # Store description at corresponding image file
                progress.update()

            progress.close()

        valid_descriptions = sum(1 for desc in descriptions.values() if desc)
        append_audit("video_processing.extracted_ocr_descriptions", valid_descriptions)

        print("Finding Image Masks")
        first_images = get_first_image_in_each_folder(video_folder_path)
        # Filter first_images to only include those from db_to_process
        images_to_process = set(db_to_process['ImagesPath'].tolist())
        filtered_first_images = [img for img in first_images
                                if img.split('/')[0] in images_to_process]
        image_masks_dict = get_video_ultrasound_region(video_folder_path, filtered_first_images)
        valid_masks = sum(1 for mask in image_masks_dict.values() if mask is not None)
        append_audit("video_processing.extracted_crop_regions", valid_masks)

        descriptions = modify_keys(descriptions)
        image_masks_dict = modify_keys(image_masks_dict)

        # Initialize with defaults
        matched_descriptions = sum(1 for key in db_to_process['ImagesPath'] if key in descriptions)
        matched_masks = sum(1 for key in db_to_process['ImagesPath'] if key in image_masks_dict)
        db_to_process['description'] = None
        db_to_process['bounding_box'] = None

        # Map descriptions and masks
        if matched_descriptions > 0:
            desc_series = pd.Series(descriptions)
            mask = db_to_process['ImagesPath'].isin(descriptions.keys())
            db_to_process.loc[mask, 'description'] = db_to_process.loc[mask, 'ImagesPath'].map(desc_series)

        if matched_masks > 0:
            mask_series = pd.Series(image_masks_dict)
            mask = db_to_process['ImagesPath'].isin(image_masks_dict.keys())
            db_to_process.loc[mask, 'bounding_box'] = db_to_process.loc[mask, 'ImagesPath'].map(mask_series)

        # Handle bounding box extraction safely, row by row
        db_to_process['crop_x'] = None
        db_to_process['crop_y'] = None
        db_to_process['crop_w'] = None
        db_to_process['crop_h'] = None

        for idx, row in db_to_process.iterrows():
            bbox = row['bounding_box']
            if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    db_to_process.at[idx, 'crop_x'] = bbox[0]
                    db_to_process.at[idx, 'crop_y'] = bbox[1]
                    db_to_process.at[idx, 'crop_w'] = bbox[2]
                    db_to_process.at[idx, 'crop_h'] = bbox[3]
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Failed to extract bbox for row {idx}: {e}")

        # Handle feature extraction safely, row by row
        feature_columns = ['area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']

        for col in feature_columns:
            if col not in db_to_process.columns:
                db_to_process[col] = None

        for idx, row in db_to_process.iterrows():
            desc = row['description']
            if desc is not None:
                try:
                    features = extract_descript_features(desc, labels_dict=description_labels_dict)
                    if isinstance(features, dict):
                        for feature_name, feature_value in features.items():
                            if feature_name in db_to_process.columns:
                                db_to_process.at[idx, feature_name] = feature_value
                except Exception as e:
                    print(f"Failed to extract features for row {idx}: {e}")

        # Overwrite non bilateral cases with known lateralities
        laterality_mapping = breast_df[breast_df['Study_Laterality'].isin(['LEFT', 'RIGHT'])].set_index('Accession_Number')['Study_Laterality'].to_dict()

        db_to_process['laterality'] = db_to_process.apply(
            lambda row: laterality_mapping.get(row['Accession_Number']).lower()
            if row['Accession_Number'] in laterality_mapping
            else row['laterality'],
            axis=1
        )

        # Count unknown lateralities after correction
        unknown_lateralities = db_to_process[db_to_process['laterality'] == 'unknown'].shape[0]
        append_audit("video_processing.bilateral_with_unknown_lat", unknown_lateralities)

        # Update database with processed results
        cursor = db.conn.cursor()

        for _, row in db_to_process.iterrows():
            cursor.execute("""
                UPDATE Videos
                SET crop_x = ?, crop_y = ?, crop_w = ?, crop_h = ?,
                    laterality = ?, area = ?, orientation = ?
                WHERE images_path = ?
            """, (
                row.get('crop_x'),
                row.get('crop_y'),
                row.get('crop_w'),
                row.get('crop_h'),
                row.get('laterality'),
                row.get('area'),
                row.get('orientation'),
                row['ImagesPath']
            ))

        db.conn.commit()
        print(f"Updated {len(db_to_process)} videos in database")