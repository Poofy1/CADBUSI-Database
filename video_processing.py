from OCR import *


def modify_keys(dictionary):
    # Create a new dictionary with modified keys
    modified_dictionary = {}
    for key, value in dictionary.items():
        modified_key = os.path.basename(key).rsplit('_', 1)[0]
        modified_dictionary[modified_key] = value
    return modified_dictionary


def get_ultrasound_region(image_folder_path, db_to_process):
    # Construct image paths for only the new data
    video_folders = [os.path.join(image_folder_path, filename) for filename in db_to_process['ImagesPath']]
    
    # Collect image data in list
    image_data = []
    
    # Thread pool and TQDM
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for video_folder in video_folders:
            image_paths = get_first_image_in_each_folder(video_folder)
            futures.update({executor.submit(process_single_image, image_path): image_path for image_path in image_paths})
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_data.append(result)
                pbar.update()
                
    image_masks_dict = {filename: mask for filename, mask in image_data}

    return image_masks_dict


def ProcessVideoData(database_path):
    
    video_folder_path = f"{database_path}/videos/"
    input_file = f'{database_path}/VideoData.csv'
    db_out = pd.read_csv(input_file)

    # Check if any new features are missing in db_out and add them
    new_features = ['crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']
    missing_features = set(new_features) - set(db_out.columns)
    for nf in missing_features:
        db_out[nf] = None
    
    

    # Check if 'processed' column exists, if not, create it and set all to False
    if 'processed' not in db_out.columns:
        db_out['processed'] = False

    # Only keep rows where 'processed' is False
    db_to_process = db_out[db_out['processed'] != True]
    db_to_process['processed'] = False
    
    print("Finding OCR Masks")
    video_folders = [os.path.join(video_folder_path, filename) for filename in db_to_process['ImagesPath']]
    _, description_masks = find_masks(video_folder_path, 'mask_model', db_to_process, 1920, 1080, video_format=True, video_folders=video_folders)

    print("Performing OCR")
    first_images = get_first_image_in_each_folder(video_folder_path)

    # Separate image names and description masks into their own lists
    image_names = [dm[0] for dm in description_masks]
    description_masks_coords = [dm[1] for dm in description_masks]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, image_file, description_mask, video_folder_path, reader, description_kw): image_file for image_file, description_mask in zip(first_images, description_masks_coords)}
        progress = tqdm(total=len(futures), desc='')

        # Initialize dictionary to store descriptions
        descriptions = {}

        for future in as_completed(futures):
            result = future.result()  # result is now a list with the filename and the description
            descriptions[result[0]] = result[1]  # Store description at corresponding image file
            progress.update()

        progress.close()

    print("Finding Image Masks")
    image_masks_dict = get_ultrasound_region(video_folder_path, db_to_process)
    
    db_to_process['processed'] = True
    
    descriptions = modify_keys(descriptions)
    image_masks_dict = modify_keys(image_masks_dict)

    # Convert dictionaries to Series for easy mapping
    descriptions_series = pd.Series(descriptions)
    image_masks_series = pd.Series(image_masks_dict)

    # Update dataframe using map
    db_to_process['description'] = db_to_process['ImagesPath'].map(descriptions_series)
    db_to_process['bounding_box'] = db_to_process['ImagesPath'].map(image_masks_series)
    
    db_to_process[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = pd.DataFrame(db_to_process['bounding_box'].tolist(), index=db_to_process.index)

    # Construct a temporary DataFrame with the feature extraction
    temp_df = db_to_process.loc[db_to_process['description'].str.len() > 0, 'description'].apply(lambda x: extract_descript_features(x, labels_dict=description_labels_dict)).apply(pd.Series)

    for column in temp_df.columns:
        db_to_process[column] = temp_df[column]

    db_out.update(db_to_process, overwrite=True)
    db_out.to_csv(input_file,index=False)


def Video_Cleanup(database_path):
    
    print("Video Data Clean Up")
    
    input_file = f'{database_path}/VideoData.csv'
    db = pd.read_csv(input_file)
    
    #Replace unknown areas with breast
    db.loc[(db['area'] == 'unknown') | (db['area'].isna()), 'area'] = 'breast'
    
    # Find crop ratio
    db['crop_aspect_ratio'] = db['crop_w'] / db['crop_h']
    
    db.to_csv(input_file, index=False)


