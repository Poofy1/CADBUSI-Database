from src.DB_processing.image_processing import *
from src.DB_processing.tools import append_audit
import concurrent.futures
from functools import partial
tqdm.pandas()

def fetch_index_for_patient_id( id, db):
    # id is a patient id number that should be listed in database
    # returns list of indices
    
    if id in db['Patient_ID'].tolist():
         indices= db.index[db['Patient_ID']==id].tolist()
    else:
        indices = []
    return indices

def add_labeling_categories(db):
    db['label_cat'] = ''

    for idx, row in db.iterrows():
        if row['label']:
            orient = row['orientation']
            image_type = row['PhotometricInterpretation']
            if image_type == 'RGB':
                label_cat = 'doppler'
            elif orient in ['trans', 'long']:
                label_cat = orient
            else:
                label_cat = 'other'
            
            db.at[idx, 'label_cat'] = label_cat

    return db


def choose_images_to_label(db, breast_df, database_path):
    initial_count = len(db)
    db['label'] = True
    
    #Remove images that are too dark
    darkness_thresh = 65
    # Extract all darkness values rounded to 2 decimals
    darkness_values = db['darkness'].round(2).tolist()
    append_audit("image_processing.darkness_values", darkness_values)
    append_audit("image_processing.darkness_thresh", darkness_thresh)
    
    dark_count_before = len(db[db['label']])
    db.loc[db['darkness'] > 65, 'label'] = False
    dark_count_after = len(db[db['label']])
    dark_removed = dark_count_before - dark_count_after
    
    
    # loop over caliper rows and tag twin images
    caliper_count_before = len(db[db['label']])
    caliper_rows = db[db['has_calipers']]
    for idx, row in caliper_rows.iterrows():
        distance = row['distance']
        if distance <= 5:
            db.at[idx,'label'] = False
    caliper_count_after = len(db[db['label']])
    caliper_removed = caliper_count_before - caliper_count_after
    
    # set label = False for all non-breast images
    area_count_before = len(db[db['label']])
    db.loc[(db['area'] == 'unknown') | (db['area'].isna()), 'area'] = 'breast'
    db.loc[(db['area'] != 'breast'), 'label'] = False
    area_count_after = len(db[db['label']])
    area_removed = area_count_before - area_count_after
    
    # Set label = False for all images with 'unknown' laterality
    lat_count_before = len(db[db['label']])
    db.loc[(db['laterality'] == 'unknown') | (db['laterality'].isna()), 'label'] = False
    lat_count_after = len(db[db['label']])
    lat_removed = lat_count_before - lat_count_after
    
    # Set label = False for all images with 'RegionCount' > 1
    region_count_before = len(db[db['label']])
    db.loc[db['RegionCount'] > 1, 'label'] = False
    region_count_after = len(db[db['label']])
    region_removed = region_count_before - region_count_after
    
    # Check the aspect ratio of the crop region
    db['crop_aspect_ratio'] = (db['crop_w'] / db['crop_h']).round(2)
    
    # Create an audit log for all the filtering operations
    append_audit("image_processing.too_dark_removed", dark_removed)
    append_audit("image_processing.caliper_issues_removed", caliper_removed) # same as caliper_with_duplicates?
    append_audit("image_processing.non_breast_removed", area_removed)
    append_audit("image_processing.unknown_lat_removed", lat_removed)
    append_audit("image_processing.multi_region_removed", region_removed)
    append_audit("image_processing.usable_images", len(db[db['label']]))
    
    total_caliper_images = len(db[db['has_calipers']])
    append_audit("image_processing.total_caliper_images", total_caliper_images)
    caliper_with_duplicates = len(db[(db['has_calipers']) & (db['distance'] <= 5)])
    append_audit("image_processing.caliper_with_duplicates", caliper_with_duplicates)
    total_near_duplicates = len(db[db['distance'] <= 5]) 
    append_audit("image_processing.total_near_duplicates", total_near_duplicates)
    
    
    return db





def find_nearest_images(db, patient_id, image_folder_path):
    subset = db[db['PhotometricInterpretation'] != 'RGB']
    # Validate crop coordinates
    invalid_coords = (
        (subset['RegionLocationMaxX1'] <= subset['RegionLocationMinX0']) |
        (subset['RegionLocationMaxY1'] <= subset['RegionLocationMinY0'])
    )
    
    if invalid_coords.any():
        subset = subset[~invalid_coords]
    
    if len(subset) == 0:
        print(f"Patient {patient_id}: No valid images after filtering")
        return {}
    
    idx = np.array(fetch_index_for_patient_id(patient_id, subset))
    result = {}
    image_pairs_checked = set()

    # Precompute cropping coordinates
    coord_cols = ['RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1']
    crop_coords = subset[coord_cols].astype(int)
    crop_coords['w'] = crop_coords['RegionLocationMaxX1'] - crop_coords['RegionLocationMinX0']
    crop_coords['h'] = crop_coords['RegionLocationMaxY1'] - crop_coords['RegionLocationMinY0']

    # Preload and process all images
    img_dict = {}
    for image_id in idx:
        file_name = subset.loc[image_id, 'ImageName']
        full_filename = os.path.join(image_folder_path, file_name)
        img = read_image(full_filename, use_pil=True)
        img = np.array(img).astype(np.uint8)
        img_dict[image_id] = img

    for j, c in enumerate(idx):
        if c in image_pairs_checked:
            continue

        x, y, w, h = crop_coords.loc[c, ['RegionLocationMinX0', 'RegionLocationMinY0', 'w', 'h']]

        img_list = []
        for image_id in idx:
            img = img_dict[image_id]
            (rows, cols) = img.shape[:2]
            if rows >= y + h and cols >= x + w:
                img, _ = make_grayscale(img)
                img = img[y:y+h, x:x+w]
            else:
                img = np.full((h, w), 255, dtype=np.uint8)
            img_list.append(img.flatten())

        img_stack = np.array(img_list, dtype=np.uint8)
        img_stack = np.abs(img_stack - img_stack[j, :])
        img_stack = np.mean(img_stack, axis=1)
        img_stack[j] = 1000
        sister_image = np.argmin(img_stack)
        distance = img_stack[sister_image]

        result[c] = {
            'image_filename': subset.at[c, 'ImageName'],
            'sister_filename': subset.at[idx[sister_image], 'ImageName'],
            'distance': distance
        }

        if idx[sister_image] not in result:
            result[idx[sister_image]] = {
                'image_filename': subset.at[idx[sister_image], 'ImageName'],
                'sister_filename': subset.at[c, 'ImageName'],
                'distance': distance
            }

        image_pairs_checked.add(c)
        image_pairs_checked.add(idx[sister_image])

    return result


def process_patient_id(pid, db_out, image_folder_path):
    subset = db_out[db_out['Patient_ID'] == pid]
    
    result = find_nearest_images(subset, pid, image_folder_path)
    idxs = result.keys()
    for i in idxs:
        subset.at[i, 'closest_fn'] = result[i]['sister_filename']
        subset.at[i, 'distance'] = result[i]['distance']
    return subset

def Remove_Green_Images(database_dir):
    print("Searching for corrupted images")
    input_file = f'{database_dir}/ImageData.csv'
    
    # Load the CSV file into a pandas DataFrame
    df = read_csv(input_file)

    image_folder_path = f"{database_dir}/images/"
    
    # Prepare a list of indices to drop from the DataFrame
    drop_indices = []
    
    # Iterate over the rows of the dataframe using tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_name = row['ImageName']
        image_path = os.path.join(image_folder_path, image_name)
        
        # Read the image using OpenCV
        img = read_image(image_path)

        # If the image is RGB
        if img is not None and len(img.shape) == 3:
            b, g, r = cv2.split(img)
            
            # Calculate mean values for each channel
            mean_b = np.mean(b)
            mean_g = np.mean(g)
            mean_r = np.mean(r)
            
            # Check if average of green channel is significantly higher than the others
            if mean_g > mean_r + 10 and mean_g > mean_b + 10:
                print(f'BAD IMAGE: {image_path}')
                os.remove(image_path)  # Delete the image file
                drop_indices.append(index)

    # Drop rows from the DataFrame
    df = df.drop(drop_indices)
    append_audit("image_processing.corrupted_removed", len(drop_indices))
    save_data(df, input_file)  # Save the updated DataFrame back to the CSV




def create_caliper_file(database_path, image_df, breast_df, max_workers=None):
    """
    Creates a separate caliper file with specific columns for images that have calipers
    and copies both the caliper images and raw images to a new directory using read_image and save_data.
    
    Args:
        database_path (str): Path to the database directory
        max_workers (int): Maximum number of threads (defaults to CPU count)
    """
    caliper_output_file = f'{database_path}/CaliperData.csv'
    images_dir = f'{database_path}/images'
    caliper_images_dir = f'{database_path}/caliper_pairs'
    
    # Filter for images that have calipers
    caliper_images = image_df[
        (image_df['has_calipers'] == True) & 
        (image_df['distance'] < 5) &
        (image_df['distance'] >= 0)
    ].copy()
    
    if caliper_images.empty:
        print("No images with calipers found.")
        return
    
    # Create the caliper dataframe with required columns
    caliper_df = pd.DataFrame()
    caliper_df['Patient_ID'] = caliper_images['Patient_ID']
    caliper_df['Accession_Number'] = caliper_images.get('Accession_Number', '')
    caliper_df['Distance'] = caliper_images['distance']
    caliper_df['Caliper_Image'] = caliper_images['ImageName']
    caliper_df['Raw_Image'] = caliper_images['closest_fn']
    
    # Merge with breast data to get Has_Malignant information
    caliper_df = caliper_df.merge(
        breast_df[['Accession_Number', 'Has_Malignant']], 
        on='Accession_Number', 
        how='left'
    )
    
    # Reorder columns to match your specification
    column_order = ['Patient_ID', 'Accession_Number', 'Has_Malignant', 'Distance', 'Raw_Image', 'Caliper_Image']
    caliper_df = caliper_df[column_order]
    
    # Function to copy both caliper and raw images for a single row
    def copy_images_for_row(row, images_dir, caliper_images_dir):
        success_count = 0
        
        # Copy caliper image
        try:
            caliper_image_path = f"{images_dir}/{row['Caliper_Image']}"
            image = read_image(caliper_image_path)
            if image is not None:
                caliper_filename = os.path.basename(caliper_image_path)
                caliper_dest_path = f"{caliper_images_dir}/{caliper_filename}"
                save_data(image, caliper_dest_path)
                success_count += 1
            else:
                print(f"Failed to read caliper image: {caliper_image_path}")
        except Exception as e:
            print(f"Error copying caliper image {caliper_image_path}: {e}")
        
        # Copy raw image
        try:
            raw_image_path = f"{images_dir}/{row['Raw_Image']}"
            image = read_image(raw_image_path)
            if image is not None:
                raw_filename = os.path.basename(raw_image_path)
                raw_dest_path = f"{caliper_images_dir}/{raw_filename}"
                save_data(image, raw_dest_path)
                success_count += 1
            else:
                print(f"Failed to read raw image: {raw_image_path}")
        except Exception as e:
            print(f"Error copying raw image {raw_image_path}: {e}")
        
        return success_count
    
    # Copy both caliper and raw images using ThreadPoolExecutor
    total_copied = 0
    copy_func = partial(copy_images_for_row, images_dir=images_dir, caliper_images_dir=caliper_images_dir)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Convert DataFrame rows to list for ThreadPoolExecutor
        rows = [row for _, row in caliper_df.iterrows()]
        
        # Submit all tasks and track progress
        future_to_row = {executor.submit(copy_func, row): row for row in rows}
        
        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_row), 
                          desc="Copying Caliper and Raw Images", 
                          total=len(caliper_df)):
            total_copied += future.result()
    
    # Save the caliper file
    save_data(caliper_df, caliper_output_file)
    
    print(f"Successfully copied {total_copied} images total ({len(caliper_df)} caliper + {len(caliper_df)} raw images expected)")
    
    return caliper_df


def Select_Data(database_path, only_labels):
    input_file = f'{database_path}/ImageData.csv'
    breast_file = f'{database_path}/BreastData.csv'
    image_folder_path = f"{database_path}/images/"
    db_out = read_csv(input_file)
    breast_df = read_csv(breast_file)
    
    # Check if 'processed' column exists, if not create it
    if 'processed' not in db_out.columns:
        db_out['processed'] = False

    # Remove rows with missing data in crop_x, crop_y, crop_w, crop_h
    rows_before = len(db_out)
    db_out.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'], inplace=True)
    rows_after = len(db_out)
    append_audit("image_processing.missing_crop_removed", rows_before - rows_after)

    if only_labels:
        db_to_process = db_out
    else:
        # Filter the rows where 'processed' is False
        db_to_process = db_out[db_out['processed'] == False]

        print("Finding Similar Images")
        patient_ids = db_to_process['Patient_ID'].unique()

        db_to_process['closest_fn']=''
        db_to_process['distance'] = -1

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(patient_ids), desc='') as progress:
            futures = {executor.submit(process_patient_id, pid, db_to_process, image_folder_path): pid for pid in patient_ids}

            for future in as_completed(futures):
                result = future.result()
                if result is not None and not result.empty:
                    db_to_process.update(result)
                progress.update()

    db_to_process = choose_images_to_label(db_to_process, breast_df, database_path)
    db_to_process = add_labeling_categories(db_to_process)
    
    # Update 'processed' status to True for processed rows and merge back to the original dataframe
    db_to_process['processed'] = True
    # List all columns that are common to both dataframes
    common_columns = db_out.columns.intersection(db_to_process.columns)

    # Set 'ImageName' as the key for merging
    key_column = 'ImageName'

    # Take in all rows from db_to_process and only rows from db_out that aren't already present.
    db_out = pd.merge(db_to_process, db_out[~db_out[key_column].isin(db_to_process[key_column])], on=common_columns.tolist(), how='outer')

    if 'latIsLeft' in db_out.columns:
        db_out = db_out.drop(columns=['latIsLeft'])

    save_data(db_out, input_file)
    
    create_caliper_file(database_path, db_out, breast_df)


def Remove_Duplicate_Data(database_path):
    print("Removing Duplicate Data")
    
    input_file = f'{database_path}/ImageData.csv'
    image_folder_path = f"{database_path}/images/"
    
    # Read the CSV file
    df = read_csv(input_file)
    
    # Identify rows with duplicate 'DicomHash' values, except for the last occurrence
    duplicates = df[df.duplicated(subset='DicomHash', keep='last')]
    
    # Get the count of duplicate rows
    duplicate_count = len(duplicates)
    
    # Extract the image names of the duplicate rows
    duplicate_image_names = duplicates['ImageName'].tolist()
    
    # Remove the duplicate rows from the dataframe
    df.drop_duplicates(subset='DicomHash', keep='last', inplace=True)
    
    # Save the cleaned dataframe back to the CSV file
    save_data(df, input_file)
    append_audit("image_processing.duplicates_removed", duplicate_count)
    
    # Delete the duplicate images
    for image_name in tqdm(duplicate_image_names):
        image_path = os.path.join(image_folder_path, image_name)
        if file_exists(image_path):
            delete_file(image_path)
    
    # Print the number of duplicates removed
    print(f"Removed {duplicate_count} duplicate rows")