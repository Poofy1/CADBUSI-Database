from src.DB_processing.image_processing import *
from src.DB_processing.tools import append_audit
import concurrent.futures
from functools import partial
tqdm.pandas()

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





def find_nearest_images(subset, image_folder_path):
    if len(subset) == 0:
        return {}
    
    idx = subset.index.to_numpy()
    result = {}
    image_pairs_checked = set()

    # All regions have same coordinates - get them once
    coord_cols = ['RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1']
    x, y, x1, y1 = subset.iloc[0][coord_cols].astype(int)
    w, h = x1 - x, y1 - y

    # Load and crop all images once
    cropped_images = {}
    for image_id in idx:
        file_name = subset.loc[image_id, 'ImageName']
        full_filename = os.path.join(image_folder_path, file_name)
        img = read_image(full_filename, use_pil=True)
        img = np.array(img).astype(np.uint8)
        img, _ = make_grayscale(img)
        
        # Crop once
        rows, cols = img.shape[:2]
        if rows >= y + h and cols >= x + w:
            cropped = img[y:y+h, x:x+w]
        else:
            cropped = np.full((h, w), 255, dtype=np.uint8)
        
        cropped_images[image_id] = cropped.flatten()

    # Convert to matrix for vectorized operations
    image_ids = list(cropped_images.keys())
    image_matrix = np.array([cropped_images[id] for id in image_ids], dtype=np.uint8)
    
    # Vectorized distance computation
    for j, current_id in enumerate(image_ids):
        if current_id in image_pairs_checked:
            continue
        
        # Compute distances to all images at once
        current_img = image_matrix[j:j+1]  # Keep 2D for broadcasting
        distances = np.mean(np.abs(image_matrix - current_img), axis=1)
        distances[j] = 1000  # Exclude self
        
        sister_idx = np.argmin(distances)
        sister_id = image_ids[sister_idx]
        distance = distances[sister_idx]
        
        # Store results for both images
        result[current_id] = {
            'image_filename': subset.at[current_id, 'ImageName'],
            'sister_filename': subset.at[sister_id, 'ImageName'],
            'distance': distance
        }
        
        if sister_id not in result:
            result[sister_id] = {
                'image_filename': subset.at[sister_id, 'ImageName'],
                'sister_filename': subset.at[current_id, 'ImageName'],
                'distance': distance
            }
        
        image_pairs_checked.add(current_id)
        image_pairs_checked.add(sister_id)
    
    return result


def process_nearest_given_ids(pid, db_out, image_folder_path):
    # Filter by accession number first
    subset = db_out[db_out['Accession_Number'] == pid]
    
    # EARLY EXIT:
    subset = subset[subset['PhotometricInterpretation'] != 'RGB']
    
    # Validate crop coordinates
    invalid_coords = (
        (subset['RegionLocationMaxX1'] <= subset['RegionLocationMinX0']) |
        (subset['RegionLocationMaxY1'] <= subset['RegionLocationMinY0'])
    )
    if invalid_coords.any():
        subset = subset[~invalid_coords]
    
    # Early termination if no valid images
    if len(subset) == 0:
        return subset
    
    # Group by crop coordinates
    coord_cols = ['RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1']
    
    # Create coordinate groups
    subset['coord_key'] = subset[coord_cols].apply(lambda row: tuple(row), axis=1)
    coordinate_groups = subset.groupby('coord_key')
    
    #print(f"Accession {pid}: {len(subset)} regions split into {len(coordinate_groups)} coordinate groups")
    
    # Process each coordinate group separately
    all_results = {}
    
    for coord_key, group_subset in coordinate_groups:
        # Check if group has multiple images AND at least one has calipers
        has_calipers_in_group = group_subset['has_calipers'].any()
        
        if len(group_subset) >= 2 and has_calipers_in_group:
            group_result = find_nearest_images(group_subset, image_folder_path)
            all_results.update(group_result)
        else:
            # Single image in group - no comparison possible
            single_idx = group_subset.index[0]
            subset.at[single_idx, 'closest_fn'] = ''
            subset.at[single_idx, 'distance'] = 99999
    
    # Apply all results back to subset
    for i in all_results.keys():
        subset.at[i, 'closest_fn'] = all_results[i]['sister_filename']
        subset.at[i, 'distance'] = all_results[i]['distance']
    
    # Clean up the temporary coordinate key column
    subset = subset.drop('coord_key', axis=1)
    
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
        (image_df['distance'] < 5)
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
    
    # Fix leading zeros issue - normalize both accession number columns
    caliper_df['Accession_Number'] = caliper_df['Accession_Number'].astype(str).str.lstrip('0')
    breast_df = breast_df.copy()  # Don't modify the original
    breast_df['Accession_Number'] = breast_df['Accession_Number'].astype(str).str.lstrip('0')
    
    # Handle edge case where all zeros becomes empty string
    caliper_df['Accession_Number'] = caliper_df['Accession_Number'].replace('', '0')
    breast_df['Accession_Number'] = breast_df['Accession_Number'].replace('', '0')
    
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
        accession_ids = db_to_process['Accession_Number'].unique()

        db_to_process['closest_fn']=''
        db_to_process['distance'] = 99999

        # 1 worker is the fastest for GCP, do not change
        with ThreadPoolExecutor(max_workers=1) as executor, tqdm(total=len(accession_ids), desc='') as progress:
            futures = {executor.submit(process_nearest_given_ids, pid, db_to_process, image_folder_path): pid for pid in accession_ids}

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