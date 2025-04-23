from DB_processing.image_processing import *

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



def choose_images_to_label(db, breast_df):
    db['label'] = True

    #Remove images that are too dark
    db.loc[db['darkness'] > 65, 'label'] = False
    
    # find all of the rows with calipers
    caliper_rows = db[db['has_calipers']]

    # loop over caliper rows and tag twin images (not efficient)
    for idx, row in caliper_rows.iterrows():
        distance = row['distance']
        if distance <= 5:
            db.at[idx,'label'] = False

    # set label = False for all non-breast images
    db.loc[(db['area'] == 'unknown') | (db['area'].isna()), 'area'] = 'breast'
    db.loc[(db['area'] != 'breast'), 'label'] = False

    # Remove Males from studies
    male_patient_ids = breast_df[breast_df['US_CORE_BIRTHSEX'] == 'M']['Patient_ID'].values
    db.loc[db['Patient_ID'].isin(male_patient_ids),'label'] = False
    
    # Set label = False for all images with 'unknown' laterality
    db.loc[(db['laterality'] == 'unknown') | (db['laterality'].isna()), 'label'] = False
    
    # If 'chest' or 'mastectomy' is present in 'StudyDescription', set 'label' to False for all images in that study
    chest_or_mastectomy_studies = breast_df[breast_df['DESCRIPTION'].fillna('').str.contains('chest|mastectomy', case=False)]['Patient_ID'].values
    db.loc[db['Patient_ID'].isin(chest_or_mastectomy_studies), 'label'] = False
    
    # Set label = False for all images with 'RegionCount' > 1
    db.loc[db['RegionCount'] > 1, 'label'] = False
    
    # Check the aspect ratio of the crop region
    db['crop_aspect_ratio'] = (db['crop_w'] / db['crop_h']).round(2)
    
    return db





def find_nearest_images(db, patient_id, image_folder_path):
    subset = db[db['PhotometricInterpretation'] != 'RGB']
    idx = np.array(fetch_index_for_patient_id(patient_id, subset))
    result = {}
    image_pairs_checked = set()

    # Precompute cropping coordinates
    crop_coords = subset[['RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1']].astype(int)
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
    save_data(df, input_file)  # Save the updated DataFrame back to the CSV
    

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
    db_out.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'], inplace=True)

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

    db_to_process = choose_images_to_label(db_to_process, breast_df)
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
    
    
tqdm.pandas()

def Rename_Images(database_path):
    input_file = f'{database_path}/ImageData.csv'
    image_folder_path = f"{database_path}/images/"
    df = read_csv(input_file)
    
    print("Renaming Images With Laterality")
    
    # Create a dictionary to keep track of the instance numbers
    instance_dict = {}

    def rename_images(row):
        old_image_name = row['ImageName']
        old_image_path = os.path.join(image_folder_path, old_image_name)
        
        # Check if the old image path exists
        if not os.path.exists(old_image_path):
            return None

        # Check if the old image name is already in the desired format
        if old_image_name.count('_') == 3:
            return old_image_name
    
        # Extract the relevant information
        patient_id = int(row['Patient_ID'])
        accession_number = row['Accession_Number']
        accession_number = '' if pd.isna(accession_number) else (accession_number)
        laterality = row['laterality']

        # Create a unique key for this combination
        key = (patient_id, accession_number, laterality)

        # Get the current instance number for this combination
        instance_number = instance_dict.get(key, 0)

        # Generate the new image name
        new_image_name = f"{patient_id}_{accession_number}_{laterality}_{instance_number}.png"
        new_image_path = os.path.join(image_folder_path, new_image_name)
    
        # If the new image name already exists, remove the old image path
        if file_exists(new_image_path):
            delete_file(old_image_path)
            return None

        # Rename the image file
        rename_file(old_image_path, new_image_path)

        # Update the dictionary for the next instance
        instance_dict[key] = instance_number + 1
        
        return new_image_name

    df['ImageName'] = df.progress_apply(rename_images, axis=1)
    df.dropna(subset=['ImageName'], inplace=True)

    # Save the updated DataFrame to the same CSV file
    save_data(df, input_file)



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
    
    # Delete the duplicate images
    for image_name in tqdm(duplicate_image_names):
        image_path = os.path.join(image_folder_path, image_name)
        if file_exists(image_path):
            delete_file(image_path)
    
    # Print the number of duplicates removed
    print(f"Removed {duplicate_count} duplicate rows")