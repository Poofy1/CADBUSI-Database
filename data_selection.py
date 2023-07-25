from pre_image_processing import *



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



def choose_images_to_label(db, case_data):
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
    male_patient_ids = case_data[case_data['PatientSex'] == 'M']['Patient_ID'].values
    db.loc[db['Patient_ID'].isin(male_patient_ids),'label'] = False
    
    #Add back majority focused images
    # Parse 'StudyDescription' for 'FOCUSED LEFT' or 'FOCUSED RIGHT'
    focus_left_studies = case_data[case_data['StudyDescription'].str.contains('FOCUSED LEFT')]['Accession_Number'].values
    db.loc[(db['Accession_Number'].isin(focus_left_studies)) & (db['laterality'] == 'right'), 'label'] = False

    focus_right_studies = case_data[case_data['StudyDescription'].str.contains('FOCUSED RIGHT')]['Accession_Number'].values
    db.loc[(db['Accession_Number'].isin(focus_right_studies)) & (db['laterality'] == 'left'), 'label'] = False
    
    # If 'BILATERAL' is present in 'StudyDescription', set 'label' to False for all images in that study
    bilateral_studies = case_data[case_data['StudyDescription'].str.contains('BILATERAL')]['Accession_Number'].values
    db.loc[db['Accession_Number'].isin(bilateral_studies), 'label'] = False
    
    # Check for mixed lateralities only in these non-focus studies
    non_focus_studies = case_data[~case_data['StudyDescription'].str.contains('FOCUSED|BILATERAL')]['Accession_Number'].values
    mixedIDs = find_mixed_lateralities( db[db['Accession_Number'].isin(non_focus_studies)] )
    db.loc[db['Accession_Number'].isin(mixedIDs),'label'] = False
    
    # Set label = False for all images with 'unknown' laterality
    db.loc[db['laterality'] == 'unknown', 'label'] = False
    

    # If 'chest' or 'mastectomy' is present in 'StudyDescription', set 'label' to False for all images in that study
    chest_or_mastectomy_studies = case_data[case_data['StudyDescription'].str.contains('chest|mastectomy', case=False)]['Accession_Number'].values
    db.loc[db['Accession_Number'].isin(chest_or_mastectomy_studies), 'label'] = False
    
     # Set label = False for all images with 'RegionCount' > 1
    db.loc[db['RegionCount'] > 1, 'label'] = False
    
    return db





def find_nearest_images(db, patient_id, image_folder_path):
    # Filter to only RGB images
    subset = db[db['PhotometricInterpretation'] != 'RGB'] 
    # Get index array 
    idx = np.array(fetch_index_for_patient_id(patient_id, subset))
    result = {}
    image_pairs_checked = set()

    for j,c in enumerate(idx):
        if c in image_pairs_checked:
            continue

        x = int(db.loc[c]['RegionLocationMinX0'])
        y = int(db.loc[c]['RegionLocationMinY0'])
        w = int(db.loc[c]['RegionLocationMaxX1']) - x
        h = int(db.loc[c]['RegionLocationMaxY1']) - y

        
        img_list = []
        for i,image_id in enumerate(idx):
            
            file_name = db.loc[image_id]['ImageName']
            full_filename = os.path.join(image_folder_path, file_name)
            img = Image.open(full_filename)
            img = np.array(img).astype(np.uint8)
            (rows, cols) = img.shape[0:2]
            if rows >= y + h and cols >= x + w:
                img,_ = make_grayscale(img)
                img = img[y:y+h,x:x+w]
            else: # fill in all ones for an image that will be distant
                img = np.full((h,w),255,dtype=np.uint8)
            img = img.flatten()
            img_list.append(img)
        
        img_stack = np.array(img_list, dtype=np.uint8)
        
        
        img_stack = np.abs(img_stack - img_stack[j, :])
        img_stack = np.mean(img_stack, axis=1)
        img_stack[j] = 1000
        sister_image = np.argmin(img_stack)
        distance = img_stack[sister_image]


        # Save result for the current image
        result[c] = {
            'image_filename': db.loc[c]['ImageName'],
            'sister_filename': db.loc[idx[sister_image]]['ImageName'],
            'distance': distance
        }

        # Save result for the sister image, if not already done
        if idx[sister_image] not in result:
            result[idx[sister_image]] = {
                'image_filename': db.loc[idx[sister_image]]['ImageName'],
                'sister_filename': db.loc[c]['ImageName'],
                'distance': distance
            }

        # Add the images to the set of checked pairs
        image_pairs_checked.add(c)
        image_pairs_checked.add(idx[sister_image])

    return result


def process_patient_id(pid, db_out, image_folder_path):
    subset = db_out[db_out['Accession_Number'] == pid]
    
    result = find_nearest_images(subset, pid, image_folder_path)
    idxs = result.keys()
    for i in idxs:
        subset.loc[i, 'closest_fn'] = result[i]['sister_filename']
        subset.loc[i, 'distance'] = result[i]['distance']
    return subset


def Parse_Data():
    input_file = f'{env}/database/ImageData.csv'
    case_file = f'{env}/database/CaseStudyData.csv'
    image_folder_path = f"{env}/database/images/"
    db_out = pd.read_csv(input_file)
    case_data = pd.read_csv(case_file)
    

    # Check if 'processed' column exists, if not create it
    if 'processed' not in db_out.columns:
        db_out['processed'] = False

    # Remove rows with missing data in crop_x, crop_y, crop_w, crop_h
    db_out = db_out.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'])

    # Filter the rows where 'processed' is False
    db_to_process = db_out[db_out['processed'] == False]

    print("Finding Similar Images")
    patient_ids = db_to_process['Accession_Number'].unique()

    db_to_process['closest_fn']=''
    db_to_process['distance'] = -1

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(patient_ids), desc='') as progress:
        futures = {executor.submit(process_patient_id, pid, db_to_process, image_folder_path): pid for pid in patient_ids}

        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                db_to_process.update(result)
            progress.update()

    db_to_process = choose_images_to_label(db_to_process, case_data)
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
    db_out.to_csv(input_file, index=False)

