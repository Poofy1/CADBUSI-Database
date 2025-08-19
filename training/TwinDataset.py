import sys
sys.path.append('..')
from DB_processing.OCR import *
from DB_processing.data_selection import find_nearest_images

def process_patient_id(pid, db_out, image_folder_path):
    subset = db_out[db_out['Patient_ID'] == pid].copy()
    result = find_nearest_images(subset, pid, image_folder_path)
    
    if result:  # Only update if result is not empty
        # Create a DataFrame from the result dictionary
        result_df = pd.DataFrame.from_dict(result, orient='index')
        
        # Update subset using vectorized operations
        subset['closest_fn'] = result_df['sister_filename']
        subset['distance'] = result_df['distance']
    
    return subset

def Parse_Data(database_path, only_labels, max_patients=None):
    input_file = f'{database_path}/ImageData.csv'
    image_folder_path = f"{database_path}/images/"
    db_out = pd.read_csv(input_file)

    # Remove rows with missing data in crop_x, crop_y, crop_w, crop_h
    db_out.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'], inplace=True)

    if only_labels:
        db_to_process = db_out
    else:
        # Filter the rows where 'processed' is False
        db_to_process = db_out[db_out['processed'] == False]

    print("Finding Similar Images")
    patient_ids = db_to_process['Patient_ID'].unique()

    # Limit the number of patients if max_patients is specified
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]
        db_to_process = db_to_process[db_to_process['Patient_ID'].isin(patient_ids)]

    db_to_process['closest_fn'] = ''
    db_to_process['distance'] = -1

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(patient_ids), desc='') as progress:
        futures = {executor.submit(process_patient_id, pid, db_to_process, image_folder_path): pid for pid in patient_ids}

        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                db_to_process.update(result)
            progress.update()

    # Create a new dataframe for the output
    output_data = []
    
    # Find all pair caliper images
    caliper_images = db_to_process[db_to_process['has_calipers']]
    
    for _, caliper_row in caliper_images.iterrows():
        duplicate_image = caliper_row['closest_fn']
        duplicate_rows = db_to_process[db_to_process['ImageName'] == duplicate_image]
        
        if not duplicate_rows.empty:
            duplicate_row = duplicate_rows.iloc[0]
            
            # Check if the duplicate image doesn't have calipers and the distance is >= 8
            if not duplicate_row['has_calipers'] and caliper_row['distance'] <= 8:
                output_data.append({
                    'Patient_ID': caliper_row['Patient_ID'],
                    'Accession_Number': caliper_row['Accession_Number'],
                    'Caliper_Image': caliper_row['ImageName'],
                    'Duplicate_Image': duplicate_image,
                    'Distance': caliper_row['distance']
                })
    
    # Create a new dataframe from the output data
    output_df = pd.DataFrame(output_data)
    
    # Save the output dataframe to a CSV file
    output_file = f'{database_path}/caliper_pairs.csv'
    output_df.to_csv(output_file, index=False)
    
    print(f"Caliper pairs dataset saved to {output_file}")
    print(f"Number of caliper pairs found: {len(output_df)}")

# Call the function with a small subset of patients (e.g., 5)
Parse_Data("D:/DATA/CASBUSI/database_PAIR_GATHER/", False, max_patients=None)