import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os
from src.DB_processing.tools import append_audit
from main import env


def get_diagnosis_column(study_laterality, pathology_laterality=None):
    """
    Determine which diagnosis column(s) to populate based on laterality.

    Args:
        study_laterality: The Study_Laterality from the US record
        pathology_laterality: The Pathology_Laterality (optional, for bilateral cases)

    Returns:
        A list of column names to populate: ['left_diagnosis'], ['right_diagnosis'], or both
    """
    if study_laterality == 'LEFT':
        return ['left_diagnosis']
    elif study_laterality == 'RIGHT':
        return ['right_diagnosis']
    elif study_laterality == 'BILATERAL':
        # For bilateral studies, use pathology laterality if available
        # Pathology_Laterality is always LEFT or RIGHT (never BILATERAL)
        if pathology_laterality == 'LEFT':
            return ['left_diagnosis']
        elif pathology_laterality == 'RIGHT':
            return ['right_diagnosis']
        else:
            # If no pathology laterality specified, populate both
            return ['left_diagnosis', 'right_diagnosis']
    return []


def check_assumed_benign(final_df):
    """
    Check for benign cases based on 18-month follow-up.
    Only considers:
    1. Cases with BI-RADS '1' or '2'
    2. Cases with no biopsy from -30 to +120 days around the US date
    3. Cases with no non-benign (non 1-2) BI-RADS in 24 month follow-up
    4. Cases with no 'MALIGNANT' in path_interpretation within 15 months
    """
    today = pd.Timestamp.now()
    
    # Pre-compute the eligible US records with BI-RADS 1 or 2 to avoid processing irrelevant rows
    us_mask = (final_df['BI-RADS'].isin(['1', '2'])) & pd.notna(final_df['DATE'])
    us_records = final_df[us_mask]
    
    # Only process patients who have eligible US records
    patient_ids = us_records['PATIENT_ID'].unique()
    
    # Create a dictionary to store final interpretation updates
    updates = {}
    
    for patient_id in patient_ids:
        # Get all records for this patient once and sort them
        patient_records = final_df[final_df['PATIENT_ID'] == patient_id].sort_values('DATE')
        
        # Process only eligible US records for this patient
        patient_us_records = us_records[us_records['PATIENT_ID'] == patient_id]
        
        for idx, row in patient_us_records.iterrows():
            """# Skip records that are less than 24 months old
            if (today - row['DATE']).days < 730:
                failure_reasons['insufficient_time_24months'] += 1
                continue"""
            
            # Define time windows
            biopsy_window_start = row['DATE'] - pd.Timedelta(days=30)
            biopsy_window_end = row['DATE'] + pd.Timedelta(days=120)
            followup_end = row['DATE'] + pd.Timedelta(days=730)  # 24 months
            malignancy_window_end = row['DATE'] + pd.Timedelta(days=450)  # 15 months
            
            # Check if there's at least 6 months of follow-up data available
            last_visit_date = patient_records[patient_records['DATE'] > row['DATE']]['DATE'].max()
            if pd.isna(last_visit_date) or (last_visit_date - row['DATE']).days < 180:  # 6 months = 180 days
                continue
            
            # Filter for records in the biopsy window efficiently
            biopsy_window_records = patient_records[
                (patient_records['DATE'] >= biopsy_window_start) &
                (patient_records['DATE'] <= biopsy_window_end)
            ]
            
            # Skip if any biopsies exist in this window
            if 'is_biopsy' in biopsy_window_records.columns and (biopsy_window_records['is_biopsy'] == 'T').any():
                continue
            
            # Filter for follow-up records efficiently
            followup_records = patient_records[
                (patient_records['DATE'] > row['DATE']) &
                (patient_records['DATE'] <= followup_end)
            ]
            
            # Check for non-benign BI-RADS in follow-up period efficiently
            if 'BI-RADS' in followup_records.columns:
                non_benign_mask = (
                    followup_records['BI-RADS'].notna() & 
                    (followup_records['BI-RADS'] != '') &
                    ~followup_records['BI-RADS'].isin(['1', '2'])
                )
                if non_benign_mask.any():
                    continue
            
            # Filter for records in malignancy window
            malignancy_records = patient_records[
                (patient_records['DATE'] > row['DATE']) &
                (patient_records['DATE'] <= malignancy_window_end)
            ]
            
            # Check for 'MALIGNANT' in path_interpretation efficiently
            if 'path_interpretation' in malignancy_records.columns:
                # Check specifically for non-null values that contain "MALIGNANT"
                has_malignancy = False
                for _, malignancy_record in malignancy_records.iterrows():
                    if pd.notna(malignancy_record.get('path_interpretation')) and 'MALIGNANT' in str(malignancy_record['path_interpretation']).upper():
                        has_malignancy = True
                        break
                
                if has_malignancy:
                    continue
            
            # If all checks pass, mark for update
            updates[idx] = ('BENIGN1', row.get('Study_Laterality'))

    # Apply all updates at once
    for idx, (value, laterality) in updates.items():
        diagnosis_cols = get_diagnosis_column(laterality)
        for col in diagnosis_cols:
            final_df.at[idx, col] = value

    return final_df

def check_assumed_benign_birads3(final_df):
    """
    Check for benign cases based on 36-month follow-up for BI-RADS 3 cases.
    Requirements:
    1. Cases with BI-RADS '3'
    2. Cases with no biopsy from -30 to +120 days around the US date
    3. Cases where all future US exams within 36 months have BI-RADS null, 1, or 2
       OR all future US exams have BI-RADS null, 0, 1, or 3 with at least one exam ≥24 months out
    4. Cases with no 'MALIGNANT' in path_interpretation within 15 months
    5. Minimum follow-up period of 4 months
    """
    today = pd.Timestamp.now()
    
    # Pre-compute the eligible US records with BI-RADS 3 to avoid processing irrelevant rows
    us_mask = (final_df['BI-RADS'] == '3') & pd.notna(final_df['DATE'])
    us_records = final_df[us_mask]
    
    # Only process patients who have eligible US records
    patient_ids = us_records['PATIENT_ID'].unique()
    
    # Create a dictionary to store final interpretation updates
    updates = {}
    
    for patient_id in patient_ids:
        # Get all records for this patient once and sort them
        patient_records = final_df[final_df['PATIENT_ID'] == patient_id].sort_values('DATE')
        
        # Process only eligible US records for this patient
        patient_us_records = us_records[us_records['PATIENT_ID'] == patient_id]
        
        for idx, row in patient_us_records.iterrows():
            """# Skip records that are less than 36 months old
            if (today - row['DATE']).days < 1095:  # 36 months = 1095 days
                failure_reasons['insufficient_time_36months'] += 1
                continue
            """
            
            # Define time windows
            biopsy_window_start = row['DATE'] - pd.Timedelta(days=30)
            biopsy_window_end = row['DATE'] + pd.Timedelta(days=120)
            followup_end = row['DATE'] + pd.Timedelta(days=1095)  # 36 months max
            malignancy_window_end = row['DATE'] + pd.Timedelta(days=450)  # 15 months

            # Check if there's at least 6 months of follow-up data available
            last_visit_date = patient_records[patient_records['DATE'] > row['DATE']]['DATE'].max()
            if pd.isna(last_visit_date) or (last_visit_date - row['DATE']).days < 180:  # 6 months
                continue
            
            # Filter for records in the biopsy window efficiently
            biopsy_window_records = patient_records[
                (patient_records['DATE'] >= biopsy_window_start) &
                (patient_records['DATE'] <= biopsy_window_end)
            ]
            
            # Skip if any biopsies exist in this window
            if 'is_biopsy' in biopsy_window_records.columns and (biopsy_window_records['is_biopsy'] == 'T').any():
                continue
            
            # Filter for follow-up US records specifically
            followup_us_records = patient_records[
                (patient_records['DATE'] > row['DATE']) &
                (patient_records['DATE'] <= followup_end) #&
                #(patient_records['MODALITY'] == 'US')
            ]

            if len(followup_us_records) > 0 and 'BI-RADS' in followup_us_records.columns:
                # FIRST CHECK: if all future US exams have BI-RADS null, 1, or 2
                valid_birads_first_check = (
                    followup_us_records['BI-RADS'].notna() & 
                    (followup_us_records['BI-RADS'] != '') &
                    followup_us_records['BI-RADS'].isin(['1', '2'])
                )
                
                if valid_birads_first_check.all():
                    pass  # First check passes, continue with remaining checks
                else:
                    # SECOND CHECK: if first check fails, verify if all are null, 0, 1, or 3
                    # AND at least one exam is ≥24 months after initial exam
                    valid_birads_second_check = (
                        followup_us_records['BI-RADS'].notna() & 
                        (followup_us_records['BI-RADS'] != '') &
                        followup_us_records['BI-RADS'].isin(['1', '2', '3'])
                    )
                    if not valid_birads_second_check.all():
                        continue  # Neither check passes, skip this record
                    
                    # Check if at least one exam is 24+ months after the initial exam
                    has_24month_followup = False
                    for _, followup_row in followup_us_records.iterrows():
                        if (followup_row['DATE'] - row['DATE']).days >= 730:  # 24 months = 730 days
                            has_24month_followup = True
                            break
                    
                    if not has_24month_followup:
                        continue  # No 24+ month followup, skip this record
            
            # Filter for records in malignancy window
            malignancy_records = patient_records[
                (patient_records['DATE'] > row['DATE']) &
                (patient_records['DATE'] <= malignancy_window_end)
            ]
            
            # Check for 'MALIGNANT' in path_interpretation efficiently
            if 'path_interpretation' in malignancy_records.columns:
                # Check specifically for non-null values that contain "MALIGNANT"
                has_malignancy = False
                for _, malignancy_record in malignancy_records.iterrows():
                    if pd.notna(malignancy_record.get('path_interpretation')) and 'MALIGNANT' in str(malignancy_record['path_interpretation']).upper():
                        has_malignancy = True
                        break
                
                if has_malignancy:
                    continue

            # If all checks pass, mark for update
            updates[idx] = ('BENIGN3', row.get('Study_Laterality'))

    # Apply all updates at once
    for idx, (value, laterality) in updates.items():
        diagnosis_cols = get_diagnosis_column(laterality)
        for col in diagnosis_cols:
            final_df.at[idx, col] = value

    return final_df

def check_malignant_from_biopsy(final_df):
    """
    Check for malignancy indicators by marking rows as MALIGNANT1
    if BI-RADS = 6 and MODALITY is 'US', for cases without interpretation,
    but only if there exists at least one 'MALIGNANT' in path_interpretation
    FOR THAT PATIENT. For bilateral cases, only marks the specific breast(s)
    that have malignancy based on Pathology_Laterality.
    """
    updates = {}

    # Process each patient separately
    for patient_id in final_df['PATIENT_ID'].unique():
        # Get all records for this patient
        patient_records = final_df[final_df['PATIENT_ID'] == patient_id]

        # Collect which breast(s) have malignancy from pathology records
        malignant_breasts = set()
        for _, record in patient_records.iterrows():
            if pd.notna(record.get('path_interpretation')) and 'MALIGNANT' in str(record['path_interpretation']):
                path_lat = record.get('Pathology_Laterality')
                if pd.notna(path_lat) and path_lat in ['LEFT', 'RIGHT']:
                    malignant_breasts.add(path_lat)

        # Only proceed if there's at least one malignant breast identified
        if malignant_breasts:
            # Find BI-RADS 6 US rows for this patient
            us_birads6_rows = patient_records[
                (pd.notna(patient_records.get('MODALITY'))) &
                #(patient_records['MODALITY'] == 'US') &
                (pd.notna(patient_records.get('BI-RADS'))) &
                (patient_records['BI-RADS'] == '6')
            ].index

            for idx in us_birads6_rows:
                row = final_df.loc[idx]
                laterality = row.get('Study_Laterality')

                if laterality == 'BILATERAL':
                    # For bilateral cases, only mark breasts that have malignancy
                    for malignant_breast in malignant_breasts:
                        # Check if the specific diagnosis column is empty
                        diag_col = 'left_diagnosis' if malignant_breast == 'LEFT' else 'right_diagnosis'
                        if pd.isna(row.get(diag_col)) or row.get(diag_col) == '':
                            # Store the update with the pathology laterality
                            if idx not in updates:
                                updates[idx] = []
                            updates[idx].append(('MALIGNANT1', laterality, malignant_breast))
                else:
                    # For non-bilateral cases, check if the laterality matches any malignant breast
                    if laterality in malignant_breasts:
                        # Check if both diagnosis columns are empty
                        left_empty = pd.isna(row.get('left_diagnosis')) or row.get('left_diagnosis') == ''
                        right_empty = pd.isna(row.get('right_diagnosis')) or row.get('right_diagnosis') == ''
                        if left_empty and right_empty:
                            if idx not in updates:
                                updates[idx] = []
                            updates[idx].append(('MALIGNANT1', laterality, None))

    # Apply all updates at once
    for idx, update_list in updates.items():
        for value, laterality, pathology_laterality in update_list:
            diagnosis_cols = get_diagnosis_column(laterality, pathology_laterality)
            for col in diagnosis_cols:
                final_df.at[idx, col] = value

    return final_df


def check_from_next_diagnosis(final_df, days=240):
    """
    For 'US' rows with empty diagnosis columns:
    - For MALIGNANT cases: Apply only to BI-RADS '4', '4A', '4B', '4C', '5', or '6'
      Set diagnosis to 'MALIGNANT2' only if:
      1. The laterality matches between the US study and the pathology
      2. At least one record in the date range has 'is_biopsy' = 'T'

    - For BENIGN cases: Apply only to BI-RADS '1', '2', '3', '4', '4A', '4B'
      Set diagnosis to 'BENIGN2' if:
      1. The laterality matches between the US study and the pathology

    Special case: If Study_Laterality is 'BILATERAL', look at Pathology_Laterality
    to determine which diagnosis column(s) to populate.
    """
    # Define the target BI-RADS values for malignant and benign cases
    target_birads_malignant = set(['4', '4A', '4B', '4C', '5', '6'])
    target_birads_benign = set(['1', '2', '3', '4', '4A', '4B'])
    updates = {}

    for patient_id in final_df['PATIENT_ID'].unique():
        patient_mask = final_df['PATIENT_ID'] == patient_id
        patient_records = final_df[patient_mask].copy().sort_values('DATE')

        # Pre-filter to only include relevant US records for this patient
        # Check if both diagnosis columns are empty
        left_empty = patient_records['left_diagnosis'].isna() | (patient_records['left_diagnosis'] == '')
        right_empty = patient_records['right_diagnosis'].isna() | (patient_records['right_diagnosis'] == '')

        relevant_records = patient_records[
            #(patient_records['MODALITY'] == 'US') &
            (left_empty & right_empty) &
            pd.notna(patient_records['DATE']) &
            pd.notna(patient_records['Study_Laterality'])
        ]
        
        for idx, current_row in relevant_records.iterrows():
            current_date = current_row['DATE']
            future_date = current_date + pd.Timedelta(days=days)
            current_laterality = current_row['Study_Laterality']
            current_birads = current_row['BI-RADS']
            
            # Find future records within the time window once
            future_records = patient_records[
                (patient_records['DATE'] > current_date) & 
                (patient_records['DATE'] <= future_date)
            ]
            
            # Check if there's at least one record with 'is_biopsy' = 'T' in the date range
            has_us_biopsy = any(
                (record['is_biopsy'] == 'T') 
                for _, record in future_records.iterrows() 
                if pd.notna(record.get('is_biopsy'))
            )
            
            # Handle BILATERAL case - check each pathology laterality separately
            if current_laterality == 'BILATERAL':
                # Track findings by laterality
                left_findings = {'malignant': False, 'benign': False}
                right_findings = {'malignant': False, 'benign': False}

                for _, future_row in future_records.iterrows():
                    if pd.notna(future_row.get('path_interpretation')) and pd.notna(future_row.get('Pathology_Laterality')):
                        path_interp = future_row['path_interpretation'].upper()
                        path_lat = future_row['Pathology_Laterality']

                        # Pathology_Laterality is always LEFT or RIGHT (never BILATERAL)
                        if path_lat == 'LEFT':
                            if path_interp == 'MALIGNANT':
                                left_findings['malignant'] = True
                            elif path_interp == 'BENIGN':
                                left_findings['benign'] = True
                        elif path_lat == 'RIGHT':
                            if path_interp == 'MALIGNANT':
                                right_findings['malignant'] = True
                            elif path_interp == 'BENIGN':
                                right_findings['benign'] = True

                # Store updates for each side separately
                if idx not in updates:
                    updates[idx] = {}

                # Apply malignant findings (only for target BI-RADS values)
                if left_findings['malignant'] and has_us_biopsy and current_birads in target_birads_malignant:
                    updates[idx]['left_diagnosis'] = 'MALIGNANT2'
                elif left_findings['benign'] and not left_findings['malignant'] and current_birads in target_birads_benign:
                    updates[idx]['left_diagnosis'] = 'BENIGN2'

                if right_findings['malignant'] and has_us_biopsy and current_birads in target_birads_malignant:
                    updates[idx]['right_diagnosis'] = 'MALIGNANT2'
                elif right_findings['benign'] and not right_findings['malignant'] and current_birads in target_birads_benign:
                    updates[idx]['right_diagnosis'] = 'BENIGN2'

            # Handle regular laterality matching
            else:
                found_malignant = False
                found_benign = False

                # Check all matching records
                for _, future_row in future_records.iterrows():
                    if pd.notna(future_row.get('path_interpretation')):
                        if (pd.notna(future_row.get('Pathology_Laterality')) and
                            future_row['Pathology_Laterality'] == current_laterality):

                            path_interp = future_row['path_interpretation'].upper()
                            if path_interp == 'MALIGNANT':
                                found_malignant = True
                            elif path_interp == 'BENIGN':
                                found_benign = True

                # Store updates
                if idx not in updates:
                    updates[idx] = {}

                # Apply malignant finding only for target BI-RADS values
                if found_malignant and has_us_biopsy and current_birads in target_birads_malignant:
                    diagnosis_cols = get_diagnosis_column(current_laterality)
                    for col in diagnosis_cols:
                        updates[idx][col] = 'MALIGNANT2'
                # Apply benign finding only for target benign BI-RADS values, and only if NO malignant cases are found
                elif found_benign and not found_malignant and current_birads in target_birads_benign:
                    diagnosis_cols = get_diagnosis_column(current_laterality)
                    for col in diagnosis_cols:
                        updates[idx][col] = 'BENIGN2'

    # Apply all updates at once
    for idx, col_updates in updates.items():
        for col, value in col_updates.items():
            final_df.at[idx, col] = value

    return final_df


def check_pathology_override(final_df):
    """
    Final classification override based on A2_PATHOLOGY_CATEGORY_DESC and A1_PATHOLOGY_CATEGORY_DESC.
    This is the last classification method that runs and will override previous classifications.

    Priority: A2_PATHOLOGY_CATEGORY_DESC > A1_PATHOLOGY_CATEGORY_DESC
    If 'malignant' (case insensitive) exists: MALIGNANT4
    Otherwise if non-empty: BENIGN4
    Updates the appropriate diagnosis column(s) based on laterality.
    """
    updates = {}

    # Track metrics
    filled_empty = 0  # Empty -> Filled
    equivalent_override = 0  # Same type (Benign* -> Benign4 or Malignant* -> Malignant4)
    disagreement = 0  # Different type (Benign* -> Malignant4 or Malignant* -> Benign4)

    # Track disagreement details for CSV export
    disagreement_records = []

    for idx, row in final_df.iterrows():
        # Get pathology category with A2 priority
        pathology_cat = None
        pathology_source = None
        if pd.notna(row.get('A2_PATHOLOGY_CATEGORY_DESC')) and str(row.get('A2_PATHOLOGY_CATEGORY_DESC')).strip():
            pathology_cat = str(row.get('A2_PATHOLOGY_CATEGORY_DESC')).strip()
            pathology_source = 'A2'
        elif pd.notna(row.get('A1_PATHOLOGY_CATEGORY_DESC')) and str(row.get('A1_PATHOLOGY_CATEGORY_DESC')).strip():
            pathology_cat = str(row.get('A1_PATHOLOGY_CATEGORY_DESC')).strip()
            pathology_source = 'A1'

        # If we have a pathology category, determine classification
        if pathology_cat:
            # Check if 'malignant' or 'invasive' appears (case insensitive)
            pathology_lower = pathology_cat.lower()
            if 'malignant' in pathology_lower or 'invasive' in pathology_lower:
                classification = 'MALIGNANT4'
            else:
                classification = 'BENIGN4'

            # Get laterality to determine which column(s) to update
            laterality = row.get('Study_Laterality')

            # Get the pathology laterality if available for bilateral cases
            path_laterality = row.get('Pathology_Laterality')

            # Get the diagnosis columns that will be updated
            diagnosis_cols = get_diagnosis_column(laterality, path_laterality)

            # Check existing diagnoses to categorize the update
            for col in diagnosis_cols:
                old_value = row.get(col)

                # Categorize the change
                if pd.isna(old_value) or old_value == '' or old_value is None:
                    filled_empty += 1
                elif 'BENIGN' in str(old_value) and 'BENIGN' in classification:
                    equivalent_override += 1
                elif 'MALIGNANT' in str(old_value) and 'MALIGNANT' in classification:
                    equivalent_override += 1
                elif ('BENIGN' in str(old_value) and 'MALIGNANT' in classification) or \
                     ('MALIGNANT' in str(old_value) and 'BENIGN' in classification):
                    disagreement += 1
                    # Record disagreement details
                    disagreement_records.append({
                        'patient_id': row.get('PATIENT_ID'),
                        'accession_number': row.get('ACCESSION_NUMBER'),
                        'study_laterality': laterality,
                        'pathology_laterality': path_laterality,
                        'diagnosis_column': col,
                        'before_classification': old_value,
                        'after_classification': classification,
                        'pathology_category': pathology_cat,
                        'pathology_source': pathology_source,
                        'A1_PATHOLOGY_CATEGORY_DESC': row.get('A1_PATHOLOGY_CATEGORY_DESC'),
                        'A2_PATHOLOGY_CATEGORY_DESC': row.get('A2_PATHOLOGY_CATEGORY_DESC')
                    })

            # Store the update
            updates[idx] = (classification, laterality, path_laterality)

    # Apply all updates at once
    for idx, (value, laterality, path_laterality) in updates.items():
        diagnosis_cols = get_diagnosis_column(laterality, path_laterality)
        for col in diagnosis_cols:
            final_df.at[idx, col] = value

    # Return dataframe, metrics, and disagreement records
    return final_df, filled_empty, equivalent_override, disagreement, disagreement_records


def determine_final_interpretation(final_df, batch_size=100, max_workers=None):

    if 'left_diagnosis' not in final_df.columns:
        final_df['left_diagnosis'] = None
    if 'right_diagnosis' not in final_df.columns:
        final_df['right_diagnosis'] = None

    if 'BI-RADS' in final_df.columns:
        # Convert to string, remove .0, and replace 'nan' with empty string
        final_df['BI-RADS'] = (
            final_df['BI-RADS']
            .astype(str)
            .str.replace('.0', '', regex=False)
            .str.replace('nan', '', regex=False)
        )

    # Get unique patients
    unique_patients = final_df['PATIENT_ID'].unique()

    # Split patients into batches of size batch_size
    patient_batches = [
        unique_patients[i:i + batch_size]
        for i in range(0, len(unique_patients), batch_size)
    ]

    # Define worker function to process a batch of patients through all classification methods
    def process_patient_batch(patient_ids):
        # Create a DEEP copy of patient records
        batch_df = final_df[final_df['PATIENT_ID'].isin(patient_ids)].copy(deep=True)

        # Process locally without touching final_df
        batch_df = check_assumed_benign(batch_df)
        batch_df = check_assumed_benign_birads3(batch_df)
        batch_df = check_malignant_from_biopsy(batch_df)
        batch_df = check_from_next_diagnosis(batch_df)
        batch_df, filled_empty, equivalent_override, disagreement, disagreement_records = check_pathology_override(batch_df)

        # Only return rows that were actually modified (either column has a value)
        modified_mask = batch_df['left_diagnosis'].notna() | batch_df['right_diagnosis'].notna()
        result_df = batch_df.loc[modified_mask, ['PATIENT_ID', 'ACCESSION_NUMBER', 'left_diagnosis', 'right_diagnosis']]

        # Return results, metrics, and disagreement records
        return result_df, filled_empty, equivalent_override, disagreement, disagreement_records
   
    # Use ThreadPoolExecutor to process batches in parallel
    results = []

    # Aggregate metrics and disagreement records from all batches
    total_filled_empty = 0
    total_equivalent_override = 0
    total_disagreement = 0
    all_disagreement_records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for processing and capture futures
        futures = {
            executor.submit(process_patient_batch, batch): batch
            for batch in patient_batches
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Determining classifications"
        ):
            try:
                batch_result, filled_empty, equivalent_override, disagreement, disagreement_records = future.result()
                results.append(batch_result)

                # Aggregate metrics
                total_filled_empty += filled_empty
                total_equivalent_override += equivalent_override
                total_disagreement += disagreement

                # Collect disagreement records
                all_disagreement_records.extend(disagreement_records)
            except Exception as e:
                print(f"Error processing batch: {e}")

    if not results:
        return final_df

    # Combine all results
    combined_results = pd.concat(results)

    # Update the original dataframe with the processed results
    # We only update the 'left_diagnosis' and 'right_diagnosis' columns
    final_df.update(combined_results)

    # Print aggregated pathology override metrics
    total_updates = total_filled_empty + total_equivalent_override + total_disagreement
    if total_updates > 0:
        print("\n=== Pathology Override Metrics ===")
        print(f"Total diagnosis updates: {total_updates}")
        print(f"Filled empty diagnoses: {total_filled_empty} ({total_filled_empty/total_updates*100:.1f}%)")
        print(f"Equivalent overrides: {total_equivalent_override} ({total_equivalent_override/total_updates*100:.1f}%)")
        print(f"Disagreements: {total_disagreement} ({total_disagreement/total_updates*100:.1f}%)")
        print("=" * 35)

        # Log to audit
        append_audit("classification.pathology_override_total", total_updates)
        append_audit("classification.pathology_override_filled_empty", total_filled_empty)
        append_audit("classification.pathology_override_equivalent", total_equivalent_override)
        append_audit("classification.pathology_override_disagreement", total_disagreement)

    # Export disagreement records to CSV
    if all_disagreement_records:
        disagreement_df = pd.DataFrame(all_disagreement_records)
        output_path = os.path.join(env, 'data', 'pathology_override_disagreements.csv')
        disagreement_df.to_csv(output_path, index=False)
        print(f"\nExported {len(all_disagreement_records)} disagreement records to: {output_path}")

    return final_df


def audit_interpretations(final_df):

    # After all processing, search for specific categories in both diagnosis columns
    # Count each diagnosis type from both columns
    benign1_count = sum((final_df['left_diagnosis'] == 'BENIGN1') | (final_df['right_diagnosis'] == 'BENIGN1'))
    benign2_count = sum((final_df['left_diagnosis'] == 'BENIGN2') | (final_df['right_diagnosis'] == 'BENIGN2'))
    benign3_count = sum((final_df['left_diagnosis'] == 'BENIGN3') | (final_df['right_diagnosis'] == 'BENIGN3'))
    benign4_count = sum((final_df['left_diagnosis'] == 'BENIGN4') | (final_df['right_diagnosis'] == 'BENIGN4'))
    malignant1_count = sum((final_df['left_diagnosis'] == 'MALIGNANT1') | (final_df['right_diagnosis'] == 'MALIGNANT1'))
    malignant2_count = sum((final_df['left_diagnosis'] == 'MALIGNANT2') | (final_df['right_diagnosis'] == 'MALIGNANT2'))
    malignant4_count = sum((final_df['left_diagnosis'] == 'MALIGNANT4') | (final_df['right_diagnosis'] == 'MALIGNANT4'))

    # Create audit log with counts for each category
    append_audit("query_clean.assumed_benign", benign1_count)
    append_audit("query_clean.birads3_benign", benign3_count)
    append_audit("query_clean.path_confirmed_benign", benign2_count)
    append_audit("query_clean.pathology_override_benign", benign4_count)
    append_audit("query_clean.path_confirmed_malignant", malignant2_count)
    append_audit("query_clean.birads6_malignant", malignant1_count)
    append_audit("query_clean.pathology_override_malignant", malignant4_count)

    # Calculate total benign (all benign categories)
    total_benign = benign1_count + benign2_count + benign3_count + benign4_count
    append_audit("query_clean.final_benign_count", total_benign)

    # Calculate total malignant (all malignant categories)
    total_malignant = malignant1_count + malignant2_count + malignant4_count
    append_audit("query_clean.final_malignant_count", total_malignant)
    
    # Calculate unknown (cases without a final interpretation)
    total_rows = len(final_df)
    total_known = total_benign + total_malignant
    unknown_count = total_rows - total_known
    append_audit("query_clean.final_unknown_count", unknown_count)


