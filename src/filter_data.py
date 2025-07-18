import os
import pandas as pd
from tqdm import tqdm
from src.DB_processing.tools import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory

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
    us_mask = (final_df['MODALITY'] == 'US') & (final_df['BI-RADS'].isin(['1', '2'])) & pd.notna(final_df['DATE'])
    us_records = final_df[us_mask]
    
    # Only process patients who have eligible US records
    patient_ids = us_records['PATIENT_ID'].unique()
    
    # Create a dictionary to store final interpretation updates
    updates = {}
    
    for patient_id in tqdm(patient_ids, desc="Checking benign based on followup"):
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
                non_benign_mask = followup_records['BI-RADS'].notna() & ~followup_records['BI-RADS'].isin(['1', '2'])
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
            updates[idx] = 'BENIGN1'
    
    # Apply all updates at once
    for idx, value in updates.items():
        final_df.at[idx, 'final_interpretation'] = value
    
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
    us_mask = (final_df['MODALITY'] == 'US') & (final_df['BI-RADS'] == '3') & pd.notna(final_df['DATE'])
    us_records = final_df[us_mask]
    
    # Only process patients who have eligible US records
    patient_ids = us_records['PATIENT_ID'].unique()
    
    # Create a dictionary to store final interpretation updates
    updates = {}
    
    for patient_id in tqdm(patient_ids, desc="Checking benign based on BI-RADS 3 followup"):
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
                (patient_records['DATE'] <= followup_end) &
                (patient_records['MODALITY'] == 'US')
            ]

            if len(followup_us_records) > 0 and 'BI-RADS' in followup_us_records.columns:
                # FIRST CHECK: if all future US exams have BI-RADS null, 1, or 2
                valid_birads_first_check = followup_us_records['BI-RADS'].isnull() | followup_us_records['BI-RADS'].isin(['1', '2'])
                
                if valid_birads_first_check.all():
                    pass  # First check passes, continue with remaining checks
                else:
                    # SECOND CHECK: if first check fails, verify if all are null, 0, 1, or 3
                    # AND at least one exam is ≥24 months after initial exam
                    valid_birads_second_check = followup_us_records['BI-RADS'].isnull() | followup_us_records['BI-RADS'].isin(['1', '2', '3'])
                    
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
            updates[idx] = 'BENIGN3'
    
    # Apply all updates at once
    for idx, value in updates.items():
        final_df.at[idx, 'final_interpretation'] = value
    
    return final_df

def check_malignant_from_biopsy(final_df):
    """
    Check for malignancy indicators by marking rows as MALIGNANT1 
    if BI-RADS = 6 and MODALITY is 'US', for cases without interpretation,
    but only if there exists at least one 'MALIGNANT' in path_interpretation.
    """
    # First, check if any path_interpretation contains 'MALIGNANT'
    has_malignant = any(
        isinstance(interp, str) and 'MALIGNANT' in interp 
        for interp in final_df['path_interpretation'] 
        if pd.notna(interp)
    )
    
    # Only proceed if there's at least one 'MALIGNANT' in path_interpretation
    if has_malignant:
        us_birads6_rows = final_df[
            (pd.notna(final_df.get('MODALITY'))) & 
            (final_df['MODALITY'] == 'US') & 
            (pd.notna(final_df.get('BI-RADS'))) & 
            (final_df['BI-RADS'] == '6')
        ].index
        
        for idx in tqdm(us_birads6_rows, desc="Checking BI-RADS 6 cases"):
            row = final_df.loc[idx]
            if pd.isna(row['final_interpretation']) or row['final_interpretation'] == '':
                final_df.at[idx, 'final_interpretation'] = 'MALIGNANT1'
    
    return final_df


def check_from_next_diagnosis(final_df, days=240):
    """
    For 'US' rows with empty final_interpretation:
    - For MALIGNANT cases: Apply only to BI-RADS '4', '4A', '4B', '4C', '5', or '6'
      Set final_interpretation to 'MALIGNANT2' only if:
      1. The laterality matches between the US study and the pathology
      2. At least one record in the date range has 'is_us_biopsy' = 'T'
    
    - For BENIGN cases: Apply only to BI-RADS '1', '2', '3', '4', '4A', '4B'
      Set final_interpretation to 'BENIGN2' if:
      1. The laterality matches between the US study and the pathology
    
    Special case: If Study_Laterality is 'BILATERAL', consider any future laterality,
    and if any MALIGNANT is present within the time frame, set to 'MALIGNANT2'
    (still requiring at least one 'is_us_biopsy' = 'T').
    """
    # Define the target BI-RADS values for malignant and benign cases
    target_birads_malignant = ['4', '4A', '4B', '4C', '5', '6']
    target_birads_benign = ['1', '2', '3', '4', '4A', '4B']
    updates = {}
    
    for patient_id in tqdm(final_df['PATIENT_ID'].unique(), desc="Checking diagnosis from next record"):
        patient_mask = final_df['PATIENT_ID'] == patient_id
        patient_records = final_df[patient_mask].copy().sort_values('DATE')
        
        # Pre-filter to only include relevant US records for this patient
        relevant_records = patient_records[
            (patient_records['MODALITY'] == 'US') & 
            ((patient_records['final_interpretation'].isna()) | (patient_records['final_interpretation'] == '')) &
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
            
            # Check if there's at least one record with 'is_us_biopsy' = 'T' in the date range
            has_us_biopsy = any(
                (record['is_us_biopsy'] == 'T') 
                for _, record in future_records.iterrows() 
                if pd.notna(record.get('is_us_biopsy'))
            )
            
            # Handle BILATERAL case
            if current_laterality == 'BILATERAL':
                path_interpretations = []
                for _, future_row in future_records.iterrows():
                    if pd.notna(future_row.get('path_interpretation')):
                        path_interpretations.append(future_row['path_interpretation'].upper())
                
                # Apply malignant case only for target BI-RADS values
                if 'MALIGNANT' in path_interpretations and has_us_biopsy and current_birads in target_birads_malignant:
                    updates[idx] = 'MALIGNANT2'
                # Apply benign case only for target benign BI-RADS values, and only if NO malignant cases are present
                elif 'BENIGN' in path_interpretations and 'MALIGNANT' not in path_interpretations and current_birads in target_birads_benign:
                    updates[idx] = 'BENIGN2'

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
                
                # Apply malignant finding only for target BI-RADS values
                if found_malignant and has_us_biopsy and current_birads in target_birads_malignant:
                    updates[idx] = 'MALIGNANT2'
                # Apply benign finding only for target benign BI-RADS values, and only if NO malignant cases are found
                elif found_benign and not found_malignant and current_birads in target_birads_benign:
                    updates[idx] = 'BENIGN2'
    
    # Apply all updates at once
    for idx, value in updates.items():
        final_df.at[idx, 'final_interpretation'] = value
        
    return final_df


def determine_final_interpretation(final_df, output_path, rad_df_length):
    """
    Determine final_interpretation for each patient based on specified rules.
    """
    # Identify BENIGN1 cases based on follow-up period
    final_df = check_assumed_benign(final_df)
    
    # Identify BENIGN3 cases based on follow-up period (Birads 3)
    final_df = check_assumed_benign_birads3(final_df)
    
    # Identify MALIGNANT1 cases from biopsy results for remaining cases
    final_df = check_malignant_from_biopsy(final_df)
    
    # Identify BENIGN2 cases and MALIGNANT2 based on next chronological path_interpretation
    final_df = check_from_next_diagnosis(final_df)
    
    return final_df


def audit_interpretations(final_df):
    
    # After all processing, search for specific categories in the final_interpretation column
    benign1_count = sum(final_df['final_interpretation'] == 'BENIGN1')
    benign2_count = sum(final_df['final_interpretation'] == 'BENIGN2')
    benign3_count = sum(final_df['final_interpretation'] == 'BENIGN3')
    malignant1_count = sum(final_df['final_interpretation'] == 'MALIGNANT1')
    malignant2_count = sum(final_df['final_interpretation'] == 'MALIGNANT2')
    
    # Create audit log with counts for each category
    append_audit("query_clean.assumed_benign", benign1_count)
    append_audit("query_clean.birads3_benign", benign3_count)
    append_audit("query_clean.path_confirmed_benign", benign2_count)
    append_audit("query_clean.path_confirmed_malignant", malignant2_count)
    append_audit("query_clean.birads6_malignant", malignant1_count)
    
    # Calculate total benign (all benign categories)
    total_benign = benign1_count + benign2_count + benign3_count
    append_audit("query_clean.final_benign_count", total_benign)
    
    # Calculate total malignant (all malignant categories)
    total_malignant = malignant1_count + malignant2_count
    append_audit("query_clean.final_malignant_count", total_malignant)
    
    # Calculate unknown (cases without a final interpretation)
    total_rows = len(final_df)
    total_known = total_benign + total_malignant
    unknown_count = total_rows - total_known
    append_audit("query_clean.final_unknown_count", unknown_count)












def prepare_dataframes(rad_df, path_df):
    """Prepare and standardize dataframes for combining."""
    
    # Convert Patient_ID to string in both dataframes - use inplace for better performance
    rad_df['PATIENT_ID'] = rad_df['PATIENT_ID'].astype(str)
    path_df['PATIENT_ID'] = path_df['PATIENT_ID'].astype(str)
    
    # Convert date columns and rename in one step
    rad_df['DATE'] = pd.to_datetime(rad_df['RADIOLOGY_DTM'], errors='coerce')
    path_df['DATE'] = pd.to_datetime(path_df['SPECIMEN_RECEIVED_DTM'], errors='coerce')
    
    # Drop columns more efficiently (in-place)
    rad_df.drop('RADIOLOGY_DTM', axis=1, inplace=True)
    path_df.drop('SPECIMEN_RECEIVED_DTM', axis=1, inplace=True)
    
    return rad_df, path_df


def combine_dataframes(rad_df, path_df):
    """Combine radiology and pathology dataframes, keeping pathology on separate rows."""
    # Select only needed columns from path_df to reduce memory usage
    needed_columns = ['PATIENT_ID', 'DATE', 'SPECIMEN_RESULT_DTM', 'Pathology_Laterality', 'final_diag', 'path_interpretation']
    path_needed = path_df[needed_columns] if all(col in path_df.columns for col in needed_columns) else path_df
    
    # Create a copy of path_needed with the same columns as rad_df, plus any additional columns we need
    path_records_df = pd.DataFrame(columns=list(set(rad_df.columns) | set(path_needed.columns)))
    
    # Fill in values from path_needed
    for col in path_needed.columns:
        path_records_df[col] = path_needed[col]
    
    # Concatenate more efficiently with optimized settings
    final_df = pd.concat([rad_df, path_records_df], ignore_index=True, copy=False)
    
    return final_df

def audit_pathology_dates(df):
    """
    Calculate days from biopsy to pathology SPECIMEN_RESULT_DTM for each patient and record in the audit.
    Only considers cases where the DATE vs DATE difference is within 2 weeks.
    
    Also audits day distance from biopsy to the most recent row before that biopsy.
    
    Args:
        df: The combined dataframe with biopsy and pathology records
    """
    print("Auditing pathology dates from biopsies using SPECIMEN_RESULT_DTM")
    
    # Ensure DATE column is in datetime format
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    # Ensure SPECIMEN_RESULT_DTM is in datetime format
    df['SPECIMEN_RESULT_DTM'] = pd.to_datetime(df['SPECIMEN_RESULT_DTM'], errors='coerce')
    
    # Create empty lists to store days difference data
    days_differences = []
    exam_to_biopsy_differences = []
    
    # Get unique patients
    unique_patients = df['PATIENT_ID'].unique()
    
    for patient_id in unique_patients:
        # Get all records for this patient, sorted by DATE
        patient_records = df[df['PATIENT_ID'] == patient_id].copy()
        patient_records = patient_records.sort_values('DATE')
        
        # Find biopsy records
        biopsy_records = patient_records[patient_records['is_biopsy'] == 'T']
        
        # Find pathology records (where path_interpretation is not null)
        pathology_records = patient_records[pd.notna(patient_records['path_interpretation'])]
        
        # Audit 1: Biopsy to pathology differences (existing logic)
        if not biopsy_records.empty and not pathology_records.empty:
            # For each biopsy, find pathology records within 2 weeks
            for _, biopsy in biopsy_records.iterrows():
                biopsy_date = biopsy['DATE']
                
                if pd.isna(biopsy_date):
                    continue
                    
                # Look for pathology records with dates within 2 weeks of biopsy
                for _, pathology in pathology_records.iterrows():
                    pathology_date = pathology['DATE']
                    result_date = pathology['SPECIMEN_RESULT_DTM']
                    
                    if pd.isna(pathology_date) or pd.isna(result_date):
                        continue
                    
                    # Only process if DATE difference is within 2 weeks (14 days)
                    date_diff = (pathology_date - biopsy_date).days
                    if 0 <= date_diff <= 14:
                        continue
                    
                    # Calculate the difference between biopsy DATE and pathology SPECIMEN_RESULT_DTM
                    result_diff = (result_date - biopsy_date).days
                    days_differences.append(result_diff)
        
        # Audit 2: Most recent exam to biopsy differences (new logic)
        if not biopsy_records.empty:
            # For each biopsy, find the most recent row before it
            for _, biopsy in biopsy_records.iterrows():
                biopsy_date = biopsy['DATE']
                
                if pd.isna(biopsy_date):
                    continue
                
                # Find all records before this biopsy date
                previous_records = patient_records[
                    (patient_records['DATE'] < biopsy_date) & 
                    (pd.notna(patient_records['DATE']))
                ]
                
                if not previous_records.empty:
                    # Get the most recent record before the biopsy
                    most_recent_exam = previous_records.iloc[-1]  # Last one since sorted by DATE
                    most_recent_date = most_recent_exam['DATE']
                    
                    # Calculate days difference
                    exam_diff = (biopsy_date - most_recent_date).days
                    exam_to_biopsy_differences.append(exam_diff)
    
    # Record both audits
    append_audit("query_clean.pathology_date_from_biopsy", days_differences)
    append_audit("query_clean.exam_date_from_biospy", exam_to_biopsy_differences)
                
def create_final_dataset(rad_df, path_df, output_path):
    """Main function to create the final dataset with pathology records on separate rows."""
    print("Creating Final Dataset")
    
    # Prepare dataframes
    rad_df, path_df = prepare_dataframes(rad_df, path_df)
    rad_df_length = len(rad_df)
    path_df_length = len(path_df)
    
    # Combine dataframes
    final_df = combine_dataframes(rad_df, path_df)
    
    # Determine final interpretation
    final_df = determine_final_interpretation(final_df, output_path, rad_df_length)
    
    # Save to CSV
    final_df.to_csv(f'{output_path}/combined_dataset_debug.csv', index=False)

    audit_pathology_dates(final_df)
    
    # Filter to keep only rows with 'US' in MODALITY
    initial_count = len(final_df)
    final_df_us = final_df[final_df['MODALITY'].str.contains('US', na=False, case=False)]
    filtered_count = initial_count - len(final_df_us)

    append_audit("query_clean.rad_non_US_removed", filtered_count - path_df_length) # path_df_length were removed here but lets keep radiology context
    
    # Remove duplicate rows based on Accession_Number
    duplicate_accessions = final_df_us[final_df_us.duplicated(subset=['ACCESSION_NUMBER'], keep=False)]['ACCESSION_NUMBER']
    duplicate_count = len(final_df_us[final_df_us['ACCESSION_NUMBER'].isin(duplicate_accessions)])
    final_df_us = final_df_us[~final_df_us['ACCESSION_NUMBER'].isin(duplicate_accessions)]
    append_audit("query_clean.rad_duplicates_removed", duplicate_count)
    
    # Remove rows with empty ENDPOINT_ADDRESS
    empty_endpoint_count = sum(final_df_us['ENDPOINT_ADDRESS'].isna())
    final_df_us = final_df_us[final_df_us['ENDPOINT_ADDRESS'].notna()]
    append_audit("query_clean.rad_missing_address_removed", empty_endpoint_count)
    
    # Remove rows with empty BI-RADS
    empty_birads_count = sum(final_df_us['BI-RADS'].isna())
    final_df_us = final_df_us[final_df_us['BI-RADS'].notna()]
    append_audit("query_clean.rad_missing_birads_removed", empty_birads_count)
    
    # Count total interpretations
    audit_interpretations(final_df_us)
    
    # Remove rows with empty final_interpretation
    empty_interpretation_count = sum(final_df_us['final_interpretation'].isna())
    final_df_us = final_df_us[final_df_us['final_interpretation'].notna()]
    append_audit("query_clean.rad_missing_final_interp", empty_interpretation_count)
    

    # Extract STUDY_ID from ENDPOINT_ADDRESS
    final_df_us['STUDY_ID'] = final_df_us['ENDPOINT_ADDRESS'].apply(
        lambda url: url.split('/')[-1] if pd.notna(url) else None
    )
    
    # Save the US-only filtered dataset
    final_df_us.to_csv(f'{env}/raw_data/endpoint_data.csv', index=False)

    # Print statistics
    print(f"Removed {duplicate_count} rows with duplicate ACCESSION_NUMBER")
    print(f"Dataset passed with {len(final_df_us)} results")
    
    append_audit("query_clean.final_case_count", len(final_df_us))
    
    return final_df



if __name__ == "__main__":
    # Load the parsed radiology and pathology data
    try:
        output_path = os.path.join(env, "raw_data")
        rad_file_path = f'{output_path}/parsed_radiology.csv'
        path_file_path = f'{output_path}/parsed_pathology.csv'
        
        rad_df = pd.read_csv(rad_file_path)
        print(f"Loaded radiology data with {len(rad_df)} records")
        
        path_df = pd.read_csv(path_file_path)
        print(f"Loaded pathology data with {len(path_df)} records")
        
        # Call the create_final_dataset function
        create_final_dataset(rad_df, path_df, output_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you've run the parsing scripts to create the parsed CSV files first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")