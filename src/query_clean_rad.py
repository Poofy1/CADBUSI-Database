
import os
import pandas as pd
import re
from tools.audit import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory


def determine_laterality(row):
    # Function to check a single text field
    def check_text_for_laterality(text, right_text, left_text):
        if pd.isna(text):
            return None
        
        text = text.upper()
        
        # Check for clear RIGHT indicators
        if any(x in text for x in right_text) and "BILATERAL" not in text:
            return "RIGHT"
        
        # Check for clear LEFT indicators
        elif any(x in text for x in left_text) and "BILATERAL" not in text:
            return "LEFT"
        
        # Check for BILATERAL indicators
        elif "BILATERAL" in text or "BOTH" in text:
            return "BILATERAL"
        
        # If no laterality is found, return None
        else:
            return None
    
    # First try DESCRIPTION column
    if 'DESCRIPTION' in row and not pd.isna(row['DESCRIPTION']):
        laterality = check_text_for_laterality(row['DESCRIPTION'], ["RIGHT", "R BI", " RT", "RT "], ["LEFT", "L BI", " LT", "LT "])
        if laterality is not None:
            return laterality
        
    # Then try DESCRIPTION column
    if 'TEST_DESCRIPTION' in row and not pd.isna(row['TEST_DESCRIPTION']):
        laterality = check_text_for_laterality(row['TEST_DESCRIPTION'], ["RIGHT", "R BI",], ["LEFT", "L BI",])
        if laterality is not None:
            return laterality
    
    # If not found or DESCRIPTION is empty, try RADIOLOGY_REPORT
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        laterality = check_text_for_laterality(row['RADIOLOGY_REPORT'], ["RIGHT", "R BI"], ["LEFT", "L BI"])
        if laterality is not None:
            return laterality
    
    # If still not found, return None
    return None


def extract_birads_and_description(row):
    # First try RADIOLOGY_REPORT if available
    if 'RADIOLOGY_REPORT' in row and not pd.isna(row['RADIOLOGY_REPORT']):
        text = row['RADIOLOGY_REPORT']
        result = extract_birads_from_text(text)
        if result[0] is not None:  # If BI-RADS was found in RADIOLOGY_REPORT
            return result
    
    # If no result from RADIOLOGY_REPORT, try RADIOLOGY_NARRATIVE
    if 'RADIOLOGY_NARRATIVE' in row and not pd.isna(row['RADIOLOGY_NARRATIVE']):
        text = row['RADIOLOGY_NARRATIVE']
        result = extract_birads_from_text(text)
        if result[0] is not None:  # If BI-RADS was found in RADIOLOGY_NARRATIVE
            return result
    
    return None, None

def extract_birads_from_text(text):
    if pd.isna(text):
        return None, None
    
    # List of keywords that should end a description
    end_keywords = [
        'benign', 'malignant', 'malignancy', 'suspicious', 
        'negative', 'positive', 'cancer', 'indeterminate', 'incomplete'
    ]
    
    # Create pattern to find any of the keywords
    end_pattern = r'(?i)(' + '|'.join(end_keywords) + r')[^\w]'
    
    patterns = [
        # "BI-RADS ASSESSMENT: CODE: X-DESCRIPTION" format
        r'BI-?RADS\s+ASSESSMENT:\s*CODE:\s*(\d+[a-z]?)-([^\.\s]+)',
        r'BI-?RADS:\s*Code\s*(\d+[a-z]?),\s*([^\.\s]+)',
        
        # Ultrasound-specific patterns with description
        r'(?:Ultrasound|US)\s+BI-?RADS:\s*\(?(\d+[a-z]?)\)?\s+([^\s\.]+)',
        
        # General BI-RADS patterns with descriptions - Combined several patterns
        r'(?:BI-?RADS\s*(?:ASSESSMENT|Category|code|Final\s+Assessment)?|ASSESSMENT:\s*BI-?RADS|OVERALL\s*STUDY\s*BI-?RADS)(?:\s*CATEGORY)?[:]?\s*\(?(\d+[a-z]?)\)?(?:[:]\s*|\s+|,\s*|\s*-\s*)([^\.]+)(?:\.)?',
        
        # Special case for impression
        r'(?:IMPRESSION:|ASSESSMENT:)?\s*(?:BI-?RADS)\s*\(?(\d+[a-z]?)\)?\s*(?:-|,)\s*([^\.]+)',
        
        # Special case for description before number
        r'BI-?RADS:\s*([^(]+)\s*\((\d+[a-z]?)\)',
        
        # Final Assessment without BI-RADS explicitly mentioned
        r'Final\s+Assessment:\s*\(?(\d+[a-z]?)\)?\s*(?:-|,)?\s*([^\.]+)',
        
        # Pattern for BI-RADS ATLAS category
        r'BI-?RADS(?:®\s*ATLAS)?\s*category\s*\(?overall\)?:\s*\(?(\d+[a-z]?)\)?\s+([^\.]+)',
        
        # Pattern for BI-RADS® Category with registered trademark
        r'BI-?RADS®\s*Category:\s*(\d+[a-z]?)\s*-\s*([^\.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and len(match.groups()) >= 2:
            # Special case for the "description before number" pattern
            if pattern == r'BI-?RADS:\s*([^(]+)\s*\((\d+[a-z]?)\)':
                birads_category = match.group(2)
                full_description = match.group(1).strip()
            else:
                birads_category = match.group(1)
                full_description = match.group(2).strip()
            
            # Convert any letters in the BI-RADS category to uppercase
            if birads_category:
                birads_category = ''.join([c.upper() if c.isalpha() else c for c in birads_category])
            
            # Truncate description at any of the specified keywords
            keyword_match = re.search(end_pattern, full_description + ' ')
            if keyword_match:
                # Get the position of the keyword plus its length
                key_end_pos = keyword_match.end() - 1  # -1 to exclude the non-word character
                description = full_description[:key_end_pos].strip()
            else:
                description = full_description
                
            return birads_category, description
    
    # Simpler patterns without description capturing
    for pattern in [
        r'(?:Ultrasound|US)\s+BI-?RADS:\s*\(?(\d+[a-z]?)\)?',
        r'(?:BI-?RADS|BIRADS)(?:\s*(?:Category|CATEGORY|code))?(?::|:?\s+CATEGORY)?\s*\(?(\d+[A-Za-z]?)\)?',
        r'OVERALL\s*STUDY\s*BI-?RADS:\s*\(?(\d+[A-Za-z]?)\)?',
        r'BI-?RADS\s+Category\s+No\.\s*(\d+[a-z]?)',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            birads_category = match.group(1)
            # Convert any letters in the BI-RADS category to uppercase
            if birads_category:
                birads_category = ''.join([c.upper() if c.isalpha() else c for c in birads_category])
            return birads_category, None
    
    # Special case for assessment with pathology but no explicit BI-RADS
    pathology_match = re.search(r'ASSESSMENT:\s*\d+:\s*(Pathology\s+\w+)', text, re.IGNORECASE)
    if pathology_match:
        return None, pathology_match.group(1).strip()
    
    return None, None


def extract_density(text):
    # Extract text after "DENSITY:" until next section header (WORD:)
    if pd.isna(text):
        return None
    
    # Check if "DENSITY:" exists in the text
    if "DENSITY:" not in text:
        return None
    
    # Split by "DENSITY:" and get the content after it
    after_density = text.split("DENSITY:")[1].strip()
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_density)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "DENSITY:" until the next section header
        density_text = after_density[:end_pos].strip()
        return density_text
    else:
        # If no next section header is found, return all text after "DENSITY:"
        return after_density


def extract_rad_pathology_txt(text):
    if pd.isna(text):
        return None
    
    # Check if "PATHOLOGY:" exists in the text
    if "PATHOLOGY:" not in text:
        return None
    
    # Split by "PATHOLOGY:" and get the content after it
    after_pathology = text.split("PATHOLOGY:")[1].strip()
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_pathology)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "PATHOLOGY:" until the next section header
        pathology_text = after_pathology[:end_pos].strip()
        return pathology_text
    else:
        # If no next section header is found, return all text after "PATHOLOGY:"
        return after_pathology

def check_for_biopsy(row):
    """
    Check for biopsy and ultrasound biopsy in DESCRIPTION or TEST_DESCRIPTION columns
    
    Args:
        row: The dataframe row with columns to check
        
    Returns:
        Tuple of (biopsy_found, ultrasound_biopsy_found) where each is 'T' if found, 'F' if not
    """
    biopsy_found = 'F'
    ultrasound_biopsy_found = 'F'
    
    # Check DESCRIPTION column
    if 'DESCRIPTION' in row and not pd.isna(row['DESCRIPTION']):
        description_upper = row['DESCRIPTION'].upper()
        
        # Check for biopsy
        if 'BIOPSY' in description_upper or 'BX' in description_upper or 'ASP' in description_upper:
            biopsy_found = 'T'
            
            # Check if it's an ultrasound biopsy in this column
            if ('US' in description_upper or 'ULTRASOUND' in description_upper or 
                re.search(r'\bUL\b', description_upper)):
                ultrasound_biopsy_found = 'T'
    
    # Check TEST_DESCRIPTION column
    if 'TEST_DESCRIPTION' in row and not pd.isna(row['TEST_DESCRIPTION']):
        test_description_upper = row['TEST_DESCRIPTION'].upper()
        
        # Check for biopsy
        if 'BIOPSY' in test_description_upper or 'BX' in test_description_upper or 'ASP' in test_description_upper:
            biopsy_found = 'T'
            
            # Check if it's an ultrasound biopsy in this column
            if ('US' in test_description_upper or 'ULTRASOUND' in test_description_upper or 
                re.search(r'\bUL\b', test_description_upper)):
                ultrasound_biopsy_found = 'T'
    
    return biopsy_found, ultrasound_biopsy_found


def extract_rad_impression(text):
    if pd.isna(text):
        return None
    
    # Check if "IMPRESSION:" exists in the text
    if "IMPRESSION:" not in text:
        return None
    
    # Check if "IMPRESSION:" or "IMPRESSION" exists in the text
    if "IMPRESSION:" in text:
        # Split by "IMPRESSION:" and get the content after it
        after_impression = text.split("IMPRESSION:", 1)[1].strip()
    elif "IMPRESSION" in text:
        # Split by "IMPRESSION" and get the content after it
        after_impression = text.split("IMPRESSION", 1)[1].strip()
        # Remove leading colon if it exists
        if after_impression.startswith(":"):
            after_impression = after_impression[1:].strip()
    else:
        return None
    
    # Use regex to find the next uppercase word followed by a colon
    match = re.search(r'([A-Z]{2,}:)', after_impression)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "IMPRESSION:" until the next section header
        impression_text = after_impression[:end_pos].strip()
        return impression_text
    else:
        # If no next section header is found, return all text after "IMPRESSION:"
        return after_impression
    
def remove_outside_records(radiology_df):
    """
    Remove rows where 'OUTSIDE' appears in the TEST_DESCRIPTION column
    
    Args:
        radiology_df: DataFrame containing radiology data
        
    Returns:
        DataFrame with outside records removed
    """
    initial_row_count = len(radiology_df)
    
    # Make a copy to avoid warnings about setting values on a slice
    filtered_df = radiology_df.copy()
    
    # Check if the column exists
    if 'TEST_DESCRIPTION' in filtered_df.columns:
        # Create a mask for rows where 'OUTSIDE' is not in TEST_DESCRIPTION
        # Handle NaN values with a boolean mask
        mask = ~filtered_df['TEST_DESCRIPTION'].fillna('').str.upper().str.contains('OUTSIDE')
        
        # Apply the mask to filter out rows with 'OUTSIDE'
        filtered_df = filtered_df[mask]
    
    # Calculate how many rows were removed
    removed_count = initial_row_count - len(filtered_df)
    print(f"Removed {removed_count} outside records")
    
    return filtered_df



def remove_bad_data(radiology_df, output_path):
    # Count and remove rows with BI-RADS = '0'
    birads_zero_mask = radiology_df['BI-RADS'].isin(['0'])
    birads_zero_count = birads_zero_mask.sum() 
    radiology_df = radiology_df[~birads_zero_mask]
    
    # Remove patients without any 'US' modality exams
    # First, find patients who have at least one 'US' modality
    patients_with_us = radiology_df[radiology_df['MODALITY'] == 'US']['PATIENT_ID'].unique()
    
    # Count patients to be removed
    patients_to_remove = set(radiology_df['PATIENT_ID'].unique()) - set(patients_with_us)
    patients_removed_count = len(patients_to_remove)
    
    # Keep only patients who have at least one 'US' modality
    radiology_df = radiology_df[radiology_df['PATIENT_ID'].isin(patients_with_us)]
    
    
    print(f"Removed {birads_zero_count} rows with BI-RADS = '0'")
    print(f"Removed {patients_removed_count} patients without any 'US' modality exams (after previous removals)")
    append_audit(output_path, f"Removed {birads_zero_count} radiology records - with BI-RADS = '0'")
    append_audit(output_path, f"Removed {patients_removed_count} radiology patients - missing >=1 'US' modality exam (after previous removals)")
    
    return radiology_df
    
def filter_rad_data(radiology_df, output_path):
    print("Parsing Radiology Data:")
    
    # Print length
    initial_count = len(radiology_df)
    print(f"Initial dataframe length: {initial_count} rows")

    rename_dict = {'PAT_PATIENT_CLINIC_NUMBER': 'PATIENT_ID',
        'IMGST_ACCESSION_IDENTIFIER_VALUE': 'Accession_Number',
        'IMGST_DESCRIPTION': 'Biopsy_Desc',}
    
    # Rename columns
    radiology_df = radiology_df.rename(columns=rename_dict)
    
    # Remove outside records
    count_before = len(radiology_df)
    radiology_df = remove_outside_records(radiology_df)
    outside_removed = count_before - len(radiology_df)
    append_audit(output_path, f"Removed {outside_removed} radiology records - Labeled as 'outside'")
    
    # Apply the extraction functions and create new columns
    radiology_df['Density_Desc'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_density)
    
    # Apply the BI-RADS extraction and create separate columns
    birads_results = radiology_df.apply(extract_birads_and_description, axis=1)
    radiology_df['BI-RADS'] = [result[0] for result in birads_results]
    radiology_df['Biopsy'] = [result[1] for result in birads_results]
    
    # Find Laterality 
    radiology_df['Study_Laterality'] = radiology_df.apply(determine_laterality, axis=1)
    
    # Extract pathology text
    radiology_df['rad_pathology_txt'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_rad_pathology_txt)
    
    # Extract impression text
    radiology_df['rad_impression'] = radiology_df['RADIOLOGY_REPORT'].apply(extract_rad_impression)

    # Check for biopsy in DESCRIPTION column
    results = radiology_df.apply(check_for_biopsy, axis=1)
    radiology_df['is_biopsy'] = results.str[0]
    radiology_df['is_us_biopsy'] = results.str[1]
            
    pd.set_option('display.max_colwidth', None)
    # Columns to drop
    columns_to_drop = ['RADIOLOGY_NARRATIVE', 'PROCEDURE_CODE_TEXT', 'SERVICE_RESULT_STATUS', 'RADIOLOGY_REPORT', 'RAD_SERVICE_RESULT_STATUS']
    radiology_df = radiology_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Remove bad data
    radiology_df = remove_bad_data(radiology_df, output_path)
    
    # Print final length
    final_count = len(radiology_df)
    append_audit(output_path, f"Total radiology record count: {final_count}")
    print(f"Final dataframe length: {len(radiology_df)} rows")
    
    # Save output
    radiology_df.to_csv(f'{output_path}/parsed_radiology.csv', index=False)
    
    return radiology_df
    
    
if __name__ == "__main__":
    rad_df = pd.read_csv(f'{env}/raw_data/raw_radiology.csv')
    filter_rad_data(rad_df)