import os
import pandas as pd
import re
from src.DB_processing.tools import append_audit

# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory

def determine_lesion_laterality(lesion_text):
    """Simple laterality determination for individual lesion diagnosis."""
    if pd.isna(lesion_text):
        return "UNKNOWN"
    
    text = str(lesion_text).upper()
    
    has_right = "RIGHT" in text
    has_left = "LEFT" in text
    
    if has_right and has_left:
        return "UNKNOWN"  # Both sides mentioned
    elif has_right:
        return "RIGHT"
    elif has_left:
        return "LEFT"
    else:
        return "UNKNOWN"  # Neither mentioned


def split_lesions(pathology_df):
    """
    Split pathology cases into separate rows by lettered parts,
    with laterality determination for each part.
    """
    expanded_rows = []
    part_count = 0
    
    for idx, row in pathology_df.iterrows():
        if pd.isna(row['final_diag']):
            # Keep the original row for cases with no final diagnosis
            expanded_rows.append(row.to_dict())
            continue
        
        text = str(row['final_diag']).upper()
        
        # Split by lettered parts
        parts = re.split(r'(?:^|\s)([A-Z])[\.\)]\s+', text)
        
        # Process parts (every odd index is a letter, followed by content)
        valid_parts_found = False
        i = 1
        while i < len(parts):
            if i+1 < len(parts):
                part_letter = parts[i]
                part_text = parts[i+1].strip()
                
                # Only include parts that have actual content
                if part_text:
                    valid_parts_found = True
                    part_count += 1
                    part_row = row.to_dict()
                    part_row['lesion_diag'] = f"{part_letter}. {part_text}"  # New column instead of overwriting final_diag
                    
                    # Determine laterality for this part
                    if "LEFT" in part_text:
                        part_row['Pathology_Laterality'] = "LEFT"
                    elif "RIGHT" in part_text:
                        part_row['Pathology_Laterality'] = "RIGHT"
                    else:
                        part_row['Pathology_Laterality'] = "UNSPECIFIED"
                        
                    part_row['Pathology_Part'] = part_letter
                    expanded_rows.append(part_row)
            i += 2
        
        # If no valid parts were found, keep the original row with laterality
        if not valid_parts_found:
            original_row = row.to_dict()
            original_row['lesion_diag'] = None  # No individual lesion for unsplit cases
            
            # Check for laterality in the full text
            if "LEFT" in text:
                original_row['Pathology_Laterality'] = "LEFT"
            elif "RIGHT" in text:
                original_row['Pathology_Laterality'] = "RIGHT"
            else:
                original_row['Pathology_Laterality'] = "UNSPECIFIED"
                
            expanded_rows.append(original_row)
    
    # Create a new dataframe from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"Split {part_count} parts into separate rows.")
    return expanded_df

def extract_final_diagnosis(text):
    if pd.isna(text):
        return None
    
    # Use regex to find "FINAL DIAGNOSIS:" or "FINAL DIAGNOSIS"
    start_match = re.search(r'FINAL DIAGNOSIS:?', text)
    if not start_match:
        return None
    
    # Get the position right after the match
    start_pos = start_match.end()
    
    # Get the content after the match
    after_diagnosis = text[start_pos:].strip()
    
    # Check if the text (after skipping whitespace) starts with A. or A)
    if not re.match(r'^\s*A[\.\)]', after_diagnosis):
        # If it doesn't start with A. or A), prepend "A. "
        after_diagnosis = "A. " + after_diagnosis.lstrip()
    
    # Look for either pattern: 
    # 1. Uppercase words (4+ letters) followed by colon
    # 2. Uppercase words (4+ letters) followed by space and dash
    next_section_pattern = r'\s+([A-Z]{4,}:|\b[A-Z][A-Z\s]{3,}\s+-)'
    
    match = re.search(next_section_pattern, after_diagnosis)
    
    if match:
        # Get position of the next section header
        end_pos = match.start()
        # Extract text from after "FINAL DIAGNOSIS:" until the next section header
        diagnosis_text = after_diagnosis[:end_pos].strip()
        return diagnosis_text
    else:
        # If no next section header is found, return all text after "FINAL DIAGNOSIS:"
        return after_diagnosis

def extract_modality(text):
   if pd.isna(text):
       return None
   
   text = str(text).upper()
   
   # Pattern to match "MODALITY:" followed by content until the next word ending with ":"
   modality_pattern = r'MODALITY:\s*([^:]+?)(?=\s+\w+:|$)'
   
   match = re.search(modality_pattern, text)
   if match:
       return match.group(1).strip()  # Return the captured modality value
   
   return None


def categorize_pathology(text):
    if pd.isna(text):
        return "UNKNOWN"
    
    # Convert to uppercase for consistent matching
    text = str(text).upper()
    
    # 1. Check for malignant findings first
    malignant_patterns = [
        r"INVASIVE\s+DUCTAL\s+CARCINOMA",
        r"DUCTAL\s+CARCINOMA\s+IN\s+SITU",
        r"\bDCIS\b",
        r"METASTATIC\s+CARCINOMA",
        r"INVASIVE\s+CARCINOMA",
        r"\bCARCINOMA\b",
        r"\bMALIGNAN[CT]\b",
        r"\bTUMOR\b",
        r"METASTATIC",
    ]
    
    # Flag to track if we found any non-negated malignant findings
    found_malignant = False
    
    for pattern in malignant_patterns:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            # Get context around the match (50 characters before)
            start_pos = max(0, match.start() - 50)
            context_before = text[start_pos:match.start()]
            
            # Check if this is a negated finding
            if not re.search(r"NEGATIVE\s+FOR|NO\s+EVIDENCE\s+OF|FREE\s+OF|ABSENCE\s+OF|NO\s+", context_before):
                found_malignant = True
                break
        
        if found_malignant:
            break
    
    if found_malignant:
        return "MALIGNANT"
    
    # 2. Check for explicit negation patterns
    negation_patterns = [
        r"NEGATIVE\s+FOR\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|ATYPIA|TUMOR|NEOPLASM|METASTATIC)",
        r"NO\s+EVIDENCE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM|ATYPIA)",
        r"ABSENCE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM)",
        r"FREE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|TUMOR|NEOPLASM)",
        r"NO\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|TUMOR|NEOPLASM)\s+SEEN",
        r"NEGATIVE\s+LYMPH",
        r"NO\s+MALIGNANCY\s+PRESENT",
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, text):
            return "BENIGN"
    
    # 3. Check for benign indicators
    benign_patterns = [
        r"\bBENIGN\b",
        r"FIBROCYSTIC",
        r"FIBROADENOMA",
        r"NORMAL\s+BREAST\s+TISSUE",
        r"INTRADUCTAL\s+PAPILLOMA",
        r"DUCT\s+ECTASIA",
        r"PERIDUCTAL\s+FIBROSIS",
        r"SCLEROSING\s+ADENOSIS",
        r"APOCRINE\s+METAPLASIA",
        r"USUAL\s+DUCTAL\s+HYPERPLASIA",
        r"ATYPICAL\s+DUCTAL\s+HYPERPLASIA", 
        r"COLUMNAR\s+CELL\s+CHANGE",
        r"RADIAL\s+SCAR",
        r"FIBROSIS",
        r"BREAST\s+IMPLANT",
        r"RUPTURED\s+IMPLANT",
        r"IMPLANT\s+CAPSULE",
        r"GROSS\s+ONLY\s+AS\s+DESCRIBED",
        r"FAT\s+NECROSIS",
        r"UNREMARKABLE",
        r"ESTROGEN",
        r"DYSTROPHIC\s+CALCIFICATIONS?",
        r"NEGATIVE",
        r"SCAR",
        r"FIBROTIC ",
    ]
    
    for pattern in benign_patterns:
        if re.search(pattern, text):
            return "BENIGN"
    
    return "UNKNOWN"


def extract_synoptic_report(text):
    """Extract synoptic report content from specimen note text."""
    if pd.isna(text):
        return None
    
    # Use regex to find "SYNOPTIC REPORT" (must be capitalized), optionally followed by ":"
    start_match = re.search(r'SYNOPTIC REPORT:?', text)
    if not start_match:
        return None
    
    # Get the position right after the match
    start_pos = start_match.end()
    
    # Get the content after the match
    after_synoptic = text[start_pos:].strip()
    
    return after_synoptic
    
    
def filter_path_data(pathology_df, output_path):
    print("Parsing Pathology Data")
    
    rows_before = len(pathology_df)
    pathology_df = pathology_df.drop_duplicates(keep='first')
    rows_after = len(pathology_df)
    duplicates_removed_early = rows_before - rows_after
    print(f"Removed {duplicates_removed_early} exact duplicate rows before processing.")
    
    # Extract final diagnosis from SPECIMEN_NOTE
    pathology_df['final_diag'] = pathology_df['SPECIMEN_NOTE'].apply(extract_final_diagnosis)
    
    # Extract synoptic report from SPECIMEN_NOTE
    pathology_df['SYNOPTIC_REPORT'] = pathology_df['SPECIMEN_NOTE'].apply(extract_synoptic_report)
    
    # Split cases into separate rows via lesions
    expanded_df = split_lesions(pathology_df)
    append_audit("query_clean_path.path_pre_lesion_count", len(pathology_df))
    append_audit("query_clean_path.path_post_lesion_count", len(expanded_df))
    
    # Re-determine laterality after splitting (for rows that didn't have it set during splitting)
    expanded_df['Pathology_Laterality'] = expanded_df['lesion_diag'].apply(determine_lesion_laterality)
    
    # Apply diagnosis classification
    expanded_df['path_interpretation'] = expanded_df['lesion_diag'].apply(categorize_pathology)
    
    # Audit laterality counts
    lat_left_count = len(expanded_df[expanded_df['Pathology_Laterality'] == 'LEFT'])
    lat_right_count = len(expanded_df[expanded_df['Pathology_Laterality'] == 'RIGHT'])
    lat_unknown_count = len(expanded_df[expanded_df['Pathology_Laterality'].isin(['UNKNOWN', None, ''])])
    append_audit("query_clean_path.lat_left", lat_left_count)
    append_audit("query_clean_path.lat_right", lat_right_count)
    append_audit("query_clean_path.lat_unknown", lat_unknown_count)
    
    # Audit interpretation counts
    interp_benign_count = len(expanded_df[expanded_df['path_interpretation'] == 'BENIGN'])
    interp_malignant_count = len(expanded_df[expanded_df['path_interpretation'] == 'MALIGNANT'])
    interp_unknown_count = len(expanded_df[expanded_df['path_interpretation'].isin(['UNKNOWN', None, ''])])
    append_audit("query_clean_path.interp_benign", interp_benign_count)
    append_audit("query_clean_path.interp_malignant", interp_malignant_count)
    append_audit("query_clean_path.interp_unknown", interp_unknown_count)
    
    # Extract Modality from SPECIMEN_COMMENT
    expanded_df['Modality'] = expanded_df['SPECIMEN_COMMENT'].apply(extract_modality)
    
    # Select columns for output
    columns_to_keep = [
        'PATIENT_ID', 
        'ENCOUNTER_ID', 
        'SPECIMEN_ACCESSION_NUMBER',
        'Pathology_Laterality',
        'final_diag',
        'lesion_diag',
        'path_interpretation',
        'Modality',
        'SYNOPTIC_REPORT',
        'DIAGNOSIS_NAME', 
        'SPECIMEN_COMMENT',
        'SPECIMEN_PART_TYPE_NAME',
        'SPECIMEN_RESULT_DTM',
        'SPECIMEN_RECEIVED_DTM',
    ]
    
    # Create output dataframe with selected columns
    output_df = expanded_df[columns_to_keep].copy()
    
    # Remove duplicate rows
    rows_before = len(output_df)
    output_df = output_df.drop_duplicates(keep='first')
    rows_after = len(output_df)
    duplicates_removed = (rows_before - rows_after) + duplicates_removed_early
    
    print(f"Removed {duplicates_removed} exact duplicate rows.")
    append_audit("query_clean_path.path_duplicated_removed", duplicates_removed)
    append_audit("query_clean_path.path_lesion_count_final", rows_after)

    
    # Save to CSV
    output_df.to_csv(f'{output_path}/parsed_pathology.csv', index=False)
    
    return output_df

if __name__ == "__main__":
    pathology_df = pd.read_csv(f'{env}/raw_data/raw_pathology.csv')
    filter_path_data(pathology_df)