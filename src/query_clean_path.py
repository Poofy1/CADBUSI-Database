import os
import pandas as pd
import re
from tools.audit import append_audit

# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory

def determine_laterality(row):
    """Determine laterality from pathology report, with improved handling of multi-part reports."""
    
    def check_text_for_laterality(text):
        if pd.isna(text):
            return None
        
        text = text.upper()
        
        # Track mentions of each side in multi-part reports
        right_mentions = 0
        left_mentions = 0
        
        # Split by lettered parts
        parts = re.split(r'(?:^|\s)([A-Z])[\.\)]\s+', text)
        
        # If no parts found, check the whole text
        if len(parts) <= 1:
            if "RIGHT" in text and "LEFT" in text:
                return None
            elif "RIGHT" in text and "BILATERAL" not in text:
                return "RIGHT"
            elif "LEFT" in text and "BILATERAL" not in text:
                return "LEFT"
        else:
            # Process each part separately
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    part_text = parts[i+1]
                    if "RIGHT" in part_text:
                        right_mentions += 1
                    if "LEFT" in part_text:
                        left_mentions += 1
            
            # Determine overall laterality based on part counts
            if right_mentions > 0 and left_mentions > 0:
                return None
            elif right_mentions > 0:
                return "RIGHT"
            elif left_mentions > 0:
                return "LEFT"
        
        # If no laterality is found
        return None
    
    # First try final_diag column if it exists
    if 'final_diag' in row and not pd.isna(row['final_diag']):
        laterality = check_text_for_laterality(row['final_diag'])
        if laterality is not None:
            return laterality
    
    # Then try PART_DESCRIPTION column
    if 'PART_DESCRIPTION' in row and not pd.isna(row['PART_DESCRIPTION']):
        laterality = check_text_for_laterality(row['PART_DESCRIPTION'])
        if laterality is not None:
            return laterality
    
    # If not found or previous columns are empty, try SPECIMEN_NOTE
    if 'SPECIMEN_NOTE' in row and not pd.isna(row['SPECIMEN_NOTE']):
        laterality = check_text_for_laterality(row['SPECIMEN_NOTE'])
        if laterality is not None:
            return laterality
    
    # If still not found, return None
    return None


def split_lesions(pathology_df):
    """
    Split pathology cases into separate rows by lettered parts,
    with laterality determination for each part.
    """
    print("Splitting cases by lettered parts with laterality...")
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
                    part_row['final_diag'] = f"{part_letter}. {part_text}"
                    
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
    start_match = re.search(r'FINAL DIAGNOSIS:?', text, re.IGNORECASE)
    if not start_match:
        return None
    
    # Get the position right after the match
    start_pos = start_match.end()
    
    # Get the content after the match
    after_diagnosis = text[start_pos:].strip()
    
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


def filter_path_data(pathology_df, output_path):
    print("Parsing Pathology Data")
    
    # Extract final diagnosis from SPECIMEN_NOTE
    pathology_df['final_diag'] = pathology_df['SPECIMEN_NOTE'].apply(extract_final_diagnosis)
    
    # Split cases into separate rows via lesions
    expanded_df = split_lesions(pathology_df)
    append_audit(output_path, f"{len(pathology_df)} pathology records had {len(expanded_df)} lesions")
    
    # Re-determine laterality after splitting (for rows that didn't have it set during splitting)
    expanded_df['Pathology_Laterality'] = expanded_df.apply(determine_laterality, axis=1)
    
    # Apply diagnosis classification
    expanded_df['path_interpretation'] = expanded_df['final_diag'].apply(categorize_pathology)
    
    # Extract Modality from SPECIMEN_COMMENT
    expanded_df['Modality'] = expanded_df['SPECIMEN_COMMENT'].apply(extract_modality)
    
    # Select columns for output
    columns_to_keep = [
        'PATIENT_ID', 
        'ENCOUNTER_ID', 
        'SPECIMEN_ACCESSION_NUMBER',
        'Pathology_Laterality',
        'final_diag',
        'path_interpretation',
        'Modality',
        'DIAGNOSIS_NAME', 
        'SPECIMEN_COMMENT',
        'SPECIMEN_PART_TYPE_NAME',
        'SPECIMEN_ACCESSION_DTM',
        'SPECIMEN_RESULT_DTM',
        'SPECIMEN_RECEIVED_DTM',
    ]
    
    # Create output dataframe with selected columns
    output_df = expanded_df[columns_to_keep].copy()
    
    # Remove duplicate rows
    rows_before = len(output_df)
    output_df = output_df.drop_duplicates(keep='first')
    rows_after = len(output_df)
    duplicates_removed = rows_before - rows_after
    
    print(f"Removed {duplicates_removed} exact duplicate rows.")
    append_audit(output_path, f"Removed {duplicates_removed} pathology lesions - Duplicates")
    append_audit(output_path, f"Pathology lesion count: {rows_after}")

    
    # Save to CSV
    output_df.to_csv(f'{output_path}/parsed_pathology.csv', index=False)
    
    return output_df

if __name__ == "__main__":
    pathology_df = pd.read_csv(f'{env}/raw_data/raw_pathology.csv')
    filter_path_data(pathology_df)