import os
import pandas as pd
import re
from src.DB_processing.tools import append_audit

# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory
env = os.path.dirname(env)  # Go back one directory

def determine_lesion_laterality(lesion_text):
    """Simple laterality determination for individual lesion diagnosis."""
    if pd.isna(lesion_text):
        return "UNKNOWN"
    
    text = str(lesion_text).upper().replace(" ", "")
    
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
        
        # Split by lettered parts - only at start or after newline
        parts = re.split(r'(?:^|\n)([A-Z])[\.\)]\s+', text)
        
        # Process parts (every odd index is a letter, followed by content)
        valid_parts_found = False
        i = 1
        while i < len(parts):
            if i+1 < len(parts):
                part_letter = parts[i]
                part_text = parts[i+1].strip()
                
                # Truncate at any boilerplate section header that follows the
                # actual diagnosis (case-insensitive). The earliest match wins.
                text_upper = part_text.upper()
                truncate_markers = ["COMMENT", "REPORT", "CLINICAL HISTORY", "CASE NUMBER"]
                positions = [p for p in (text_upper.find(m) for m in truncate_markers) if p != -1]
                if positions:
                    part_text = part_text[:min(positions)].strip()
                
                # Only include parts that have actual content
                if part_text:
                    valid_parts_found = True
                    part_count += 1
                    part_row = row.to_dict()
                    part_row['lesion_diag'] = f"{part_letter}. {part_text}"  # New column instead of overwriting final_diag
                    expanded_rows.append(part_row)
            i += 2
        
        # If no valid parts were found, keep the original row
        if not valid_parts_found:
            original_row = row.to_dict()
            original_row['lesion_diag'] = None  # No individual lesion for unsplit cases
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
        r"CARCINOMA",
        r"\bMALIGNAN[CT]\b",
        r"\bTUMOR\s+CELLS\b",
        r"METASTATIC",
        r"\bMELANOMA\b",
    ]
    
    # Flag to track if we found any non-negated malignant findings
    found_malignant = False
    
    for pattern in malignant_patterns:
        for match in re.finditer(pattern, text):
            start_pos = max(0, match.start() - 50)
            context_before = text[start_pos:match.start()]
            context_after  = text[match.end():match.end() + 30]

            # Negated mention -> skip
            if re.search(r"NEGATIVE\s+FOR|NO\s+EVIDENCE\s+OF|FREE\s+OF|ABSENCE\s+OF|NOT\s+DIAGNOSTIC|NO\s+", context_before):
                continue

            # LCIS (lobular carcinoma in situ) is clinically a high-risk benign
            # lesion, not invasive cancer. Skip when "LOBULAR " precedes and
            # "IN SITU" follows the match. PLEOMORPHIC LCIS is also caught
            # because LOBULAR still appears immediately before CARCINOMA.
            if (re.search(r"\bLOBULAR\s*$", context_before) and
                    re.search(r"^[\s,\-]*IN[\s\-]+SITU", context_after)):
                continue

            found_malignant = True
            break

        if found_malignant:
            break
    
    if found_malignant:
        return "MALIGNANT"
    
    # 2. Check for explicit negation patterns
    negation_patterns = [
        r"NEGATIVE\s+FOR\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|ATYPIA|TUMOR|NEOPLASM|METASTATIC)",
        r"NO\s+EVIDENCE\s+OF\s+(RESIDUAL\s+)?(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM|ATYPIA)",
        r"ABSENCE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|DCIS|TUMOR|NEOPLASM)",
        r"FREE\s+OF\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|TUMOR|NEOPLASM)",
        r"NO\s+(MALIGNAN[CT]|CARCINOMA|INVASIVE|TUMOR|NEOPLASM)\s+SEEN",
        r"NEGATIVE\s+LYMPH",
        r"NO\s+MALIGNANCY\s+PRESENT",
        r"NO\s+RESIDUAL\s+(INVASIVE|CARCINOMA|DCIS|TUMOR|DISEASE|MALIGNAN[CT])",
        r"NO\s+LYMPH\s+NODE\s+TISSUE",
        r"WITHOUT\s+(HISTOLOGIC(AL)?\s+|SIGNIFICANT\s+)?ABNORMALITY",
        r"NO\s+HISTOLOGIC(AL)?\s+ABNORMALITY",
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
        r"BREAST\s+CAPSULE",
        r"FIBROMUSCULAR\s+TISSUE",
        r"FIBROADIPOSE\s+TISSUE",
        r"FIBROUS\s+TISSUE",
        r"BIOPSY\s+SITE\s+CHANGES?",
        r"NEUROFIBROMA",
        r"\bNEVUS\b",
        r"PSEUDOANGIOMATOUS",
        r"\bNODULAR\s+FASCIITIS\b",
        r"\bLCIS\b",
        r"LOBULAR\s+CARCINOMA\s+IN[\s\-]+SITU",
        r"ATYPICAL\s+LOBULAR\s+HYPERPLASIA",
        r"\bALH\b",
        r"LOBULAR\s+NEOPLASIA",
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


def _is_pre_2018(accession_number):
    if pd.isna(accession_number):
        return False
    return str(accession_number).lower().startswith('amml')


def _split_rtf_blobs(specimen_note):
    if pd.isna(specimen_note):
        return []
    parts = re.split(r'(?=\{\\rtf1)', str(specimen_note))
    return [p for p in parts if p.strip().startswith('{\\rtf1')]


def _strip_rtf(rtf_blob):
    if not rtf_blob:
        return ""
    text = rtf_blob

    # Drop the RTF header up to the content marker. The font/color/stylesheet
    # tables before `\plain\fN\fsNN ` are noise.
    marker = re.search(r'\\plain\\f\d+\\fs\d+\s?', text)
    if marker:
        text = text[marker.end():]

    text = re.sub(r'\\par\b', '\n', text)
    text = re.sub(r'\\tab\b', ' ', text)
    # Hex escapes (e.g. \'93 for typographic quote) — drop to a space.
    text = re.sub(r"\\'[0-9a-fA-F]{2}", ' ', text)
    # Remaining control words: `\name`, optionally followed by an integer and
    # an optional delimiting space.
    text = re.sub(r'\\[a-zA-Z]+-?\d*\s?', '', text)
    # Lingering escapes like `\\`, `\{`, `\}`.
    text = re.sub(r'\\[^a-zA-Z]', '', text)
    text = text.replace('{', '').replace('}', '')

    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    cleaned = []
    prev_blank = False
    for line in lines:
        if line:
            cleaned.append(line)
            prev_blank = False
        elif not prev_blank:
            cleaned.append('')
            prev_blank = True
    return '\n'.join(cleaned).strip()


def _classify_blob(plain_text):
    """Classify a stripped pre-2018 RTF blob as one of GROSS, MARKERS, FROZEN,
    ADDENDUM, FINAL, or OTHER. Used to pick the final-diagnosis blob from the
    multi-document SPECIMEN_NOTE."""
    if not plain_text:
        return "OTHER"

    text = plain_text

    if re.search(r'Received fresh', text):
        return "GROSS"

    has_organ_part = bool(re.search(
        r'(?:^|\n)A\.\s+(Breast|Lymph nodes?|Skin|Nipple)[^:\n]{0,80}:', text
    ))

    if not has_organ_part:
        if text.startswith('Source:') or (
            'Estrogen:' in text and 'Progesterone:' in text and 'HER2' in text
        ):
            return "MARKERS"

    if ('Frozen section histologic interpretation' in text
            or 'HOLDOVER' in text
            or 'HOLD OVER' in text):
        return "FROZEN"

    if len(text) < 200 or text.startswith('Stains for'):
        return "ADDENDUM"

    if has_organ_part:
        return "FINAL"

    return "OTHER"


def extract_final_diagnosis_pre_2018(specimen_note):
    if pd.isna(specimen_note):
        return None

    blobs = _split_rtf_blobs(specimen_note)
    if not blobs:
        return None

    candidates = []
    for blob in blobs:
        plain = _strip_rtf(blob)
        if _classify_blob(plain) == "FINAL":
            candidates.append(plain)

    if not candidates:
        return None

    with_synoptic = [c for c in candidates if re.search(r'Synoptic Report', c, re.IGNORECASE)]
    if with_synoptic:
        return max(with_synoptic, key=len)
    return max(candidates, key=len)


def extract_synoptic_report_pre_2018(specimen_note):
    final_text = extract_final_diagnosis_pre_2018(specimen_note)
    if not final_text:
        return None

    # Require the colon (and proper case) — lowercase "synoptic report" appears
    # as a body reference (e.g. "See synoptic report and comment.") and is not
    # the heading.
    match = re.search(r'Synoptic Report\s*:', final_text)
    if not match:
        return None
    return final_text[match.end():].strip()


def _join_unique(series):
    """Join unique non-null values with `; ` separator for fan-out aggregation."""
    vals = sorted({str(v).strip() for v in series.dropna() if str(v).strip()})
    return '; '.join(vals) if vals else None


def _collapse_join_fanout(pathology_df):
    """The BigQuery in query.py joins through DIM_PATHOLOGY_DIAGNOSIS_CODE_BRIDGE
    and FACT_PATHOLOGY_SPECIMEN_DETAIL, both of which can produce multiple rows
    per real pathology record. Group by SPECIMEN_ACCESSION_NUMBER and merge the
    fan-out columns to unique values so each pathology record is a single row
    before lesion-splitting."""
    fanout_cols = [
        c for c in ['DIAGNOSIS_NAME', 'SPECIMEN_PART_TYPE_NAME',
                    'PART_DESCRIPTION', 'SPECIMEN_PART_TYPE_CODE']
        if c in pathology_df.columns
    ]
    if not fanout_cols or 'SPECIMEN_ACCESSION_NUMBER' not in pathology_df.columns:
        return pathology_df

    has_accession = pathology_df['SPECIMEN_ACCESSION_NUMBER'].notna()
    keyed = pathology_df[has_accession]
    unkeyed = pathology_df[~has_accession]
    if keyed.empty:
        return pathology_df

    other_cols = [c for c in keyed.columns
                  if c not in fanout_cols and c != 'SPECIMEN_ACCESSION_NUMBER']
    agg_spec = {c: _join_unique for c in fanout_cols}
    for c in other_cols:
        agg_spec[c] = 'first'

    collapsed = keyed.groupby('SPECIMEN_ACCESSION_NUMBER', as_index=False).agg(agg_spec)
    return pd.concat([collapsed, unkeyed], ignore_index=True)


def filter_path_data(pathology_df, output_path):
    print("Parsing Pathology Data")

    rows_before = len(pathology_df)
    pathology_df = pathology_df.drop_duplicates(keep='first')
    rows_after = len(pathology_df)
    duplicates_removed_early = rows_before - rows_after
    print(f"Removed {duplicates_removed_early} exact duplicate rows before processing.")

    rows_before_collapse = len(pathology_df)
    pathology_df = _collapse_join_fanout(pathology_df)
    fanout_collapsed = rows_before_collapse - len(pathology_df)
    print(f"Collapsed {fanout_collapsed} fan-out rows from diagnosis/specimen-part joins.")
    append_audit("query_clean_path.path_fanout_collapsed", fanout_collapsed)

    # Pre-2018 records (SPECIMEN_ACCESSION_NUMBER prefixed with "amml") have a
    # different SPECIMEN_NOTE shape (multiple concatenated RTF blobs) and need
    # their own extraction path; downstream pipeline is shared.
    pre_mask = pathology_df['SPECIMEN_ACCESSION_NUMBER'].apply(_is_pre_2018)
    append_audit("query_clean_path.path_pre2018_count", int(pre_mask.sum()))
    append_audit("query_clean_path.path_post2018_count", int((~pre_mask).sum()))

    pathology_df['final_diag'] = None
    pathology_df['SYNOPTIC_REPORT'] = None

    pathology_df.loc[pre_mask, 'final_diag'] = (
        pathology_df.loc[pre_mask, 'SPECIMEN_NOTE'].apply(extract_final_diagnosis_pre_2018)
    )
    pathology_df.loc[~pre_mask, 'final_diag'] = (
        pathology_df.loc[~pre_mask, 'SPECIMEN_NOTE'].apply(extract_final_diagnosis)
    )
    pathology_df.loc[pre_mask, 'SYNOPTIC_REPORT'] = (
        pathology_df.loc[pre_mask, 'SPECIMEN_NOTE'].apply(extract_synoptic_report_pre_2018)
    )
    pathology_df.loc[~pre_mask, 'SYNOPTIC_REPORT'] = (
        pathology_df.loc[~pre_mask, 'SPECIMEN_NOTE'].apply(extract_synoptic_report)
    )
    
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
        'SPECIMEN_NOTE',
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
    pathology_df = pd.read_csv(f'{env}/data/raw_pathology.csv')
    filter_path_data(pathology_df)