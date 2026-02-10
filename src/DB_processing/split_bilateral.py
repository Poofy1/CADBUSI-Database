from tqdm import tqdm
import pandas as pd
from src.DB_processing.tools import append_audit

def determine_has_malignant(row, laterality):
    """
    Determine if a breast has malignancy based on diagnosis columns and laterality.

    Args:
        row: DataFrame row with left_diagnosis and right_diagnosis columns
        laterality: 'LEFT' or 'RIGHT'

    Returns:
        1 if malignant, 0 if not malignant, -1 if both diagnoses are NULL
    """
    # Check if both diagnoses are NULL
    left_diag = row.get('left_diagnosis', None)
    right_diag = row.get('right_diagnosis', None)

    if pd.isna(left_diag) and pd.isna(right_diag):
        return -1

    if laterality == 'LEFT':
        diagnosis = left_diag
    elif laterality == 'RIGHT':
        diagnosis = right_diag
    else:
        return 0

    # Check if diagnosis contains MALIGNANT
    if pd.notna(diagnosis) and 'MALIGNANT' in str(diagnosis).upper():
        return 1
    return 0


def split_bilateral_cases(breast_df, image_df):
    """
    Split bilateral cases into separate LEFT and RIGHT breast rows based on available images.
    Only splits if images exist for both lateralities. Otherwise converts to single-sided.
    Keeps bilateral cases with unknown laterality and flags them with has_unknown_laterality=1.

    Returns:
        Updated breast_df with bilateral cases split or converted
    """
    bilateral_df = breast_df[breast_df['study_laterality'] == 'BILATERAL'].copy()
    non_bilateral_df = breast_df[breast_df['study_laterality'] != 'BILATERAL'].copy()

    # Add was_bilateral and has_unknown_laterality columns to non-bilateral cases (set to 0)
    non_bilateral_df['was_bilateral'] = 0
    non_bilateral_df['has_unknown_laterality'] = 0

    if bilateral_df.empty:
        return non_bilateral_df

    # Pre-group images by accession_number to avoid repeated filtering (HUGE speedup!)
    print(f"Pre-grouping {len(image_df)} images by accession number...")
    image_groups = image_df.groupby('accession_number')['laterality'].apply(list).to_dict()

    split_rows = []
    converted_rows = []
    removed_count = 0
    flagged_unknown_laterality = 0

    print(f"Processing {len(bilateral_df)} bilateral cases...")
    for _, row in tqdm(bilateral_df.iterrows(), total=len(bilateral_df), desc="Splitting bilateral cases"):
        accession = row['accession_number']

        # Get lateralities for this accession (much faster than filtering!)
        lateralities = image_groups.get(accession, [])

        if not lateralities:
            # No images for this accession
            removed_count += 1
            continue

        # Check for unknown/null laterality images
        has_unknown = any(
            pd.isna(lat) or lat == '' or str(lat).upper() == 'UNKNOWN'
            for lat in lateralities
        )

        if has_unknown:
            flagged_unknown_laterality += 1

        has_left = 'LEFT' in lateralities
        has_right = 'RIGHT' in lateralities

        if has_left and has_right:
            # Split into two rows
            left_row = row.copy()
            left_row['study_laterality'] = 'LEFT'
            left_row['has_malignant'] = determine_has_malignant(row, 'LEFT')
            left_row['was_bilateral'] = 1
            left_row['has_unknown_laterality'] = 1 if has_unknown else 0

            right_row = row.copy()
            right_row['study_laterality'] = 'RIGHT'
            right_row['has_malignant'] = determine_has_malignant(row, 'RIGHT')
            right_row['was_bilateral'] = 1
            right_row['has_unknown_laterality'] = 1 if has_unknown else 0

            split_rows.extend([left_row, right_row])
        elif has_left:
            # Convert to LEFT only
            converted_row = row.copy()
            converted_row['study_laterality'] = 'LEFT'
            converted_row['has_malignant'] = determine_has_malignant(row, 'LEFT')
            converted_row['was_bilateral'] = 1
            converted_row['has_unknown_laterality'] = 1 if has_unknown else 0
            converted_rows.append(converted_row)
        elif has_right:
            # Convert to RIGHT only
            converted_row = row.copy()
            converted_row['study_laterality'] = 'RIGHT'
            converted_row['has_malignant'] = determine_has_malignant(row, 'RIGHT')
            converted_row['was_bilateral'] = 1
            converted_row['has_unknown_laterality'] = 1 if has_unknown else 0
            converted_rows.append(converted_row)
        else:
            # No images for either side, remove this case
            removed_count += 1

    # Combine all dataframes
    result_dfs = [non_bilateral_df]
    if split_rows:
        result_dfs.append(pd.DataFrame(split_rows))
    if converted_rows:
        result_dfs.append(pd.DataFrame(converted_rows))

    result_df = pd.concat(result_dfs, ignore_index=True)

    print(f"Bilateral processing: {len(split_rows)//2} split into L+R, {len(converted_rows)} converted to single-sided, {removed_count} removed (no images), {flagged_unknown_laterality} flagged (unknown laterality)")
    append_audit("export.bilateral_split", len(split_rows)//2)
    append_audit("export.bilateral_converted", len(converted_rows))
    append_audit("export.bilateral_removed_no_images", removed_count)
    append_audit("export.bilateral_flagged_unknown_laterality", flagged_unknown_laterality)

    return result_df
