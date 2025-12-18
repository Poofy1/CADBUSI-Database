from tqdm import tqdm
import pandas as pd
import os
from src.DB_processing.tools import append_audit
from tools.storage_adapter import save_data

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


def apply_filters(image_df, video_df, breast_df, CONFIG, output_dir=None):
    """
    Apply all quality and relevance filters to the image and video data.
    Tracks excluded images and saves them to a CSV file.

    Args:
        image_df: DataFrame of image data
        video_df: DataFrame of video data
        breast_df: DataFrame of breast/study data
        CONFIG: Configuration dictionary
        output_dir: Directory to save excluded_images.csv (optional)

    Returns:
        tuple: (filtered_image_df, filtered_video_df, filtered_breast_df)
    """
    audit_stats = {}
    excluded_images = []  # Track all excluded images

    # Track initial counts
    audit_stats['init_images'] = len(image_df)
    audit_stats['init_videos'] = len(video_df)
    audit_stats['init_breasts'] = len(breast_df)

    # Merge study_laterality from breast_df for laterality filtering logic
    if 'study_laterality' not in image_df.columns:
        image_df = image_df.merge(
            breast_df[['accession_number', 'study_laterality']],
            on='accession_number',
            how='left'
        )

    # Filter 1: Remove images that are too dark
    darkness_thresh = 75
    darkness_values = image_df['darkness'].round(2).tolist()
    append_audit("export.darkness_values", darkness_values)
    append_audit("export.darkness_thresh", darkness_thresh)

    dark_mask = image_df['darkness'] > darkness_thresh
    dark_images = image_df[dark_mask][['image_name', 'patient_id', 'accession_number']].copy()
    dark_images['exclusion_reason'] = 'too dark (>75)'
    excluded_images.append(dark_images)

    image_df = image_df[~dark_mask]
    audit_stats['too_dark_removed'] = len(dark_images)

    # Filter 2: Remove non-breast images
    # First, fix unknown areas
    image_df.loc[(image_df['area'] == 'unknown') | (image_df['area'].isna()), 'area'] = 'breast'

    non_breast_mask = image_df['area'] != 'breast'
    non_breast_images = image_df[non_breast_mask][['image_name', 'patient_id', 'accession_number']].copy()
    non_breast_images['exclusion_reason'] = 'non-breast area'
    excluded_images.append(non_breast_images)

    image_df = image_df[~non_breast_mask]
    audit_stats['non_breast_removed'] = len(non_breast_images)

    # Filter 3: Remove images with unknown laterality in BILATERAL studies
    bilateral_unknown_mask = (
        ((image_df['laterality'] == 'unknown') | (image_df['laterality'].isna())) &
        (image_df['study_laterality'].str.upper() == 'BILATERAL')
    )
    lat_images = image_df[bilateral_unknown_mask][['image_name', 'patient_id', 'accession_number']].copy()
    lat_images['exclusion_reason'] = 'unknown laterality (bilateral study)'
    excluded_images.append(lat_images)

    image_df = image_df[~bilateral_unknown_mask]
    audit_stats['unknown_lat_removed'] = len(lat_images)

    # Filter 4: Remove images with multiple regions
    multi_region_mask = image_df['region_count'] > 1
    region_images = image_df[multi_region_mask][['image_name', 'patient_id', 'accession_number']].copy()
    region_images['exclusion_reason'] = 'multiple regions'
    excluded_images.append(region_images)

    image_df = image_df[~multi_region_mask]
    audit_stats['multi_region_removed'] = len(region_images)

    # Split or convert bilateral cases based on available images
    breast_df = split_bilateral_cases(breast_df, image_df)

    # Update has_malignant for all cases (vectorized)
    both_null = breast_df['left_diagnosis'].isna() & breast_df['right_diagnosis'].isna()
    left_mask = breast_df['study_laterality'] == 'LEFT'
    left_malignant = breast_df['left_diagnosis'].fillna('').str.upper().str.contains('MALIGNANT')
    right_mask = breast_df['study_laterality'] == 'RIGHT'
    right_malignant = breast_df['right_diagnosis'].fillna('').str.upper().str.contains('MALIGNANT')

    breast_df.loc[both_null, 'has_malignant'] = -1
    breast_df.loc[left_mask & ~both_null, 'has_malignant'] = left_malignant[left_mask & ~both_null].astype(int)
    breast_df.loc[right_mask & ~both_null, 'has_malignant'] = right_malignant[right_mask & ~both_null].astype(int)

    # Keep only images whose patient_id exists in breast_df
    valid_patient_ids = breast_df['patient_id'].unique()
    image_df = image_df[image_df['patient_id'].isin(valid_patient_ids)]
    video_df = video_df[video_df['patient_id'].isin(valid_patient_ids)]

    # Filter 5: Remove bad aspect ratios
    min_aspect_ratio = CONFIG.get('MIN_ASPECT_RATIO', 0.5)
    max_aspect_ratio = CONFIG.get('MAX_ASPECT_RATIO', 4.0)

    bad_aspect_mask = (
        (image_df['crop_aspect_ratio'] < min_aspect_ratio) |
        (image_df['crop_aspect_ratio'] > max_aspect_ratio)
    )
    aspect_images = image_df[bad_aspect_mask][['image_name', 'patient_id', 'accession_number']].copy()
    aspect_images['exclusion_reason'] = f'bad aspect ratio ({min_aspect_ratio}-{max_aspect_ratio})'
    excluded_images.append(aspect_images)

    image_df = image_df[~bad_aspect_mask]
    audit_stats['bad_aspect_removed'] = len(aspect_images)

    # Filter 6: Remove images that are too small
    min_dimension = CONFIG.get('MIN_DIMENSION', 200)

    too_small_mask = (
        (image_df['crop_w'] < min_dimension) |
        (image_df['crop_h'] < min_dimension)
    )
    small_images = image_df[too_small_mask][['image_name', 'patient_id', 'accession_number']].copy()
    small_images['exclusion_reason'] = f'too small (<{min_dimension}px)'
    excluded_images.append(small_images)

    image_df = image_df[~too_small_mask]
    audit_stats['too_small_removed'] = len(small_images)

    # Track final usable images
    audit_stats['usable_images'] = len(image_df)

    # Save excluded images to CSV
    if output_dir and excluded_images:
        excluded_df = pd.concat(excluded_images, ignore_index=True)
        excluded_path = os.path.join(output_dir, 'ExclusionData.csv')
        save_data(excluded_df, excluded_path)
        print(f"Saved {len(excluded_df)} excluded images to {excluded_path}")

    # Log audit statistics
    for key, value in audit_stats.items():
        append_audit(f"export.{key}", value)

    return image_df, video_df, breast_df