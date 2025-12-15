from tools.storage_adapter import *
from src.DB_processing.tools import append_audit

def merge_labelbox_labels(instance_data, instance_labels_csv_file):
    """
    Merge labelbox instance labels into instance_data based on dicom_hash.
    
    Args:
        instance_data: DataFrame with instance data
        instance_labels_csv_file: Path to InstanceLabels.csv
    
    Returns:
        instance_data with labelbox columns added (NaN for unmatched rows)
    """
    if not file_exists(instance_labels_csv_file):
        print("InstanceLabels.csv not found - skipping labelbox label merge")
        return instance_data
    
    # Read labelbox data
    labelbox_data = read_csv(instance_labels_csv_file)
    
    # Rename DicomHash to dicom_hash for consistency
    if 'DicomHash' in labelbox_data.columns:
        labelbox_data = labelbox_data.rename(columns={'DicomHash': 'dicom_hash'})
    
    # Select columns to merge (exclude Reject Image as it's handled separately)
    label_columns = [
        'dicom_hash',
        'Only Normal Tissue',
        'Cyst Lesion Present',
        'Benign Lesion Present',
        'Malignant Lesion Present'
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in label_columns if col in labelbox_data.columns]
    labelbox_data = labelbox_data[available_columns]
    
    # Merge with instance_data
    instance_data = instance_data.merge(
        labelbox_data,
        on='dicom_hash',
        how='left'
    )
    
    # Count how many images actually matched
    matched_count = instance_data[available_columns[1:]].notna().any(axis=1).sum()
    total_count = len(instance_data)
    print(f"Merged labelbox labels for {matched_count}/{total_count} images ({total_count - matched_count} unmatched)")
    
    return instance_data

def apply_reject_system(image_df, instance_labels_csv_file, use_reject_system):
    """
    Filter image_df based on 'Reject Image' column from labelbox data.
    
    Args:
        image_df: DataFrame with image data
        instance_labels_csv_file: Path to InstanceLabels.csv
        use_reject_system: Boolean - if True, remove rejected images
    
    Returns:
        Filtered image_df
    """
    # Only process if labelbox data exists
    if not file_exists(instance_labels_csv_file):
        return image_df
    
    # Read labelbox data
    labelbox_data = read_csv(instance_labels_csv_file)
    
    # Rename DicomHash to dicom_hash for consistency
    if 'DicomHash' in labelbox_data.columns:
        labelbox_data = labelbox_data.rename(columns={'DicomHash': 'dicom_hash'})
    
    # Check if Reject Image column exists
    if 'Reject Image' not in labelbox_data.columns:
        return image_df
    
    # Handle reject system
    if use_reject_system:
        # Get rejected images
        rejected_hashes = labelbox_data[labelbox_data['Reject Image'] == True]['dicom_hash']
        
        before_count = len(image_df)
        # Filter out rejected images
        image_df = image_df[~image_df['dicom_hash'].isin(rejected_hashes)]
        
        removed_count = before_count - len(image_df)
        if removed_count > 0:
            print(f"Removed {removed_count} rejected images from labelbox")
        append_audit("export.labeled_reject_removed", removed_count)
    
    return image_df