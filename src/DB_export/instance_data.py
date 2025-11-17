from storage_adapter import *
from src.DB_processing.tools import append_audit

def merge_labelbox_labels(instance_data, instance_labels_csv_file):
    """
    Merge labelbox instance labels into instance_data based on dicom_hash.
    
    Args:
        instance_data: DataFrame with instance data
        instance_labels_csv_file: Path to InstanceLabels.csv
    
    Returns:
        instance_data with labelbox columns added
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
    
    # Fill NaN values with False for boolean columns
    bool_columns = ['Only Normal Tissue', 'Cyst Lesion Present', 
                    'Benign Lesion Present', 'Malignant Lesion Present']
    for col in bool_columns:
        if col in instance_data.columns:
            instance_data[col] = instance_data[col].fillna(False)
    
    print(f"Merged labelbox labels for {instance_data[available_columns[1:]].notna().any(axis=1).sum()} images")
    
    return instance_data

def apply_reject_system(image_df, instance_labels_csv_file, use_reject_system):
    # Merge labelbox data if available
    if file_exists(instance_labels_csv_file):
        labelbox_instance_data = read_csv(instance_labels_csv_file)
        
        instance_data = instance_data.merge(
            labelbox_instance_data, 
            on='dicom_hash', 
            how='left',
            suffixes=('', '_labelbox')
        )
        
        if 'image_name_labelbox' in instance_data.columns:
            instance_data.drop(columns=['image_name_labelbox'], inplace=True)
        
        instance_data = instance_data[instance_data['dicom_hash'].isin(image_df['dicom_hash'])]
        
        # Handle reject system
        if 'Reject Image' in instance_data.columns:
            if use_reject_system:
                before_count = len(image_df)
                rejected_images = instance_data[instance_data['Reject Image'] == True][['dicom_hash', 'image_name']]
                instance_data = instance_data[instance_data['Reject Image'] != True]
                image_df = image_df[~image_df['dicom_hash'].isin(rejected_images['dicom_hash'])]
                
                removed_count = before_count - len(image_df)
                append_audit("export.labeled_reject_removed", removed_count)
                instance_data.drop(columns=['Reject Image'], inplace=True)
            else:
                instance_data['Reject Image'] = instance_data['Reject Image'].fillna(False)
                
                
    return image_df