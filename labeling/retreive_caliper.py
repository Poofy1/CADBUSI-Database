import labelbox, requests, re, time
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID):
    print("(Newly created data in labelbox will take time to update!)")
    
    image_df_file = 'ImageData.csv'
    
    # Read the ImageData.csv file to get the ImageName to crop coordinates mapping
    try:
        yolo_df = pd.read_csv(f'{env}/{image_df_file}')
        print(f"Loaded {len(yolo_df)} rows from {image_df_file}")
        
        # Create a mapping from ImageName to crop coordinates and prediction
        image_to_crops = {}
        for _, row in yolo_df.iterrows():
            image_to_crops[row['ImageName']] = {
                'crop_x': row['crop_x'],
                'crop_y': row['crop_y'],
                'crop_w': row['crop_w'],
                'crop_h': row['crop_h'],
                'has_calipers_prediction': row['has_calipers_prediction']
            }
        print(f"Created mapping for {len(image_to_crops)} images")
        
    except FileNotFoundError:
        print(f"Error: Could not find {env}/{image_df_file}")
        return
    except KeyError as e:
        print(f"Error: Missing column in {image_df_file}: {e}")
        return
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)
    
    print("Contacting Labelbox")
    
    # Updated method: Use export() instead of export_v2()
    export_params = {
        "data_row_details": True,
        "label_details": True,
        "project_details": True
    }
    
    export_task = project.export(params=export_params)
    
    # Wait for the export task to complete
    export_task.wait_till_done()
    
    # Get the export data properly using the buffered stream
    print("Getting export data...")
    export_data = []
    
    # Use the correct method to get the data
    for data_row in export_task.get_buffered_stream():
        export_data.append(data_row.json)
    
    print(f"Retrieved {len(export_data)} data rows")
    print("Saving raw export data for debugging...")
    with open(f'{env}/labelbox_export_raw.json', 'w') as f:
        json.dump(export_data, f, indent=2)
        
    # Parse the data into a CSV format
    csv_data = parse_labelbox_annotations(export_data, image_to_crops)
    
    # Save to CSV
    csv_data.to_csv(f'{env}/train_caliper.csv', index=False)
    print(f"Annotations saved to {env}/train_caliper.csv")

def parse_labelbox_annotations(export_data, image_to_crops):
    """
    Parse Labelbox export data and extract classification for calipers
    Only includes items in IN_REVIEW status with valid classifications
    """
    print("Parsing annotations...")
    
    results = []
    missing_crops = []
    skipped_not_in_review = 0
    no_classification_count = 0
    
    for item in tqdm(export_data):
        # Get the image name from external_id
        image_name = item['data_row']['external_id']
        
        # Look up the crop coordinates for this image
        crop_data = image_to_crops.get(image_name)
        
        if crop_data is None:
            missing_crops.append(image_name)
            continue
        
        classification = None
        workflow_status = None
        
        # Go through each project (should be just one in most cases)
        for project_id, project_data in item['projects'].items():
            # Get workflow status if available
            if 'project_details' in project_data:
                workflow_status = project_data['project_details'].get('workflow_status')
            
            # Skip if not IN_REVIEW
            if workflow_status != 'IN_REVIEW':
                skipped_not_in_review += 1
                continue
            
            # Go through each label
            for label in project_data.get('labels', []):
                # Check if there are classifications
                if 'classifications' in label.get('annotations', {}):
                    classifications_list = label['annotations']['classifications']
                    
                    for classification_obj in classifications_list:
                        # Check if this is the has_calipers classification
                        if classification_obj.get('name') == 'has_calipers':
                            # The actual classification is in checklist_answers
                            if 'checklist_answers' in classification_obj:
                                checklist_answers = classification_obj['checklist_answers']
                                if checklist_answers and len(checklist_answers) > 0:
                                    # Get the name from the first checklist answer
                                    classification = checklist_answers[0].get('name')
                            
                            if classification:
                                break
                    
                    if classification:
                        break
                
                if classification:
                    break
        
        # Only add if workflow status was IN_REVIEW and has a valid classification
        if workflow_status == 'IN_REVIEW':
            # Convert classification to binary: 1 for has_calipers, 0 for has_no_calipers
            has_calipers_value = None
            if classification == 'has_calipers':
                has_calipers_value = 1
            elif classification == 'has_no_calipers':
                has_calipers_value = 0
            
            # Only add rows with valid classification
            if has_calipers_value is not None:
                results.append({
                    'ImageName': image_name,
                    'crop_x': crop_data['crop_x'],
                    'crop_y': crop_data['crop_y'],
                    'crop_w': crop_data['crop_w'],
                    'crop_h': crop_data['crop_h'],
                    'has_calipers_prediction': crop_data['has_calipers_prediction'],
                    'has_calipers': has_calipers_value,
                    'workflow_status': workflow_status
                })
            else:
                no_classification_count += 1
    
    # Report statistics
    if missing_crops:
        print(f"Warning: Could not find crop data for {len(missing_crops)} images:")
        for img in missing_crops[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_crops) > 10:
            print(f"  ... and {len(missing_crops) - 10} more")
    
    print(f"\nSkipped {skipped_not_in_review} images not in IN_REVIEW status")
    print(f"Images in IN_REVIEW without valid classifications: {no_classification_count}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nProcessed {len(df)} images with valid classifications")
    
    if len(df) > 0:
        print("\nClassification distribution:")
        print(df['has_calipers'].value_counts())
        
        # Create validation set
        # Initialize all as training (val=0)
        df['val'] = 0
        
        # Get positive and negative samples
        positive_indices = df[df['has_calipers'] == 1].index
        negative_indices = df[df['has_calipers'] == 0].index
        
        # Calculate 20% of positive samples
        n_positive_val = int(len(positive_indices) * 0.2)
        
        # Randomly sample 20% of positive samples
        np.random.seed(42)  # For reproducibility
        val_positive_indices = np.random.choice(positive_indices, size=n_positive_val, replace=False)
        
        # Sample the same number of negative samples
        if len(negative_indices) >= n_positive_val:
            val_negative_indices = np.random.choice(negative_indices, size=n_positive_val, replace=False)
        else:
            print(f"Warning: Not enough negative samples. Using all {len(negative_indices)} negative samples for validation.")
            val_negative_indices = negative_indices
        
        # Mark validation samples
        val_indices = np.concatenate([val_positive_indices, val_negative_indices])
        df.loc[val_indices, 'val'] = 1
        
        print(f"\nValidation set split:")
        print(f"  Total samples: {len(df)}")
        print(f"  Training samples (val=0): {len(df[df['val'] == 0])}")
        print(f"  Validation samples (val=1): {len(df[df['val'] == 1])}")
        print(f"\nValidation set distribution:")
        print(f"  Positive (has_calipers=1): {len(df[(df['val'] == 1) & (df['has_calipers'] == 1)])}")
        print(f"  Negative (has_calipers=0): {len(df[(df['val'] == 1) & (df['has_calipers'] == 0)])}")
        print(f"\nTraining set distribution:")
        print(f"  Positive (has_calipers=1): {len(df[(df['val'] == 0) & (df['has_calipers'] == 1)])}")
        print(f"  Negative (has_calipers=0): {len(df[(df['val'] == 0) & (df['has_calipers'] == 0)])}")
    
    return df

# Run the function
Read_Labelbox_Data(CONFIG['LABELBOX_API_KEY'], "cmg9zh5yk1n0c07190jkq3423")