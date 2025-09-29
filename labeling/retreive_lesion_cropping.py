import labelbox, requests, re, time
import pandas as pd
from tqdm import tqdm
import os
import json
env = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID):
    print("(Newly created data in labelbox will take time to update!)")
    
    # Read the yolo_accessions.csv file to get the ImageName to DicomHash mapping
    print("Loading yolo_accessions.csv...")
    try:
        yolo_df = pd.read_csv(f'{env}/yolo_accessions.csv')
        print(f"Loaded {len(yolo_df)} rows from yolo_accessions.csv")
        
        # Create a mapping from ImageName to DicomHash
        image_to_hash = dict(zip(yolo_df['ImageName'], yolo_df['DicomHash']))
        print(f"Created mapping for {len(image_to_hash)} images")
        
    except FileNotFoundError:
        print(f"Error: Could not find {env}/yolo_accessions.csv")
        return
    except KeyError as e:
        print(f"Error: Missing column in yolo_accessions.csv: {e}")
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
    
    # Parse the data into a CSV format
    csv_data = parse_labelbox_annotations(export_data, image_to_hash)
    
    # Save to CSV
    csv_data.to_csv(f'{env}/labelbox_annotations.csv', index=False)
    print(f"Annotations saved to {env}/labelbox_annotations.csv")


def parse_labelbox_annotations(export_data, image_to_hash):
    """
    Parse Labelbox export data and extract bounding box coordinates for lesions and axillary nodes
    """
    print("Parsing annotations...")
    
    results = []
    missing_hashes = []
    
    for item in tqdm(export_data):
        # Get the image name from external_id
        image_name = item['data_row']['external_id']
        
        # Look up the DicomHash for this image
        dicom_hash = image_to_hash.get(image_name)
        
        if dicom_hash is None:
            missing_hashes.append(image_name)
            # Skip this image if we can't find the DicomHash
            continue
        
        lesion_coords = []
        axillary_node_coords = []
        
        # Go through each project (should be just one in most cases)
        for project_id, project_data in item['projects'].items():
            # Go through each label
            for label in project_data['labels']:
                # Go through each annotation object
                for obj in label['annotations']['objects']:
                    class_name = obj['name']
                    bbox = obj['bounding_box']
                    
                    # Format as [left,top,width,height]
                    coord_string = f"[{int(bbox['left'])},{int(bbox['top'])},{int(bbox['width'])},{int(bbox['height'])}]"
                    
                    if class_name == 'lesion':
                        lesion_coords.append(coord_string)
                    elif class_name == 'axillary node':
                        axillary_node_coords.append(coord_string)
        
        # Join multiple coordinates with semicolon, but each coordinate is in brackets
        lesion_coords_str = ';'.join(lesion_coords) if lesion_coords else ''
        axillary_node_coords_str = ';'.join(axillary_node_coords) if axillary_node_coords else ''
        
        results.append({
            'dicom_hash': dicom_hash,  # Use DicomHash instead of image_name
            'lesion_coords': lesion_coords_str,
            'axillary_node_coords': axillary_node_coords_str
        })
    
    # Report any missing hashes
    if missing_hashes:
        print(f"Warning: Could not find DicomHash for {len(missing_hashes)} images:")
        for img in missing_hashes[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_hashes) > 10:
            print(f"  ... and {len(missing_hashes) - 10} more")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"Processed {len(df)} images")
    print(f"Images with lesions: {len(df[df['lesion_coords'] != ''])}")
    print(f"Images with axillary nodes: {len(df[df['axillary_node_coords'] != ''])}")
    
    return df


# Run the function
Read_Labelbox_Data(CONFIG['LABELBOX_API_KEY'], CONFIG['PROJECT_ID'])