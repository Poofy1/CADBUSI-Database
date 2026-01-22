import labelbox as lb
import pandas as pd
import sys
import os
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

def export_labelbox_annotations(project_id, project_name, client):
    """Export annotations from a Labelbox project"""
    print(f"\n{'='*50}")
    print(f"Exporting from: {project_name}")
    print(f"{'='*50}")
    
    project = client.get_project(project_id)
    
    export_params = {
        "data_row_details": True,
        "label_details": True,
        "project_details": True
    }
    
    print("Starting export...")
    export_task = project.export(params=export_params)
    export_task.wait_till_done()
    
    print("Retrieving export data...")
    export_data = []
    for data_row in export_task.get_buffered_stream():
        export_data.append(data_row.json)
    
    print(f"Retrieved {len(export_data)} data rows")
    
    return export_data

def get_with_retry(url, headers, max_retries=5):
    """Download with retry logic for unreliable servers"""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response
            else:
                print(f"\nFailed to download mask, retrying {retries}/{max_retries} (status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        
        # Sleep before retrying
        time.sleep(2 ** retries)  # Exponential backoff
        retries += 1
    
    print(f"Failed to download mask after {max_retries} retries")
    return None

def parse_mask_annotations(export_data, output_mask_dir, api_key):
    """Parse segmentation mask annotations from Labelbox export"""
    print(f"\nParsing mask annotations...")
    
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Headers for authenticated requests
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    results = []
    no_labels_count = 0
    duplicate_count = 0
    
    for item in tqdm(export_data):
        # Get the image name from external_id
        image_name = item['data_row']['external_id']
        
        # Track if we found any labels
        found_label = False
        
        # Track unique masks per image to avoid duplicates
        seen_masks = set()
        
        # Go through each project
        for project_id, project_data in item['projects'].items():
            # Go through each label
            for label in project_data.get('labels', []):
                # Check if there are mask objects
                if 'objects' in label.get('annotations', {}):
                    objects_list = label['annotations']['objects']
                    
                    for obj in objects_list:
                        # Check if this is a lesion_seg mask with mask data
                        if obj.get('name') == 'lesion_seg' and 'mask' in obj:
                            mask_data = obj['mask']
                            
                            # Create unique mask filename
                            # Strip extension from image_name if present
                            base_name = os.path.splitext(image_name)[0]
                            mask_filename = f"{base_name}_mask.png"
                            
                            # Skip if we've already processed this mask
                            if mask_filename in seen_masks:
                                duplicate_count += 1
                                continue
                            
                            seen_masks.add(mask_filename)
                            
                            try:
                                # Get the mask image
                                if 'png' in mask_data:
                                    # Direct base64 PNG
                                    mask_bytes = base64.b64decode(mask_data['png'])
                                    mask_img = Image.open(BytesIO(mask_bytes))
                                elif 'url' in mask_data:
                                    # Download from URL with auth and retry
                                    response = get_with_retry(mask_data['url'], headers)
                                    if response is None:
                                        print(f"  Failed to download mask for {image_name}")
                                        continue
                                    mask_img = Image.open(BytesIO(response.content))
                                else:
                                    print(f"  Warning: No PNG or URL found for {image_name}")
                                    continue
                                
                                # Save to output directory
                                mask_path = os.path.join(output_mask_dir, mask_filename)
                                mask_img.save(mask_path)
                                
                                results.append({
                                    'caliper_image': image_name,
                                    'mask_file': mask_filename
                                })
                                
                                found_label = True
                            except Exception as e:
                                print(f"  Error saving mask for {image_name}: {e}")
        
        if not found_label:
            no_labels_count += 1
    
    print(f"Saved {len(results)} mask files to {output_mask_dir}")
    print(f"Skipped {duplicate_count} duplicate annotations")
    print(f"Images without labels: {no_labels_count}")
    
    return pd.DataFrame(results)

def main():
    client = lb.Client(api_key=CONFIG['LABELBOX_API_KEY'])
    
    # Segmentation project ID from your upload script
    project_id = "cmkn2bljg1ehp07v909jr9zbx"
    
    # Output directory for masks
    output_mask_dir = "C:/Users/Tristan/Desktop/labeled_masks/"
    
    # Export from project
    export_data = export_labelbox_annotations(
        project_id, 
        "Lesion Segmentation", 
        client
    )
    
    # Parse mask annotations and save masks
    df_labeled = parse_mask_annotations(export_data, output_mask_dir, CONFIG['LABELBOX_API_KEY'])
    
    print(f"\n{'='*50}")
    print("EXPORT RESULTS")
    print(f"{'='*50}")
    print(f"Total masks exported: {len(df_labeled)}")
    
    if len(df_labeled) == 0:
        print("\n⚠ WARNING: No mask annotations found!")
        return
    
    # Load original CSV to merge
    print(f"\nLoading original CSV...")
    df_original = pd.read_csv("C:/Users/Tristan/Desktop/seg_data2.csv")
    print(f"Original CSV has {len(df_original)} rows")
    
    # Check for duplicates in original CSV
    orig_dupes = df_original['caliper_image'].duplicated().sum()
    if orig_dupes > 0:
        print(f"⚠ Warning: Original CSV has {orig_dupes} duplicate caliper_image entries")
        print("Removing duplicates from original CSV before merge...")
        df_original = df_original.drop_duplicates(subset=['caliper_image'], keep='first')
        print(f"Original CSV now has {len(df_original)} unique rows")
    
    # Drop mask_file column from original if it exists
    if 'mask_file' in df_original.columns:
        df_original = df_original.drop(columns=['mask_file'])
    
    # Merge to get other columns from original
    df_merged = df_labeled.merge(
        df_original,
        on='caliper_image',
        how='left'
    )
    
    # Final deduplication - keep first occurrence
    rows_before_final = len(df_merged)
    df_merged = df_merged.drop_duplicates(keep='first')
    rows_after_final = len(df_merged)
    
    if rows_before_final != rows_after_final:
        print(f"\nRemoved {rows_before_final - rows_after_final} duplicate rows after merge")
    
    # Check how many matched
    matched = df_merged['caliper_image'].isin(df_original['caliper_image']).sum()
    print(f"Matched {matched}/{len(df_merged)} labeled images to original CSV")
    
    print(f"\nFinal data: {len(df_merged)} rows")
    print(f"Columns: {list(df_merged.columns)}")
    
    # Save to new CSV
    output_file = "C:/Users/Tristan/Desktop/seg_data_labeled.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\nSaved labeled data to: {output_file}")
    
    print(f"\n✓ Complete!")
    print(f"Masks saved to: {output_mask_dir}")
    print(f"CSV saved to: {output_file}")

if __name__ == "__main__":
    main()