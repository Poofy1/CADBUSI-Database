import labelbox as lb
import pandas as pd
import sys
import os
from tqdm import tqdm

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

def parse_bbox_annotations(export_data, project_name):
    """Parse bounding box annotations from Labelbox export"""
    print(f"\nParsing annotations from {project_name}...")
    
    results = []
    no_labels_count = 0
    duplicate_count = 0
    
    for item in tqdm(export_data):
        # Get the image name from external_id
        image_name = item['data_row']['external_id']
        
        # Track if we found any labels
        found_label = False
        
        # Track unique bboxes per image to avoid duplicates
        seen_bboxes = set()
        
        # Go through each project
        for project_id, project_data in item['projects'].items():
            # Go through each label
            for label in project_data.get('labels', []):
                # Check if there are bounding box objects
                if 'objects' in label.get('annotations', {}):
                    objects_list = label['annotations']['objects']
                    
                    for obj in objects_list:
                        # Check if this is a lesion_crop bbox with bounding_box field
                        if obj.get('name') == 'lesion_crop' and 'bounding_box' in obj:
                            bbox = obj['bounding_box']
                            
                            # Create a unique key for this bbox
                            bbox_key = (bbox['left'], bbox['top'], bbox['width'], bbox['height'])
                            
                            # Skip if we've already seen this exact bbox for this image
                            if bbox_key in seen_bboxes:
                                duplicate_count += 1
                                continue
                            
                            seen_bboxes.add(bbox_key)
                            
                            # Convert bbox format to x1, y1, x2, y2
                            x1 = int(bbox['left'])
                            y1 = int(bbox['top'])
                            x2 = x1 + int(bbox['width'])
                            y2 = y1 + int(bbox['height'])
                            
                            results.append({
                                'caliper_image': image_name,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'project': project_name
                            })
                            
                            found_label = True
        
        if not found_label:
            no_labels_count += 1
    
    print(f"Found {len(results)} bounding box annotations")
    print(f"Skipped {duplicate_count} duplicate annotations")
    print(f"Images without labels: {no_labels_count}")
    
    return pd.DataFrame(results)

def main():
    client = lb.Client(api_key=CONFIG['LABELBOX_API_KEY'])
    
    # Project IDs from your first script
    project_accept_id = "cmkg6htrn0af207z35frjaaz5"
    project_review_id = "cmkgil9ei1mb207x037954tov"
    
    # Export from both projects
    export_accept = export_labelbox_annotations(
        project_accept_id, 
        "Accept", 
        client
    )
    
    export_review = export_labelbox_annotations(
        project_review_id,
        "Needs Review",
        client
    )
    
    # Parse annotations from both projects
    df_accept = parse_bbox_annotations(export_accept, "Accept")
    df_review = parse_bbox_annotations(export_review, "Needs Review")
    
    # Check if dataframes are empty
    print(f"\nAccept project annotations: {len(df_accept)}")
    print(f"Needs Review project annotations: {len(df_review)}")
    
    # Combine both dataframes
    dfs_to_combine = []
    if len(df_accept) > 0:
        dfs_to_combine.append(df_accept)
    if len(df_review) > 0:
        dfs_to_combine.append(df_review)
    
    if len(dfs_to_combine) == 0:
        print("\n⚠ WARNING: No annotations found in either project!")
        return
    
    df_labeled = pd.concat(dfs_to_combine, ignore_index=True)
    
    print(f"\n{'='*50}")
    print("COMBINED RESULTS")
    print(f"{'='*50}")
    print(f"Total annotations: {len(df_labeled)}")
    print(f"\nBy project:")
    print(df_labeled['project'].value_counts())
    
    # Load original CSV to get additional columns (like status, etc)
    print(f"\nLoading original CSV...")
    df_original = pd.read_csv("C:/Users/Tristan/Desktop/crop_data.csv")
    print(f"Original CSV has {len(df_original)} rows")
    
    # Check for duplicates in original CSV
    orig_dupes = df_original['caliper_image'].duplicated().sum()
    if orig_dupes > 0:
        print(f"⚠ Warning: Original CSV has {orig_dupes} duplicate caliper_image entries")
        print("Removing duplicates from original CSV before merge...")
        df_original = df_original.drop_duplicates(subset=['caliper_image'], keep='first')
        print(f"Original CSV now has {len(df_original)} unique rows")
    
    # Drop coordinate columns from original to avoid duplicates
    cols_to_drop = ['x1', 'y1', 'x2', 'y2']
    df_original = df_original.drop(columns=[col for col in cols_to_drop if col in df_original.columns])
    
    # Merge to get other columns from original (like status, etc)
    df_merged = df_labeled.merge(
        df_original,
        on='caliper_image',
        how='left'
    )
    
    # Final deduplication after merge - keep first occurrence
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
    output_file = "C:/Users/Tristan/Desktop/crop_data_labeled.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\nSaved labeled data to: {output_file}")
    
    print("\n✓ Complete!")

if __name__ == "__main__":
    main()