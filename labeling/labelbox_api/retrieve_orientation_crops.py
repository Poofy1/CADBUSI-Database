import labelbox as lb
import pandas as pd
import sys
import os
import shutil
import cv2
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

def parse_orientation_annotations(export_data, project_name):
    """Parse orientation crop annotations from Labelbox export"""
    print(f"\nParsing annotations from {project_name}...")

    results = []
    no_labels_count = 0

    for item in tqdm(export_data):
        # Get the image name from external_id
        image_name = item['data_row']['external_id']

        # Initialize crop regions
        circle_ori_x1 = None
        circle_ori_y1 = None
        circle_ori_x2 = None
        circle_ori_y2 = None
        arrow_ori_x1 = None
        arrow_ori_y1 = None
        arrow_ori_x2 = None
        arrow_ori_y2 = None
        chest_ori_x1 = None
        chest_ori_y1 = None
        chest_ori_x2 = None
        chest_ori_y2 = None
        empty = False

        # Track if we found any labels
        found_label = False

        # Go through each project
        for project_id, project_data in item['projects'].items():
            # Go through each label
            for label in project_data.get('labels', []):
                # Check if there are bounding box objects
                if 'objects' in label.get('annotations', {}):
                    objects_list = label['annotations']['objects']

                    for obj in objects_list:
                        obj_name = obj.get('name', '')

                        # Check for circle_ori
                        if obj_name == 'circle_ori' and 'bounding_box' in obj:
                            bbox = obj['bounding_box']
                            circle_ori_x1 = int(bbox['left'])
                            circle_ori_y1 = int(bbox['top'])
                            circle_ori_x2 = circle_ori_x1 + int(bbox['width'])
                            circle_ori_y2 = circle_ori_y1 + int(bbox['height'])
                            found_label = True

                        # Check for arrow_ori
                        elif obj_name == 'arrow_ori' and 'bounding_box' in obj:
                            bbox = obj['bounding_box']
                            arrow_ori_x1 = int(bbox['left'])
                            arrow_ori_y1 = int(bbox['top'])
                            arrow_ori_x2 = arrow_ori_x1 + int(bbox['width'])
                            arrow_ori_y2 = arrow_ori_y1 + int(bbox['height'])
                            found_label = True

                        # Check for chest_ori
                        elif obj_name == 'chest_ori' and 'bounding_box' in obj:
                            bbox = obj['bounding_box']
                            chest_ori_x1 = int(bbox['left'])
                            chest_ori_y1 = int(bbox['top'])
                            chest_ori_x2 = chest_ori_x1 + int(bbox['width'])
                            chest_ori_y2 = chest_ori_y1 + int(bbox['height'])
                            found_label = True

                # Check for 'empty' classification
                if 'classifications' in label.get('annotations', {}):
                    classifications_list = label['annotations']['classifications']

                    for classification in classifications_list:
                        if classification.get('name') == 'empty':
                            # Check if it's a radio/checklist with answer
                            if 'radio_answer' in classification:
                                empty = True
                                found_label = True
                            elif 'checklist_answers' in classification:
                                # If checklist has any answers, mark as empty
                                if len(classification['checklist_answers']) > 0:
                                    empty = True
                                    found_label = True

        # Add result if we found any label
        if found_label:
            results.append({
                'image_name': image_name,
                'circle_ori_x1': circle_ori_x1,
                'circle_ori_y1': circle_ori_y1,
                'circle_ori_x2': circle_ori_x2,
                'circle_ori_y2': circle_ori_y2,
                'arrow_ori_x1': arrow_ori_x1,
                'arrow_ori_y1': arrow_ori_y1,
                'arrow_ori_x2': arrow_ori_x2,
                'arrow_ori_y2': arrow_ori_y2,
                'chest_ori_x1': chest_ori_x1,
                'chest_ori_y1': chest_ori_y1,
                'chest_ori_x2': chest_ori_x2,
                'chest_ori_y2': chest_ori_y2,
                'empty': empty,
                'project': project_name
            })
        else:
            no_labels_count += 1

    print(f"Found {len(results)} labeled images")
    print(f"Images without labels: {no_labels_count}")

    return pd.DataFrame(results)

def create_yolo_dataset(df_labeled, images_dir, output_dir):
    """Create YOLO dataset from labeled data"""
    print(f"\n{'='*50}")
    print("CREATING YOLO DATASET")
    print(f"{'='*50}")

    # Create output directories
    yolo_images_dir = os.path.join(output_dir, 'images')
    yolo_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # Class mapping
    # 0: circle_ori
    # 1: arrow_ori
    # 2: chest_ori

    copied_images = 0
    skipped_images = 0
    multi_label_skipped = 0

    for idx, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Creating YOLO dataset"):
        image_name = row['image_name']

        # Count how many crop labels this image has
        crop_count = sum([
            pd.notna(row['circle_ori_x1']),
            pd.notna(row['arrow_ori_x1']),
            pd.notna(row['chest_ori_x1'])
        ])

        # Skip images with more than one crop label
        if crop_count > 1:
            multi_label_skipped += 1
            continue

        src_image_path = os.path.join(images_dir, image_name)

        # Check if source image exists
        if not os.path.exists(src_image_path):
            skipped_images += 1
            continue

        # Load and convert to grayscale
        img = cv2.imread(src_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped_images += 1
            continue

        # Get original dimensions
        orig_h, orig_w = img.shape[:2]

        # Crop to bottom half only
        bottom_half_start = orig_h // 2

        # Check if any annotations would be removed due to bottom half crop
        # If so, skip this entire image
        skip_image = False
        if not row['empty']:
            # Check circle_ori
            if pd.notna(row['circle_ori_x1']):
                y1 = row['circle_ori_y1'] - bottom_half_start
                y2 = row['circle_ori_y2'] - bottom_half_start
                if y1 < 0 or y2 <= 0:
                    skip_image = True

            # Check arrow_ori
            if pd.notna(row['arrow_ori_x1']):
                y1 = row['arrow_ori_y1'] - bottom_half_start
                y2 = row['arrow_ori_y2'] - bottom_half_start
                if y1 < 0 or y2 <= 0:
                    skip_image = True

            # Check chest_ori
            if pd.notna(row['chest_ori_x1']):
                y1 = row['chest_ori_y1'] - bottom_half_start
                y2 = row['chest_ori_y2'] - bottom_half_start
                if y1 < 0 or y2 <= 0:
                    skip_image = True

        if skip_image:
            skipped_images += 1
            continue

        # Crop the image
        img = img[bottom_half_start:, :]

        # Save cropped grayscale image
        dst_image_path = os.path.join(yolo_images_dir, image_name)
        cv2.imwrite(dst_image_path, img)

        # Get new dimensions after crop
        img_h, img_w = img.shape[:2]

        # Create label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)

        # Convert bounding boxes to YOLO format
        yolo_labels = []

        # Check if this is an empty image
        if row['empty']:
            # Empty image - create empty label file (negative sample)
            pass
        else:
            # Process circle_ori
            if pd.notna(row['circle_ori_x1']):
                x1, y1 = row['circle_ori_x1'], row['circle_ori_y1']
                x2, y2 = row['circle_ori_x2'], row['circle_ori_y2']

                # Adjust y coordinates for bottom half crop
                y1 -= bottom_half_start
                y2 -= bottom_half_start

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Process arrow_ori
            if pd.notna(row['arrow_ori_x1']):
                x1, y1 = row['arrow_ori_x1'], row['arrow_ori_y1']
                x2, y2 = row['arrow_ori_x2'], row['arrow_ori_y2']

                # Adjust y coordinates for bottom half crop
                y1 -= bottom_half_start
                y2 -= bottom_half_start

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                yolo_labels.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Process chest_ori
            if pd.notna(row['chest_ori_x1']):
                x1, y1 = row['chest_ori_x1'], row['chest_ori_y1']
                x2, y2 = row['chest_ori_x2'], row['chest_ori_y2']

                # Adjust y coordinates for bottom half crop
                y1 -= bottom_half_start
                y2 -= bottom_half_start

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                yolo_labels.append(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

        copied_images += 1

    # Create data.yaml file
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO dataset configuration\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: 3\n")
        f.write(f"names: ['circle_ori', 'arrow_ori', 'chest_ori']\n")

    print(f"\nYOLO Dataset created:")
    print(f"  Images copied: {copied_images}")
    print(f"  Images skipped (not found): {skipped_images}")
    print(f"  Images skipped (multiple labels): {multi_label_skipped}")
    print(f"  Output directory: {output_dir}")
    print(f"  Images: {yolo_images_dir}")
    print(f"  Labels: {yolo_labels_dir}")
    print(f"  Config: {yaml_path}")

    # Count negative samples
    negative_samples = df_labeled['empty'].sum()
    positive_samples = len(df_labeled) - negative_samples
    print(f"\nDataset statistics:")
    print(f"  Positive samples (with labels): {positive_samples}")
    print(f"  Negative samples (empty): {negative_samples}")

def main():
    client = lb.Client(api_key=CONFIG['LABELBOX_API_KEY'])

    # Project ID
    project_id = "cmla2poo4054f07sk53rfe8zj"

    # Export from project
    export_data = export_labelbox_annotations(
        project_id,
        "Orientation Crops",
        client
    )

    # Parse annotations
    df_labeled = parse_orientation_annotations(export_data, "Orientation Crops")

    # Check if dataframe is empty
    if len(df_labeled) == 0:
        print("\n⚠ WARNING: No annotations found in project!")
        return

    # Remove duplicates - keep first occurrence
    rows_before = len(df_labeled)
    df_labeled = df_labeled.drop_duplicates(subset=['image_name'], keep='first')
    rows_after = len(df_labeled)

    if rows_before != rows_after:
        print(f"\nRemoved {rows_before - rows_after} duplicate image entries")

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Total labeled images: {len(df_labeled)}")

    print(f"\nCrop counts:")
    print(f"  circle_ori: {df_labeled['circle_ori_x1'].notna().sum()}")
    print(f"  arrow_ori: {df_labeled['arrow_ori_x1'].notna().sum()}")
    print(f"  chest_ori: {df_labeled['chest_ori_x1'].notna().sum()}")
    print(f"  empty: {df_labeled['empty'].sum()}")

    print(f"\nColumns: {list(df_labeled.columns)}")

    # Save to CSV
    output_file = "C:/Users/Tristan/Desktop/orientation_crops_labeled.csv"
    df_labeled.to_csv(output_file, index=False)
    print(f"\nSaved labeled data to: {output_file}")

    # Create YOLO dataset
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
    images_dir = os.path.join(tools_dir, 'inputs')
    yolo_output_dir = "C:/Users/Tristan/Desktop/orientation_yolo_dataset"

    create_yolo_dataset(df_labeled, images_dir, yolo_output_dir)

    print("\n✓ Complete!")

if __name__ == "__main__":
    main()
