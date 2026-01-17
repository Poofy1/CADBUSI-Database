import labelbox as lb
import pandas as pd
import numpy as np
from PIL import Image
import sys
import os
from datetime import datetime
import uuid as uuid_lib
import base64
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

client = lb.Client(api_key=CONFIG['LABELBOX_API_KEY'])

# === TWO PROJECTS ===
project_accept = client.get_project("cmkhda98508i10725fz3e4fbr")  # Accept
project_review = client.get_project("cmkhdavvv196i072s626j559t")  # Needs Review

print(f"Project 1 (Accept): {project_accept.name}")
print(f"Project 2 (Needs Review): {project_review.name}")

# === ENSURE BOTH PROJECTS HAVE ONTOLOGY (Segmentation tool) ===
for proj in [project_accept, project_review]:
    if not proj.ontology():
        print(f"\nCreating ontology for {proj.name}...")
        ontology_builder = lb.OntologyBuilder(
            tools=[lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="lesion_seg")]
        )
        ontology = client.create_ontology(
            f"Segmentation Ontology - {proj.name}",
            ontology_builder.asdict(),
            media_type=lb.MediaType.Image
        )
        proj.connect_ontology(ontology)
        print(f"  Ontology attached!")
    else:
        print(f"\n{proj.name} already has ontology: {proj.ontology().name}")

# === DELETE ALL EXISTING BATCHES FROM BOTH ===
print("\nDeleting batches from both projects...")
for proj in [project_accept, project_review]:
    for batch in proj.batches():
        try:
            batch.delete()
            print(f"  Deleted: {batch.name} from {proj.name}")
        except Exception as e:
            print(f"  Failed: {e}")

# === GET DATASET MAPPING ===
dataset_name = "2026_1_15_lesion_crop+seg"
dataset = None
for ds in client.get_datasets():
    if ds.name == dataset_name:
        dataset = ds
        break

dataset_mapping = {}
for dr in dataset.data_rows():
    dataset_mapping[dr.external_id] = dr.uid

# === READ CSV ===
mask_dir = "C:/Users/Tristan/Desktop/masks/"
df = pd.read_csv("C:/Users/Tristan/Desktop/seg_data.csv")
df['data_row_id'] = df['caliper_image'].map(dataset_mapping)
df = df[df['data_row_id'].notna()].copy()

df_accept = df[df['status'] == 'Accept'].sample(n=130, random_state=42).reset_index(drop=True)
df_review = df[df['status'] == 'Needs Review'].copy()

print(f"\nAccept rows (sampled): {len(df_accept)}")
print(f"Needs Review rows (all): {len(df_review)}")


def mask_to_base64_png(mask_path):
    """Load binary mask and convert to base64-encoded PNG.
    Labelbox requires 2D grayscale PNG with values 0 and 1 only.
    """
    img = Image.open(mask_path).convert('L')  # Grayscale
    arr = np.array(img)
    
    # Convert to strict binary (0 or 1)
    if arr.max() > 1:
        arr = (arr > 127).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    
    # Create grayscale image with values 0 and 1
    mask_img = Image.fromarray(arr, mode='L')
    
    # Encode to base64
    buffer = BytesIO()
    mask_img.save(buffer, format='PNG')
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return b64_str


# === HELPER FUNCTION ===
def process_project(project, df_source, label_name):
    """Create batch and upload mask annotations."""
    
    print(f"\n{'='*50}")
    print(f"Processing: {project.name} ({label_name})")
    print(f"{'='*50}")
    
    # Create batch
    data_row_ids = df_source['data_row_id'].unique().tolist()
    print(f"Creating batch with {len(data_row_ids)} data rows...")
    
    try:
        batch = project.create_batch(
            f"{label_name}-batch",
            data_rows=data_row_ids,
            priority=5
        )
        print(f"Created batch: {batch.uid}")
    except Exception as e:
        print(f"Batch error: {e}")
    
    # Build annotations
    labels_ndjson = []
    skipped = 0
    
    for idx, row in df_source.iterrows():
        mask_path = os.path.join(mask_dir, row['mask_file'])
        
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask not found: {mask_path}")
            skipped += 1
            continue
        
        try:
            mask_b64 = mask_to_base64_png(mask_path)
            
            annotation = {
                "uuid": str(uuid_lib.uuid4()),
                "name": "lesion_seg",
                "dataRow": {"id": row['data_row_id']},
                "mask": {
                    "png": mask_b64
                }
            }
            labels_ndjson.append(annotation)
            
        except Exception as e:
            print(f"  Error processing {row['mask_file']}: {e}")
            skipped += 1
    
    print(f"Created {len(labels_ndjson)} annotations (skipped {skipped})")
    
    if not labels_ndjson:
        print("No labels to upload!")
        return 0, skipped
    
    # Upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Uploading...")
    
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name=f"seg_annotations_{label_name}_{timestamp}",
        predictions=labels_ndjson
    )
    
    upload_job.wait_till_done()
    
    successes = sum(1 for s in upload_job.statuses if s.get('status') == 'SUCCESS')
    failures = sum(1 for s in upload_job.statuses if s.get('status') == 'FAILURE')
    
    print(f"State: {upload_job.state}")
    print(f"Success: {successes}/{len(labels_ndjson)}")
    
    # Print first few errors for debugging
    if upload_job.errors:
        print(f"Errors: {upload_job.errors[:5]}")
    
    # Check individual statuses for error messages
    failure_reasons = [s for s in upload_job.statuses if s.get('status') == 'FAILURE']
    if failure_reasons:
        print(f"First 3 failure details:")
        for f in failure_reasons[:3]:
            print(f"  {f}")
    
    if failures > 0:
        print(f"Failures: {failures}")
    
    return successes, failures + skipped


# === PROCESS BOTH PROJECTS ===
successes1, failures1 = process_project(project_accept, df_accept, "Accept")
successes2, failures2 = process_project(project_review, df_review, "NeedsReview")

# === SUMMARY ===
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Project 1 (Accept):       {successes1} success, {failures1} failed/skipped")
print(f"Project 2 (Needs Review): {successes2} success, {failures2} failed/skipped")
print(f"\nProject 1 URL: https://app.labelbox.com/projects/{project_accept.uid}")
print(f"Project 2 URL: https://app.labelbox.com/projects/{project_review.uid}")