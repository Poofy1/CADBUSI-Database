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

# === SINGLE PROJECT ===
project = client.get_project("cmkn2bljg1ehp07v909jr9zbx")
dataset_name = "2026_1_20_lesion_seg"
mask_dir = "C:/Users/Tristan/Desktop/gold_biopsy_masks/"
df = pd.read_csv("C:/Users/Tristan/Desktop/seg_data2.csv")

print(f"Project: {project.name}")

# === ENSURE PROJECT HAS ONTOLOGY ===
if not project.ontology():
    print(f"\nCreating ontology for {project.name}...")
    ontology_builder = lb.OntologyBuilder(
        tools=[lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="lesion_seg")]
    )
    ontology = client.create_ontology(
        f"Segmentation Ontology - {project.name}",
        ontology_builder.asdict(),
        media_type=lb.MediaType.Image
    )
    project.connect_ontology(ontology)
    print(f"  Ontology attached!")
else:
    print(f"\n{project.name} already has ontology: {project.ontology().name}")

# === DELETE ALL EXISTING BATCHES ===
print("\nDeleting existing batches...")
for batch in project.batches():
    try:
        batch.delete()
        print(f"  Deleted: {batch.name}")
    except Exception as e:
        print(f"  Failed: {e}")

# === GET DATASET MAPPING ===
dataset = None
for ds in client.get_datasets():
    if ds.name == dataset_name:
        dataset = ds
        break

dataset_mapping = {}
for dr in dataset.data_rows():
    dataset_mapping[dr.external_id] = dr.uid

# === READ CSV ===
df['data_row_id'] = df['caliper_image'].map(dataset_mapping)
df = df[df['data_row_id'].notna()].copy()

df_accept = df[df['status'] == 'Accept'].copy()
df_review = df[df['status'] == 'Needs Review'].copy()

print(f"\nAccept rows: {len(df_accept)}")
print(f"Needs Review rows: {len(df_review)}")


def mask_to_base64_png(mask_path):
    """Load binary mask and convert to base64-encoded PNG."""
    img = Image.open(mask_path).convert('L')
    arr = np.array(img)
    
    if arr.max() > 1:
        arr = (arr > 127).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    
    mask_img = Image.fromarray(arr, mode='L')
    
    buffer = BytesIO()
    mask_img.save(buffer, format='PNG')
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return b64_str


def create_batch_and_upload(project, df_source, batch_name, priority):
    """Create batch and upload mask annotations."""
    
    print(f"\n{'='*50}")
    print(f"Processing: {batch_name}")
    print(f"{'='*50}")
    
    # Create batch
    data_row_ids = df_source['data_row_id'].unique().tolist()
    print(f"Creating batch with {len(data_row_ids)} data rows (priority={priority})...")
    
    try:
        batch = project.create_batch(
            batch_name,
            data_rows=data_row_ids,
            priority=priority
        )
        print(f"Created batch: {batch.uid}")
    except Exception as e:
        print(f"Batch error: {e}")
        return 0, 0
    
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
        name=f"seg_annotations_{batch_name}_{timestamp}",
        predictions=labels_ndjson
    )
    
    upload_job.wait_till_done()
    
    successes = sum(1 for s in upload_job.statuses if s.get('status') == 'SUCCESS')
    failures = sum(1 for s in upload_job.statuses if s.get('status') == 'FAILURE')
    
    print(f"State: {upload_job.state}")
    print(f"Success: {successes}/{len(labels_ndjson)}")
    
    if upload_job.errors:
        print(f"Errors: {upload_job.errors[:5]}")
    
    failure_reasons = [s for s in upload_job.statuses if s.get('status') == 'FAILURE']
    if failure_reasons:
        print(f"First 3 failure details:")
        for f in failure_reasons[:3]:
            print(f"  {f}")
    
    if failures > 0:
        print(f"Failures: {failures}")
    
    return successes, failures + skipped


# === PROCESS BOTH BATCHES (Needs Review first with higher priority) ===
successes_review, failures_review = create_batch_and_upload(
    project, df_review, "NeedsReview-batch", priority=10
)
successes_accept, failures_accept = create_batch_and_upload(
    project, df_accept, "Accept-batch", priority=5
)

# === SUMMARY ===
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Needs Review batch: {successes_review} success, {failures_review} failed/skipped")
print(f"Accept batch:       {successes_accept} success, {failures_accept} failed/skipped")
print(f"\nProject URL: https://app.labelbox.com/projects/{project.uid}")