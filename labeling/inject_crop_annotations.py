import labelbox as lb
import pandas as pd
import sys
import os
from datetime import datetime
import uuid as uuid_lib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

client = lb.Client(api_key=CONFIG['LABELBOX_API_KEY'])

# === TWO PROJECTS ===
project_accept = client.get_project("cmkg6htrn0af207z35frjaaz5")  # 400 Accept
project_review = client.get_project("cmkgil9ei1mb207x037954tov")  # All Needs Review

print(f"Project 1 (Accept): {project_accept.name}")
print(f"Project 2 (Needs Review): {project_review.name}")

# === ENSURE BOTH PROJECTS HAVE ONTOLOGY ===
for proj in [project_accept, project_review]:
    if not proj.ontology():
        print(f"\nCreating ontology for {proj.name}...")
        ontology_builder = lb.OntologyBuilder(
            tools=[lb.Tool(tool=lb.Tool.Type.BBOX, name="lesion_crop")]
        )
        ontology = client.create_ontology(
            f"Crop Detection Ontology - {proj.name}",
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
df = pd.read_csv("C:/Users/Tristan/Desktop/crop_data.csv")
df['data_row_id'] = df['caliper_image'].map(dataset_mapping)
df = df[df['data_row_id'].notna()].copy()

df_accept = df[df['status'] == 'Accept'].sample(n=400, random_state=42).reset_index(drop=True)
df_review = df[df['status'] == 'Needs Review'].copy()

print(f"\nAccept rows (sampled): {len(df_accept)}")
print(f"Needs Review rows (all): {len(df_review)}")


# === HELPER FUNCTION ===
def process_project(project, df_source, label_name):
    """Create batch and upload annotations."""
    
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
    for idx, row in df_source.iterrows():
        annotation = {
            "uuid": str(uuid_lib.uuid4()),
            "name": "lesion_crop",
            "dataRow": {"id": row['data_row_id']},
            "bbox": {
                "top": int(row['y1']),
                "left": int(row['x1']),
                "height": int(row['y2'] - row['y1']),
                "width": int(row['x2'] - row['x1'])
            }
        }
        labels_ndjson.append(annotation)
    
    print(f"Created {len(labels_ndjson)} annotations")
    
    # Upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Uploading...")
    
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name=f"annotations_{label_name}_{timestamp}",
        predictions=labels_ndjson
    )
    
    upload_job.wait_till_done()
    
    successes = sum(1 for s in upload_job.statuses if s.get('status') == 'SUCCESS')
    failures = sum(1 for s in upload_job.statuses if s.get('status') == 'FAILURE')
    
    print(f"State: {upload_job.state}")
    print(f"Success: {successes}/{len(labels_ndjson)}")
    
    if failures > 0:
        print(f"Failures: {failures}")
    
    return successes, failures


# === PROCESS BOTH PROJECTS ===
successes1, failures1 = process_project(project_accept, df_accept, "Accept")
successes2, failures2 = process_project(project_review, df_review, "NeedsReview")

# === SUMMARY ===
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Project 1 (Accept):       {successes1} success, {failures1} failed")
print(f"Project 2 (Needs Review): {successes2} success, {failures2} failed")
print(f"\nProject 1 URL: https://app.labelbox.com/projects/{project_accept.uid}")
print(f"Project 2 URL: https://app.labelbox.com/projects/{project_review.uid}")