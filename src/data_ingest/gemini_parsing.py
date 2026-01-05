import os
import json
import re
import pandas as pd
import time
from pathlib import Path
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import CONFIG
from labeling.findings_prepare import anonymize_dates_times_and_names

# Initialize Vertex AI client
os.environ['GOOGLE_CLOUD_PROJECT'] = CONFIG['env']['project_id']
os.environ['GOOGLE_CLOUD_LOCATION'] = CONFIG['env']['region']
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

client = genai.Client(http_options=HttpOptions(api_version="v1"))
storage_client = storage.Client()

SYSTEM_PROMPT = """You are a medical data extraction assistant specializing in breast imaging reports.

Your task: Extract ALL lesions mentioned in radiology text and output them as a JSON array.

Each lesion should have these fields:
- direction: clock position (e.g., "2:00", "12:00") or "na" if not specified
- distance: distance from nipple (e.g., "4cm", "2cm") or "na" if not specified  
- size: maximum dimension (e.g., "5mm", "1.2cm") or "na" if not specified
- type: lesion type (e.g., "mass", "cyst", "nodule", "complex") or "na" if unclear

CRITICAL RULES:
1. When multiple locations share the same size/type (distributed attributes), create separate entries for EACH location with the shared attributes repeated.
2. Use "na" for any missing values - never guess.
3. Output ONLY valid JSON - no explanation, no markdown code blocks.
4. Order lesions as they appear in the text.
5. Normalize units: keep as written (don't convert mm to cm or vice versa).
6. For size ranges like "up to 3mm", use the maximum value.

Output format:
[{"direction": "...", "distance": "...", "size": "...", "type": "..."}, ...]

If no lesions are found, output: []
"""

FEW_SHOT_EXAMPLES = [
    (
        """Additional evaluation was performed for an asymmetry within the left 
        upper outer breast, which persists as a well-circumscribed ovoid 5 mm
        mass on additional diagnostic imaging. Targeted ultrasound was performed 
        within the left upper outer quadrant which demonstrates a well-circumscribed 
        fibrocystic complex measuring 5 mm x 2 mm x 2 mm in the left breast 2:00 4 cm from the 
        nipple. Incidental note made of a 3 mm anechoic cyst within the left 6-7 o'clock 
        periareolar position. No suspicious masses or other abnormalities 
        are identified.""",
        [
            {"direction": "2:00", "distance": "4cm", "size": "5mm", "type": "mass"},
            {"direction": "6:30", "distance": "0cm", "size": "3mm", "type": "cyst"}
        ]
    ),
    (
        """Ultrasound of the left breast upper outer quadrant at 1:00, 2 cm from 
        the nipple, 2:00, 3 cm from the nipple, and 3:00 6 cm from the nipple 
        show multiple small benign cysts measuring up to 3 mm x 3 mm x 2 mm 
        which account for the mammographic findings.""",
        [
            {"direction": "1:00", "distance": "2cm", "size": "3mm", "type": "cyst"},
            {"direction": "2:00", "distance": "3cm", "size": "3mm", "type": "cyst"},
            {"direction": "3:00", "distance": "6cm", "size": "3mm", "type": "cyst"}
        ]
    ),
    (
        """There is an oval hypoechoic mass measuring 3.8 x 1.4 x 3.6 cm with well defined, 
        thin margins in the right breast upper outer quadrant at 10 o'clock.  The mass is parallel to the chest wall.  
        This is at the site of palpable concern marked on skin. Ultrasound-guided biopsy recommended.    
        Findings and recommendations were discussed with the patient and Dr. Hines.    Ultrasound guided core biopsy is recommended.       
        BI-RADS Category 4: Suspicious Abnormality       Addendum:    
        Pathology results from US-guided right breast biopsy in the 10 o'clock position performed on 10/7/11 reveal fibroadenoma.  
        (Refer to pathology report for detailed description.) This is a benign, concordant and specific diagnosis. Given size of lesion, 
        surgical consult recommended for excision.""",
        [
            {"direction": "10:00", "distance": "na", "size": "3.8cm", "type": "mass"}
        ]
    ),
]


def build_vertex_contents(text: str) -> list[dict]:
    """Build Vertex AI contents format with few-shot examples."""
    contents = []
    
    # Add system prompt as first user message
    contents.append({
        "role": "user",
        "parts": [{"text": SYSTEM_PROMPT}]
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Understood. I will extract lesions according to these rules."}]
    })
    
    # Add few-shot examples
    for example_input, example_output in FEW_SHOT_EXAMPLES:
        contents.append({
            "role": "user",
            "parts": [{"text": example_input.strip()}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": json.dumps(example_output)}]
        })
    
    # Add actual query
    contents.append({
        "role": "user",
        "parts": [{"text": text.strip()}]
    })
    
    return contents


def format_as_label(lesions: list[dict]) -> str:
    """Convert lesion dicts back to label format: [dir, dist, size, type]"""
    labels = []
    for l in lesions:
        label = f"[{l['direction']}, {l['distance']}, {l['size']}, {l['type']}]"
        labels.append(label)
    return ", ".join(labels)


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload file to GCS and return gs:// URI."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def download_from_gcs(gcs_uri: str, local_path: str):
    """Download file from GCS."""
    # Parse gs://bucket/path
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def get_batch_error_details(job):
    """Get detailed error information from a failed batch job."""
    print(f"\n{'='*60}")
    print("ERROR DETAILS")
    print(f"{'='*60}")
    print(f"Job name: {job.name}")
    print(f"State: {job.state}")
    
    # Try to get error from job object
    if hasattr(job, 'error'):
        print(f"Error: {job.error}")
    
    # The job object might have more details
    print(f"\nFull job info:")
    print(job)
    print(f"{'='*60}")
    
def submit_batch(
    input_jsonl_path: str,
    gcs_bucket: str,
    gcs_input_prefix: str,
    gcs_output_prefix: str,
    model: str = "gemini-2.5-flash-lite"
):
    """
    Submit a batch job to Vertex AI.
    
    Args:
        input_jsonl_path: Local path to JSONL file
        gcs_bucket: GCS bucket name (without gs://)
        gcs_input_prefix: Path prefix in bucket for input
        gcs_output_prefix: Path prefix in bucket for output
        model: Gemini model to use
    
    Returns:
        Batch job object
    """
    print(f"\nSubmitting batch from: {input_jsonl_path}")
    
    # Upload input file to GCS
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_blob_name = f"{gcs_input_prefix}/batch_input_{timestamp}.jsonl"
    input_uri = upload_to_gcs(input_jsonl_path, gcs_bucket, input_blob_name)
    print(f"Uploaded input to: {input_uri}")
    
    # Create output URI
    output_uri = f"gs://{gcs_bucket}/{gcs_output_prefix}/batch_output_{timestamp}/"
    
    # Submit batch job
    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(dest=output_uri)
    )
    
    print(f"\nBatch job created: {job.name}")
    print(f"Job state: {job.state}")
    print(f"Output will be at: {output_uri}")
    
    return job, output_uri


def check_batch_status(job_name: str):
    """Check the status of a batch job."""
    # List all batches and find the one with matching name
    for job in client.batches.list():
        if job.name == job_name:
            print(f"\nJob name: {job.name}")
            print(f"State: {job.state}")
            print(f"Create time: {job.create_time}")
            return job
    
    raise ValueError(f"Job {job_name} not found")


def wait_for_batch(job_name: str, check_interval: int = 60):
    """Wait for batch job to complete."""
    print(f"\nWaiting for batch job to complete...")
    print(f"Checking every {check_interval} seconds...")
    
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_EXPIRED
    }
    
    start_time = time.time()
    
    while True:
        # Find the job by listing and filtering
        job = None
        for batch_job in client.batches.list():
            if batch_job.name == job_name:
                job = batch_job
                break
        
        if job is None:
            raise ValueError(f"Job {job_name} not found")
        
        elapsed = time.time() - start_time
        print(f"\rState: {job.state.name} | Elapsed: {elapsed/60:.1f}m", end="", flush=True)
        
        if job.state in completed_states:
            print(f"\n\nBatch completed with state: {job.state.name}")
            print(f"Total time: {elapsed/60:.1f} minutes")
            return job
        
        time.sleep(check_interval)


def download_batch_results(output_uri: str, local_output_dir: str):
    """
    Download results from GCS. Results should be in a single file.
    """
    print(f"\nDownloading results from: {output_uri}")
    
    # Parse bucket and prefix
    parts = output_uri.replace("gs://", "").rstrip("/").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Download all result files
    result_files = []
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            local_path = Path(local_output_dir) / Path(blob.name).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            result_files.append(str(local_path))
            print(f"Downloaded: {blob.name}")
    
    if not result_files:
        raise ValueError("No result files found!")
    
    # Sort files alphabetically (important if multiple files)
    result_files.sort()
    
    print(f"\n{len(result_files)} result file(s) found")
    if len(result_files) > 1:
        print("WARNING: Multiple result files - processing in alphabetical order:")
        for i, f in enumerate(result_files):
            print(f"  {i+1}. {Path(f).name}")
    
    return result_files

def create_batch_jsonl(
    csv_path: str, 
    output_jsonl: str, 
    text_column: str = "ultrasound_findings",
    model: str = "gemini-2.5-flash-lite"
):
    """Create JSONL with custom_id for result mapping."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    row_indices = []
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            text = row[text_column]
            
            if pd.isna(text) or not str(text).strip():
                continue
            
            anonymized_text = anonymize_dates_times_and_names(str(text))
            contents = build_vertex_contents(anonymized_text)
            
            # ADD custom_id to track which row this is
            batch_request = {
                "custom_id": f"row_{int(idx)}",  # <-- FIX HERE
                "request": {
                    "contents": contents
                }
            }
            
            f.write(json.dumps(batch_request) + '\n')
            row_indices.append(int(idx))
    
    # Still save indices file for reference
    index_file = output_jsonl.replace('.jsonl', '_indices.json')
    with open(index_file, 'w') as f:
        json.dump(row_indices, f)
    
    print(f"\nCreated batch file: {output_jsonl}")
    print(f"Row index mapping: {index_file}")
    print(f"Total requests: {len(row_indices)}")
    
    return len(row_indices)


def parse_batch_results_with_comparison(
    result_files: list[str],
    original_csv: str,
    output_csv: str,
    index_file: str = None  # No longer needed
):
    """Parse results using custom_id from each result."""
    print(f"\nParsing results from {len(result_files)} file(s)")
    print(f"Loading original data from: {original_csv}")
    
    # Load original CSV
    df_original = pd.read_csv(original_csv)
    
    results = []
    
    # Parse all result files
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                
                # Extract row index from custom_id
                custom_id = result.get('custom_id', '')
                if not custom_id.startswith('row_'):
                    print(f"Warning: Unexpected custom_id format: {custom_id}")
                    continue
                
                row_idx = int(custom_id.split('_')[1])
                
                # Parse response
                try:
                    response = result.get('response', {})
                    candidates = response.get('candidates', [])
                    
                    if not candidates:
                        raise ValueError("No candidates in response")
                    
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    
                    if not parts:
                        raise ValueError("No parts in content")
                    
                    raw_output = parts[0].get('text', '').strip()
                    
                    if raw_output.startswith("```"):
                        raw_output = re.sub(r"```(?:json)?\n?", "", raw_output).strip()
                    
                    lesions = json.loads(raw_output)
                    prediction = format_as_label(lesions)
                    prediction_json = json.dumps(lesions)
                    num_lesions = len(lesions)
                    error = None
                    
                except Exception as e:
                    prediction = ""
                    prediction_json = "[]"
                    num_lesions = 0
                    error = str(e)
                
                # Get original row
                try:
                    original_row = df_original.loc[row_idx]
                except KeyError:
                    print(f"Warning: row_idx {row_idx} not found in DataFrame")
                    continue
                
                results.append({
                    'row_index': row_idx,
                    'custom_id': custom_id,
                    'input_text': original_row.get('ultrasound_findings', ''),
                    'structured_output': original_row.get('structured_output', ''),
                    'prediction': prediction,
                    'prediction_json': prediction_json,
                    'num_lesions': num_lesions,
                    'error': error
                })
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('row_index')
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(df)} results to: {output_csv}")
    print(f"Successful extractions: {df['error'].isna().sum()}")
    print(f"Errors: {df['error'].notna().sum()}")
    
    # Print sample comparison
    if len(df) > 0:
        print("\n=== SAMPLE COMPARISON ===")
        sample = df[df['error'].isna()].head(3)
        for idx, row in sample.iterrows():
            print(f"\n--- Row {row['row_index']} ---")
            print(f"Input: {row['input_text'][:100]}...")
            print(f"Original: {row['structured_output']}")
            print(f"Predicted: {row['prediction']}")
    
    return df

def run_batch_pipeline(
    csv_path: str,
    gcs_bucket: str,
    text_column: str = "ultrasound_findings",
    model: str = "gemini-2.5-flash-lite",
    output_dir: str = "batch_results",
    gcs_input_prefix: str = "batch_input",
    gcs_output_prefix: str = "batch_output",
    wait: bool = True
):
    """
    Complete Vertex AI batch pipeline.
    
    Args:
        csv_path: Path to input CSV
        gcs_bucket: GCS bucket name (without gs://)
        text_column: Column containing text to process
        model: Gemini model to use
        output_dir: Local directory to save outputs
        gcs_input_prefix: GCS prefix for input files
        gcs_output_prefix: GCS prefix for output files
        wait: Whether to wait for batch completion
    
    Returns:
        Dictionary with job info and output paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_jsonl = output_path / f"batch_input_{timestamp}.jsonl"
    results_csv = output_path / f"batch_results_{timestamp}.csv"
    
    print("="*60)
    print("VERTEX AI GEMINI BATCH PIPELINE")
    print("="*60)
    print(f"Model: {model}")
    print(f"GCS Bucket: gs://{gcs_bucket}")
    print("="*60)
    
    # Step 1: Create batch file
    num_requests = create_batch_jsonl(csv_path, str(batch_jsonl), text_column, model)
    
    # Step 2: Submit batch
    job, output_uri = submit_batch(
        str(batch_jsonl),
        gcs_bucket,
        gcs_input_prefix,
        gcs_output_prefix,
        model
    )
    
    # Save job name to file
    job_name_file = output_path / f"job_name_{timestamp}.txt"
    with open(job_name_file, 'w') as f:
        f.write(job.name)
    print(f"\nJob name saved to: {job_name_file}")
    
    if not wait:
        print("\n" + "="*60)
        print("Batch submitted! Not waiting for completion.")
        print(f"To check status later, use: check_batch_status('{job.name}')")
        print("="*60)
        return {
            'job_name': job.name,
            'job_name_file': str(job_name_file),
            'output_uri': output_uri
        }
    
    # Step 3: Wait for completion
    job = wait_for_batch(job.name)
    
    if job.state != JobState.JOB_STATE_SUCCEEDED:
        print(f"\nBatch did not succeed: {job.state}")
        get_batch_error_details(job)
        return {
            'job_name': job.name,
            'status': job.state,
            'output_uri': output_uri
        }
    
    # Step 4: Download results
    result_files = download_batch_results(output_uri, str(output_path))
    
    # Step 5: Parse results with comparison
    df = parse_batch_results_with_comparison(
        result_files,
        csv_path,
        str(results_csv),
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Job name: {job.name}")
    print(f"Input file: {batch_jsonl}")
    print(f"Results CSV: {results_csv}")
    print(f"GCS output: {output_uri}")
    print("="*60)
    
    return {
        'job_name': job.name,
        'batch_input': str(batch_jsonl),
        'results_csv': str(results_csv),
        'output_uri': output_uri,
        'dataframe': df
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    result = run_batch_pipeline(
        csv_path=f'{parent_dir}/training/dataset/t5_data/train.csv',
        gcs_bucket=CONFIG['BUCKET'],
        text_column='ultrasound_findings',
        model='gemini-2.5-flash',
        output_dir='batch_results_gemini',
        gcs_input_prefix='cadbusi/batch_input',
        gcs_output_prefix='cadbusi/batch_output',
        wait=True
    )