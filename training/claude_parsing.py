import anthropic
import os
import json
import re
import pandas as pd
import time
from pathlib import Path
# Add parent directory to path
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'labeling'))

from config import CONFIG
from labeling.findings_prepare import anonymize_dates_times_and_names

client = anthropic.Anthropic(api_key=CONFIG['CLAUDE_API_KEY'])

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

# =============================================================================
# FEW-SHOT EXAMPLES - Add your labeled examples here
# Format: (input_text, expected_output)
# =============================================================================
FEW_SHOT_EXAMPLES = [
    # Example 1: Multiple lesions, straightforward
    (
        """Additional evaluation was performed for an asymmetry within the left 
        upper outer breast, which persists as a well-circumscribed ovoid 5 mm 
        mass on additional diagnostic imaging. Targeted ultrasound was performed 
        within the left upper outer quadrant which demonstrates a well-circumscribed 
        fibrocystic complex measuring 5 mm in the left breast 2:00 4 cm from the 
        nipple. Incidental note made of a 3 mm anechoic cyst within the left 1:00 
        breast periareolar. No suspicious masses or other abnormalities 
        are identified.""",
        [
            {"direction": "2:00", "distance": "4cm", "size": "5mm", "type": "mass"},
            {"direction": "1:00", "distance": "0cm", "size": "3mm", "type": "cyst"}
        ]
    ),
    
    # Example 2: Distributed attributes - multiple locations, shared size/type
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
    
    # Example 3: Single lesion with missing values
    (
        """There is a 19mm hypoechoic mass identified in the right breast.""",
        [
            {"direction": "na", "distance": "na", "size": "19mm", "type": "mass"}
        ]
    ),
    
]


def build_messages(text: str) -> list[dict]:
    """Build messages with prompt caching."""
    messages = []
    
    # Add few-shot examples with cache control on LAST example
    for i, (example_input, example_output) in enumerate(FEW_SHOT_EXAMPLES):
        messages.append({"role": "user", "content": example_input.strip()})
        
        # Mark the last assistant message for caching
        assistant_content = json.dumps(example_output)
        if i == len(FEW_SHOT_EXAMPLES) - 1:  # Last example
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })
        else:
            messages.append({"role": "assistant", "content": assistant_content})
    
    # Add actual query (NOT cached - this is what changes)
    messages.append({"role": "user", "content": text.strip()})
    
    return messages


def extract_lesions(text: str, model: str = "claude-haiku-4-5-20251001") -> list[dict]:
    """
    Extract lesions from radiology text.

    Args:
        text: The radiology report text (will be anonymized before sending to API)
        model: Claude model to use

    Returns:
        List of lesion dictionaries
    """
    # Anonymize text before sending to API
    anonymized_text = anonymize_dates_times_and_names(text)

    messages = build_messages(anonymized_text)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    raw_output = response.content[0].text.strip()

    # Parse JSON response
    try:
        # Handle potential markdown code blocks (shouldn't happen but just in case)
        if raw_output.startswith("```"):
            raw_output = re.sub(r"```(?:json)?\n?", "", raw_output).strip()

        lesions = json.loads(raw_output)
        return lesions
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw output: {raw_output}")
        return []


def format_as_label(lesions: list[dict]) -> str:
    """Convert lesion dicts back to your label format: [dir, dist, size, type]"""
    labels = []
    for l in lesions:
        label = f"[{l['direction']}, {l['distance']}, {l['size']}, {l['type']}]"
        labels.append(label)
    return ", ".join(labels)


def process_batch(texts: list[str], model: str = "claude-haiku-4-5-20251001") -> list[list[dict]]:
    """Process multiple texts. For large batches, consider using the Batch API instead."""
    results = []
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}...")
        try:
            lesions = extract_lesions(text, model)
            results.append(lesions)
        except Exception as e:
            print(f"Error on item {i}: {e}")
            results.append([])
    return results


# =============================================================================
# BATCH API FUNCTIONS
# =============================================================================

def create_batch_jsonl(csv_path: str, output_jsonl: str, text_column: str = "ultrasound_findings", model: str = "claude-haiku-4-5-20251001"):
    """
    Create a JSONL file for Claude Batch API from CSV data.

    Args:
        csv_path: Path to input CSV file
        output_jsonl: Path to output JSONL file
        text_column: Name of column containing the text to process
        model: Claude model to use

    Returns:
        Number of requests created
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df)}")

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            text = row[text_column]

            # Skip empty texts
            if pd.isna(text) or not str(text).strip():
                continue

            # Anonymize text
            anonymized_text = anonymize_dates_times_and_names(str(text))

            # Build messages with few-shot examples
            messages = build_messages(anonymized_text)

            # Create batch request
            batch_request = {
                "custom_id": f"row_{idx}",
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "system": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": messages
                }
            }

            f.write(json.dumps(batch_request) + '\n')

    # Count lines in file
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        num_requests = sum(1 for _ in f)

    print(f"\nCreated batch file: {output_jsonl}")
    print(f"Total requests: {num_requests}")

    return num_requests


def submit_batch(jsonl_path: str, description: str = "Radiology lesion extraction"):
    """
    Submit a batch job to Claude API.

    Args:
        jsonl_path: Path to JSONL batch file
        description: Optional description for the batch

    Returns:
        Batch ID
    """
    print(f"\nSubmitting batch from: {jsonl_path}")

    # Read requests from JSONL file
    requests = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            requests.append(json.loads(line))

    print(f"Loaded {len(requests)} requests")

    # Create the batch directly with requests
    batch = client.messages.batches.create(
        requests=requests
    )

    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.processing_status}")

    return batch.id

def check_batch_status(batch_id: str):
    """
    Check the status of a batch job.

    Args:
        batch_id: The batch ID to check

    Returns:
        Batch object with status information
    """
    batch = client.messages.batches.retrieve(batch_id)

    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")

    if batch.processing_status == "ended":
        print(f"Results file ID: {batch.results_url}")

    return batch


def wait_for_batch(batch_id: str, check_interval: int = 60):
    """
    Wait for a batch to complete, polling at regular intervals.

    Args:
        batch_id: The batch ID to wait for
        check_interval: Seconds between status checks (default: 60)

    Returns:
        Completed batch object
    """
    print(f"\nWaiting for batch {batch_id} to complete...")
    print(f"Checking every {check_interval} seconds...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        print(f"\rStatus: {status} | {batch.request_counts}", end="", flush=True)

        if status == "ended":
            print("\n\nBatch completed!")
            return batch
        elif status in ["failed", "expired", "canceled"]:
            print(f"\n\nBatch {status}!")
            return batch

        time.sleep(check_interval)


def download_batch_results(batch_id: str, output_jsonl: str):
    """
    Download results from a completed batch.
    """
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        raise ValueError(f"Batch not completed. Status: {batch.processing_status}")

    print(f"\nDownloading results from batch: {batch_id}")

    # Stream results and save to file
    num_results = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in client.messages.batches.results(batch_id):
            f.write(json.dumps(result.model_dump()) + '\n')
            num_results += 1

    print(f"Downloaded {num_results} results to: {output_jsonl}")

    return output_jsonl


def parse_batch_results(results_jsonl: str, output_csv: str):
    """
    Parse batch results JSONL and save to CSV.

    Args:
        results_jsonl: Path to results JSONL file
        output_csv: Path to output CSV file

    Returns:
        DataFrame with parsed results
    """
    print(f"\nParsing results from: {results_jsonl}")

    results = []

    with open(results_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)

            custom_id = result['custom_id']
            row_idx = int(custom_id.split('_')[1])

            # Check if request succeeded
            if result['result']['type'] == 'succeeded':
                response = result['result']['message']
                raw_output = response['content'][0]['text'].strip()

                # Parse JSON response
                try:
                    if raw_output.startswith("```"):
                        raw_output = re.sub(r"```(?:json)?\n?", "", raw_output).strip()

                    lesions = json.loads(raw_output)
                    prediction = format_as_label(lesions)
                    prediction_json = json.dumps(lesions)
                    num_lesions = len(lesions)
                    error = None
                except json.JSONDecodeError as e:
                    prediction = ""
                    prediction_json = "[]"
                    num_lesions = 0
                    error = f"JSON parse error: {str(e)}"
            else:
                # Request failed
                prediction = ""
                prediction_json = "[]"
                num_lesions = 0
                error = result['result'].get('error', {}).get('message', 'Unknown error')

            results.append({
                'row_index': row_idx,
                'custom_id': custom_id,
                'prediction': prediction,
                'prediction_json': prediction_json,
                'num_lesions': num_lesions,
                'error': error
            })

    # Create DataFrame and sort by row index
    df = pd.DataFrame(results)
    df = df.sort_values('row_index')

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} results to: {output_csv}")
    print(f"Successful extractions: {df['error'].isna().sum()}")
    print(f"Errors: {df['error'].notna().sum()}")

    return df

def parse_batch_results_with_comparison(results_jsonl: str, original_csv: str, output_csv: str):
    """
    Parse batch results and merge with original data for comparison.
    
    Args:
        results_jsonl: Path to results JSONL file
        original_csv: Path to original CSV with input data
        output_csv: Path to output CSV file
    
    Returns:
        DataFrame with input text, predictions, and original labels
    """
    print(f"\nParsing results from: {results_jsonl}")
    print(f"Loading original data from: {original_csv}")
    
    # Load original CSV
    df_original = pd.read_csv(original_csv)
    
    results = []
    
    with open(results_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            
            custom_id = result['custom_id']
            row_idx = int(custom_id.split('_')[1])
            
            # Check if request succeeded
            if result['result']['type'] == 'succeeded':
                response = result['result']['message']
                raw_output = response['content'][0]['text'].strip()
                
                # Parse JSON response
                try:
                    if raw_output.startswith("```"):
                        raw_output = re.sub(r"```(?:json)?\n?", "", raw_output).strip()
                    
                    lesions = json.loads(raw_output)
                    prediction = format_as_label(lesions)
                    prediction_json = json.dumps(lesions)
                    num_lesions = len(lesions)
                    error = None
                except json.JSONDecodeError as e:
                    prediction = ""
                    prediction_json = "[]"
                    num_lesions = 0
                    error = f"JSON parse error: {str(e)}"
            else:
                # Request failed
                prediction = ""
                prediction_json = "[]"
                num_lesions = 0
                error = result['result'].get('error', {}).get('message', 'Unknown error')
            
            # Get original data for this row
            original_row = df_original.iloc[row_idx]
            
            results.append({
                'row_index': row_idx,
                'input_text': original_row.get('ultrasound_findings', ''),
                'structured_output': original_row.get('structured_output', ''),  # Original label
                'prediction': prediction,  # Formatted [x, y, z, w] output
                'prediction_json': prediction_json,  # Raw JSON for debugging
                'num_lesions': num_lesions,
                'error': error
            })
    
    # Create DataFrame and sort by row index
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
    text_column: str = "ultrasound_findings",
    model: str = "claude-haiku-4-5-20251001",
    output_dir: str = "batch_results",
    wait: bool = True
):
    """
    Complete pipeline: Create batch, submit, wait, download, and parse results.

    Args:
        csv_path: Path to input CSV
        text_column: Column containing text to process
        model: Claude model to use
        output_dir: Directory to save outputs
        wait: Whether to wait for batch completion (default: True)

    Returns:
        Dictionary with batch_id and output paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_jsonl = output_path / f"batch_input_{timestamp}.jsonl"
    results_jsonl = output_path / f"batch_results_{timestamp}.jsonl"
    results_csv = output_path / f"batch_results_{timestamp}.csv"

    print("="*60)
    print("CLAUDE BATCH API PIPELINE")
    print("="*60)

    # Step 1: Create batch file
    num_requests = create_batch_jsonl(csv_path, str(batch_jsonl), text_column, model)

    # Step 2: Submit batch
    batch_id = submit_batch(str(batch_jsonl))

    # Save batch ID to file
    batch_id_file = output_path / f"batch_id_{timestamp}.txt"
    with open(batch_id_file, 'w') as f:
        f.write(batch_id)
    print(f"\nBatch ID saved to: {batch_id_file}")

    if not wait:
        print("\n" + "="*60)
        print("Batch submitted! Not waiting for completion.")
        print(f"To check status later, use: check_batch_status('{batch_id}')")
        print("="*60)
        return {
            'batch_id': batch_id,
            'batch_input': str(batch_jsonl),
            'batch_id_file': str(batch_id_file)
        }

    # Step 3: Wait for completion
    batch = wait_for_batch(batch_id)

    if batch.processing_status != "ended":
        print(f"\nBatch did not complete successfully: {batch.processing_status}")
        return {
            'batch_id': batch_id,
            'batch_input': str(batch_jsonl),
            'status': batch.processing_status
        }

    # Step 4: Download results
    download_batch_results(batch_id, str(results_jsonl))

    # Step 5: Parse results
    df = parse_batch_results(str(results_jsonl), str(results_csv))
    
    df = parse_batch_results_with_comparison(
        str(results_jsonl), 
        csv_path,  # Pass original CSV path
        str(results_csv)
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Batch ID: {batch_id}")
    print(f"Input file: {batch_jsonl}")
    print(f"Results JSONL: {results_jsonl}")
    print(f"Results CSV: {results_csv}")
    print("="*60)

    return {
        'batch_id': batch_id,
        'batch_input': str(batch_jsonl),
        'results_jsonl': str(results_jsonl),
        'results_csv': str(results_csv),
        'dataframe': df
    }


# =============================================================================
# MAIN - Batch processing
# =============================================================================
if __name__ == "__main__":
    result = run_batch_pipeline(
        csv_path='F:/CODE/CADBUSI/CADBUSI-Database/training/dataset/t5_data/val.csv',
        text_column='ultrasound_findings',
        model='claude-haiku-4-5-20251001', #'claude-3-haiku-20240307'
        output_dir='batch_results',
        wait=True
    )