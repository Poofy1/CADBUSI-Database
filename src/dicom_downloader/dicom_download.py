#!/usr/bin/env python3
import datetime
import csv
import random
import os
import subprocess
import tqdm
import time
from google.cloud import pubsub_v1

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Cloud Build configuration
FASTAPI_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
TARGET_TAG = f"us-central1-docker.pkg.dev/{CONFIG['env']['project_id']}/{CONFIG['cloud_run']['ar']}/{CONFIG['cloud_run']['ar_name']}:{CONFIG['cloud_run']['version']}"

# The URL will be obtained after deployment
CLOUD_RUN_URL = None

# DEBUG: Set to a number to limit messages, or None/0 for no limit
DEBUG_MESSAGE_LIMIT = None

def wake_up_service():
    """Send a request to wake up the Cloud Run service before sending messages"""
    import requests
    
    global CLOUD_RUN_URL
    if not CLOUD_RUN_URL:
        return False
    
    print(f"Warming up Cloud Run service at {CLOUD_RUN_URL}...")
    try:
        # Get a token for authentication
        token_cmd = ["gcloud", "auth", "print-identity-token", 
                     f"--audiences={CLOUD_RUN_URL}"]
        token_result = subprocess.run(token_cmd, check=True, capture_output=True, text=True)
        token = token_result.stdout.strip()
        
        # Send a GET request to wake up the service
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{CLOUD_RUN_URL}", headers=headers, timeout=30)
        print(f"Warm-up request status: {response.status_code}")
        
        # Give the service time to fully initialize
        print("Waiting 10 seconds for service to be fully ready...")
        time.sleep(10)
        return True
    except Exception as e:
        print(f"Warning: Failed to warm up service: {e}")
        return False
    
def build_and_push_image():
    """Builds and pushes the Docker image using Cloud Build."""
    print("Building and pushing Docker image...")
    
    if not os.path.exists(FASTAPI_DIR):
        print(f"ERROR: FastAPI directory not found: {FASTAPI_DIR}")
        raise FileNotFoundError(f"Directory not found: {FASTAPI_DIR}")
    
    command = [
        "gcloud", "builds", "submit",
        "--gcs-source-staging-dir", CONFIG['storage']['gcs_stage'],
        "--gcs-log-dir", CONFIG['storage']['gcs_log'],
        "--tag", TARGET_TAG,
        ".",
    ]
    
    try:
        print(f"Executing in directory: {FASTAPI_DIR}")
        result = subprocess.run(command, cwd=FASTAPI_DIR, check=True, capture_output=True, text=True)
        print(f"Build successful: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e.stderr}")
        raise


def wait_for_cloud_run_ready(cr_name, max_retries=5, retry_interval=10):
    """
    Check if the Cloud Run service is available, with retries.
    
    Args:
        cr_name (str): Cloud Run service name
        max_retries (int): Maximum number of retries
        retry_interval (int): Seconds to wait between retries
    
    Returns:
        bool: True if service is available, False otherwise
    """
    print(f"Waiting for Cloud Run service '{cr_name}' to be ready...")
    
    for attempt in range(max_retries):
        try:
            check_command = [
                "gcloud", "run", "services", "describe", cr_name,
                f"--region={CONFIG['env']['region']}", f"--project={CONFIG['env']['project_id']}",
                "--format=value(status.url)"
            ]
            check_result = subprocess.run(check_command, check=True, capture_output=True, text=True)
            url = check_result.stdout.strip()
            
            if url:
                print(f"Cloud Run service '{cr_name}' is ready at {url}")
                return url
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1}/{max_retries}: Service not ready yet: {e.stderr}")
        
        print(f"Waiting {retry_interval} seconds before retrying...")
        time.sleep(retry_interval)
    
    print(f"Cloud Run service '{cr_name}' not ready after {max_retries} attempts")
    return None


def deploy_cloud_run(bucket_name=None, bucket_path=None):
    """Deploy the FastAPI application to Cloud Run."""
    global CLOUD_RUN_URL
    
    print("Deploying Cloud Run service...")
    
    cr_name = CONFIG['cloud_run']['ar_name'].replace("_", "-")
    vpc_connector = f"projects/{CONFIG['cloud_run']['vpc_shared']}/locations/{CONFIG['env']['region']}/connectors/{CONFIG['cloud_run']['vpc_name']}"
    
    # Create environment variables string
    env_vars = f"BUCKET_NAME={bucket_name},BUCKET_PATH={bucket_path}"
    
    command = [
        "gcloud", "run", "deploy", cr_name,
        "--binary-authorization=default",
        f"--image={TARGET_TAG}",
        "--ingress=internal-and-cloud-load-balancing",
        "--no-allow-unauthenticated",
        "--port=5000",
        f"--project={CONFIG['env']['project_id']}",
        "--quiet",
        f"--region={CONFIG['env']['region']}",
        f"--service-account={CONFIG['env']['service_account_identity']}",
        f"--vpc-connector={vpc_connector}",
        "--vpc-egress=all-traffic",
        "--timeout=3000",
        "--cpu=2",
        "--memory=8192Mi",
        "--concurrency=35",
        "--max-instances=500",
        "--min-instances=3", 
        f"--set-env-vars={env_vars}"
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully deployed Cloud Run service: {cr_name}")
        
        # Wait for service to be fully available and get URL
        url = wait_for_cloud_run_ready(cr_name)
        if url:
            CLOUD_RUN_URL = url
            print(f"Cloud Run URL: {CLOUD_RUN_URL}")
            return CLOUD_RUN_URL
        else:
            raise Exception(f"Failed to get URL for Cloud Run service: {cr_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error deploying Cloud Run service: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error accessing Cloud Run service: {str(e)}")
        raise


def check_pubsub_exists():
    """Check if Pub/Sub topic and subscription already exist."""
    topic_exists = False
    subscription_exists = False
    
    # Check if topic exists
    try:
        check_topic_cmd = [
            "gcloud", "pubsub", "topics", "describe", CONFIG['env']['topic_name'],
            f"--project={CONFIG['env']['project_id']}", "--format=json"
        ]
        subprocess.run(check_topic_cmd, check=True, capture_output=True, text=True)
        topic_exists = True
        print(f"Topic '{CONFIG['env']['topic_name']}' already exists.")
    except subprocess.CalledProcessError:
        pass
    
    # Check if subscription exists
    try:
        check_sub_cmd = [
            "gcloud", "pubsub", "subscriptions", "describe", CONFIG['env']['subscription_name'],
            f"--project={CONFIG['env']['project_id']}", "--format=json"
        ]
        subprocess.run(check_sub_cmd, check=True, capture_output=True, text=True)
        subscription_exists = True
        print(f"Subscription '{CONFIG['env']['subscription_name']}' already exists.")
    except subprocess.CalledProcessError:
        pass
    
    return topic_exists, subscription_exists


def setup_pubsub():
    """Create Pub/Sub topic and subscription with push configuration."""
    global CLOUD_RUN_URL
    
    if CLOUD_RUN_URL is None:
        print("ERROR: Cloud Run URL is not available. Deploy Cloud Run first.")
        return False
    
    push_endpoint = f"{CLOUD_RUN_URL}/push_handlers/receive_messages"
    print(f"Push endpoint: {push_endpoint}")
    
    topic_exists, subscription_exists = check_pubsub_exists()
    
    if not topic_exists:
        print(f"Setting up Pub/Sub topic: {CONFIG['env']['topic_name']}")
        try:
            subprocess.run([
                "gcloud", "pubsub", "topics", "create", CONFIG['env']['topic_name'],
                f"--project={CONFIG['env']['project_id']}"
            ], check=True)
            print(f"Topic '{CONFIG['env']['topic_name']}' created successfully.")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                print(f"Topic '{CONFIG['env']['topic_name']}' already exists.")
            else:
                raise
    
    if not subscription_exists:
        print(f"Setting up Pub/Sub subscription: {CONFIG['env']['subscription_name']}")
        try:
            # Create dead letter topic first
            dead_letter_topic = f"{CONFIG['env']['topic_name']}-failed"
            subprocess.run([
                "gcloud", "pubsub", "topics", "create", dead_letter_topic,
                f"--project={CONFIG['env']['project_id']}"
            ], check=True)
            
            # Simple subscription without delivery limits
            subprocess.run([
                "gcloud", "pubsub", "subscriptions", "create", CONFIG['env']['subscription_name'],
                f"--topic={CONFIG['env']['topic_name']}",
                f"--push-endpoint={push_endpoint}",
                f"--push-auth-service-account={CONFIG['env']['service_account_identity']}",
                f"--project={CONFIG['env']['project_id']}",
                "--ack-deadline=600",
                "--max-delivery-attempts=5",
                f"--dead-letter-topic=projects/{CONFIG['env']['project_id']}/topics/{dead_letter_topic}",
            ], check=True)
            print(f"Subscription '{CONFIG['env']['subscription_name']}' created successfully.")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                print(f"Subscription '{CONFIG['env']['subscription_name']}' already exists.")
            else:
                raise
    
    return True


def publish_message(url):
    global PUBLISHER
    global TOPIC_PATH
    
    future = PUBLISHER.publish(TOPIC_PATH, data=url.encode("utf-8"))
    return future

from google.cloud import storage

def get_immediate_folders(bucket_name, folder_prefix):
    """
    Get the names of immediate subdirectories in a GCP bucket folder.
    """
    # Initialize the storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Normalize the folder_prefix to prevent double slashes
    folder_prefix = folder_prefix.strip('/')
    if folder_prefix:
        folder_prefix += '/'
    
    # Get all objects with the prefix
    blobs = bucket.list_blobs(prefix=folder_prefix, delimiter='/')
    
    # Get immediate folder names
    immediate_folders = set()
    
    # Consume the iterator to get prefixes
    list(blobs)  # This processes the blobs and populates prefixes
    
    # Extract folder names from prefixes
    if hasattr(blobs, 'prefixes'):
        for prefix in blobs.prefixes:
            # Extract folder name from path like "folder_prefix/FOLDER_NAME/"
            folder_name = prefix.rstrip('/').split('/')[-1]
            if folder_name:  # Make sure it's not empty
                immediate_folders.add(folder_name)
    
    return immediate_folders


def process_csv_file(csv_file, bucket_name=None, bucket_path=None):
    """
    Read the CSV file containing DICOM URLs and publish each URL to Pub/Sub.
    """
    global DEBUG_MESSAGE_LIMIT
    
    print(f"Processing CSV file: {csv_file}")
    
    # Get existing studies in one batch operation if bucket_path is provided
    existing_studies = set()
    if bucket_path and bucket_name:
        print("Checking for existing studies...")
        existing_studies = get_immediate_folders(bucket_name, bucket_path)
        print(f"Found {len(existing_studies)} existing studies to skip")
    
    # Read all rows into memory first
    all_rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('ENDPOINT_ADDRESS')
            study_id = row.get('STUDY_ID')
            
            # Skip rows we don't want to process
            if study_id in existing_studies or not url:
                continue
                
            all_rows.append((url, study_id, row))
    
    # Shuffle the rows to randomize processing order
    random.shuffle(all_rows)
    print(f"Shuffled {len(all_rows)} rows for processing")
    
    # Apply debug limit if set
    total_rows = len(all_rows)
    if DEBUG_MESSAGE_LIMIT and DEBUG_MESSAGE_LIMIT > 0:
        total_rows = min(total_rows, DEBUG_MESSAGE_LIMIT)
        all_rows = all_rows[:DEBUG_MESSAGE_LIMIT]
        print(f"DEBUG: Limiting processing to {DEBUG_MESSAGE_LIMIT} messages")
    
    num_skipped = len(existing_studies) if bucket_path and bucket_name else 0
    num_published = 0 
    
    # Process the shuffled rows
    pbar = tqdm.tqdm(total=total_rows, desc="Publishing messages")
    
    for url, study_id, row in all_rows:
        publish_message(url)
        num_published += 1 
        pbar.update(1)
        
    pbar.close()
    
    if num_skipped > 0:
        print(f"Skipped {num_skipped} existing studies")
    
    print(f"Published {num_published} new messages from {csv_file}")
    print(f"Wait for bucket storage to fill up to {num_published} new folders, then cleanup with: python main.py --cleanup")


def cleanup_resources():
    print("Cleaning up resources...")
    
    # Delete subscription
    try:
        subprocess.run([
            "gcloud", "pubsub", "subscriptions", "delete", CONFIG['env']['subscription_name'],
            f"--project={CONFIG['env']['project_id']}", "--quiet"
        ], check=True, capture_output=True, text=True)
        print(f"Deleted subscription '{CONFIG['env']['subscription_name']}'")
    except subprocess.CalledProcessError as e:
        if "NOT_FOUND" in e.stderr or "not found" in e.stderr.lower():
            print(f"  Subscription '{CONFIG['env']['subscription_name']}' not found (already deleted or never created)")
        else:
            print(f"Failed to delete subscription: {e.stderr.strip()}")
    
    # Delete topic
    try:
        subprocess.run([
            "gcloud", "pubsub", "topics", "delete", CONFIG['env']['topic_name'],
            f"--project={CONFIG['env']['project_id']}", "--quiet"
        ], check=True, capture_output=True, text=True)
        print(f"Deleted topic '{CONFIG['env']['topic_name']}'")
    except subprocess.CalledProcessError as e:
        if "NOT_FOUND" in e.stderr or "not found" in e.stderr.lower():
            print(f"  Topic '{CONFIG['env']['topic_name']}' not found (already deleted or never created)")
        else:
            print(f"Failed to delete topic: {e.stderr.strip()}")
    
    # Delete dead-letter topic
    try:
        dead_letter_topic = f"{CONFIG['env']['topic_name']}-failed"
        subprocess.run([
            "gcloud", "pubsub", "topics", "delete", dead_letter_topic,
            f"--project={CONFIG['env']['project_id']}", "--quiet"
        ], check=True, capture_output=True, text=True)
        print(f"Deleted dead-letter topic '{dead_letter_topic}'")
    except subprocess.CalledProcessError as e:
        if "NOT_FOUND" in e.stderr or "not found" in e.stderr.lower():
            print(f"  Dead-letter topic '{dead_letter_topic}' not found (already deleted or never created)")
        else:
            print(f"Failed to delete dead-letter topic: {e.stderr.strip()}")
    
    # Delete Cloud Run service
    try:
        cr_name = CONFIG['cloud_run']['ar_name'].replace("_", "-")
        subprocess.run([
            "gcloud", "run", "services", "delete", cr_name,
            f"--region={CONFIG['env']['region']}", 
            f"--project={CONFIG['env']['project_id']}", "--quiet"
        ], check=True, capture_output=True, text=True)
        print(f"Deleted Cloud Run service '{cr_name}'")
    except subprocess.CalledProcessError as e:
        if "could not be found" in e.stderr.lower() or "not found" in e.stderr.lower():
            print(f"  Cloud Run service '{cr_name}' not found (already deleted or never created)")
        else:
            print(f"Failed to delete Cloud Run service: {e.stderr.strip()}")
    
    # Also delete the container image
    try:
        registry = f"us-central1-docker.pkg.dev/{CONFIG['env']['project_id']}/{CONFIG['cloud_run']['ar']}/{CONFIG['cloud_run']['ar_name']}"
        subprocess.run([
            "gcloud", "artifacts", "docker", "images", "delete", 
            registry, "--quiet", "--delete-tags"
        ], check=True, capture_output=True, text=True)
        print(f"Deleted container image '{CONFIG['cloud_run']['ar_name']}'")
    except subprocess.CalledProcessError as e:
        if "NOT_FOUND" in e.stderr or "not found" in e.stderr.lower():
            print(f"  Container image not found (already deleted or never created)")
        else:
            print(f"Failed to delete container image: {e.stderr.strip()}")
    
    print("\nCleanup complete")


def get_existing_cloud_run_url():
    """Get the URL of an existing Cloud Run service."""
    cr_name = CONFIG['cloud_run']['ar_name'].replace("_", "-")
    try:
        command = [
            "gcloud", "run", "services", "describe", cr_name,
            f"--region={CONFIG['env']['region']}", f"--project={CONFIG['env']['project_id']}",
            "--format=value(status.url)"
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        url = result.stdout.strip()
        if url:
            return url
        return None
    except subprocess.CalledProcessError:
        return None


def dicom_download_remote_start(csv_file=None, deploy=False, cleanup=False):
    global CLOUD_RUN_URL
    global PUBLISHER
    global TOPIC_PATH
    
    # Check if certificate file exists
    if not os.path.exists(os.path.join(FASTAPI_DIR, "CertEmulationCA.crt")):
        print(f"Error: Certificate file not found at {FASTAPI_DIR}")
        print("Please ensure the certificate file exists before running this function.")
        sys.exit(1)  # Exit with error code
        
        
    PUBLISHER = pubsub_v1.PublisherClient()
    TOPIC_PATH = PUBLISHER.topic_path(CONFIG['env']['project_id'], CONFIG['env']['topic_name'])
    
    # Generate a timestamp-based path if not provided
    bucket_path = f"{CONFIG['storage']['download_path']}/"
    
    # Handle cleanup first - this can be run without other flags
    if cleanup:
        cleanup_resources()
        return 0
    
    # For other operations, we need a CSV file
    if not csv_file and (deploy):
        print("Error: CSV file is required for deployment or setup operations")
        return 1
    
    if deploy:
        # Build and push the image first, then deploy to Cloud Run
        build_and_push_image()
        deploy_cloud_run(bucket_name=CONFIG['storage']['bucket_name'], bucket_path=bucket_path)
    elif CLOUD_RUN_URL is None:
        # Try to get the URL of an existing deployment first
        existing_url = get_existing_cloud_run_url()
        if existing_url:
            CLOUD_RUN_URL = existing_url
            print(f"Using existing Cloud Run URL: {CLOUD_RUN_URL}")
        else:
            # Fall back to hardcoded URL format if service exists but we can't get the URL
            CLOUD_RUN_URL = f"https://{CONFIG['cloud_run']['service']}-243026470979.{CONFIG['env']['region']}.run.app"
            print(f"Using fallback Cloud Run URL: {CLOUD_RUN_URL}")
    
    if deploy:
        # Add a brief pause before setting up PubSub to ensure Cloud Run is ready
        time.sleep(5)
        setup_pubsub()
    
    # Wake up the service before processing messages
    if csv_file and os.path.exists(csv_file):
        # Make sure service is warmed up before sending messages
        wake_up_service()
        
        # Now process the CSV file
        process_csv_file(csv_file, bucket_name=CONFIG['storage']['bucket_name'], bucket_path=bucket_path)
        
        # Wait a bit to allow processing to complete
        print("Waiting 20 seconds for message processing to complete...")
        time.sleep(20)
    elif csv_file:
        print(f"Error: CSV file not found: {csv_file}")
        return 1
    
    return 0