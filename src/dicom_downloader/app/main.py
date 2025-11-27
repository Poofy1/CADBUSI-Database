import logging
import os
import base64
import random
import asyncio
from fastapi import FastAPI
from starlette.status import HTTP_204_NO_CONTENT
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.structures import CaseInsensitiveDict
from requests_toolbelt import MultipartDecoder
from google.cloud import storage
import google.auth.transport.requests
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import threading
from google.api_core import exceptions as gcs_exceptions
import pydicom
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dicom-processor")

app = FastAPI()

# Concurrency tracking
MAX_CONCURRENT_REQUESTS = 30
ACTIVE_REQUESTS = 0
CONCURRENCY_LOCK = threading.Lock()

# Create a session with connection pooling and retry logic
def create_robust_session():
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        allowed_methods=["GET", "POST"]
    )
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20,
        pool_block=False
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session instance
http_session = create_robust_session()

class SimpleLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as ex:
            logger.exception(f"Request failed: {ex}")
            # Always return 204 to prevent any retries
            return Response(status_code=HTTP_204_NO_CONTENT)

app.add_middleware(SimpleLoggingMiddleware)



# Global token cache with thread safety
TOKEN_CACHE = {"token": None, "expires_at": 0}
TOKEN_LOCK = threading.Lock()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def get_oauth2_token():
    """Retrieves an OAuth2 token with caching to avoid rate limits."""
    now = time.time()
    
    # Check cache first (with thread safety)
    with TOKEN_LOCK:
        if TOKEN_CACHE["token"] and now < TOKEN_CACHE["expires_at"]:
            return TOKEN_CACHE["token"]
    
    # Need to refresh - only one thread should do this
    with TOKEN_LOCK:
        # Double-check in case another thread just refreshed
        if TOKEN_CACHE["token"] and now < TOKEN_CACHE["expires_at"]:
            return TOKEN_CACHE["token"]
            
        logger.info("Refreshing OAuth2 token...")
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        
        # Cache token for 45 minutes (3600 - 900 second buffer)
        TOKEN_CACHE["token"] = creds.token
        TOKEN_CACHE["expires_at"] = now + 2700  # 45 minutes
        
        logger.info("OAuth2 token refreshed and cached")
        return creds.token

async def retrieve_and_store_dicom(url, bucket_name, bucket_path):
    """Improved version with better error handling and connection management"""
    study_id_from_url = url.split('/')[-1]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    headers = CaseInsensitiveDict()
    headers['Accept'] = 'multipart/related; type="application/dicom"; transfer-syntax=*'
    headers["Authorization"] = f"Bearer {get_oauth2_token()}"
    
    # Create shared tracking set for this study
    processed_hashes = set()
            
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Get DICOM response
            with http_session.get(url, headers=headers, timeout=600) as response:
    
                if response.status_code != 200:
                    logger.error(f"Failed to retrieve DICOM: Status {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False
                
                content_type = response.headers.get('Content-Type', '')
                if 'multipart/related' not in content_type:
                    logger.error(f"Unsupported content type: {content_type}")
                    return False
                
                # Read all content (you need it all for MultipartDecoder anyway)
                content = response.content
                
            # Process multipart content 
            decoder = MultipartDecoder(content, content_type)
            
            tasks = []
            for part in decoder.parts:
                part_content_type = part.headers.get(b'Content-Type', b'').decode('utf-8')
                
                if 'application/dicom' not in part_content_type:
                    continue
                
                task = process_dicom_part(
                    part, bucket, bucket_path, study_id_from_url, processed_hashes
                )
                tasks.append(task)
                    
            if not tasks:
                logger.warning("No DICOM parts found in response")
                return False
            
            # Process in batches to avoid overwhelming the system
            batch_size = 5
            success_count = 0
            failed_count = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process DICOM part: {result}")
                        failed_count += 1
                    elif result:
                        success_count += 1
                    else:
                        failed_count += 1
            
            if failed_count > 0:
                logger.warning(f"Processed {success_count} parts successfully, {failed_count} failed")
            return success_count > 0
                
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException
        ) as network_error:
            logger.error(f"Network error on attempt {attempt + 1} for {url}: {type(network_error).__name__}: {network_error}")
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)
                continue
            return False
            
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return False
    
    return False

async def process_dicom_part(part, bucket, bucket_path, study_id_from_url, processed_hashes):
    try:
        content_hash = hashlib.md5(part.content).hexdigest()
        
        if content_hash in processed_hashes:
            logger.info(f"Skipping duplicate content with hash: {content_hash}")
            return True
            
        processed_hashes.add(content_hash)
        
        # Read metadata only (no pixels loaded into memory)
        try:
            dcm = pydicom.dcmread(BytesIO(part.content), stop_before_pixels=True)
            patient_id = getattr(dcm, 'PatientID', 'unknown_patient').strip()
            accession_number = getattr(dcm, 'AccessionNumber', 'unknown_accession').strip()
            
            # Sanitize for filesystem (remove slashes, backslashes, nulls)
            patient_id = patient_id.replace('/', '_').replace('\\', '_').replace('\x00', '')
            accession_number = accession_number.replace('/', '_').replace('\\', '_').replace('\x00', '')
            
            # Handle empty strings
            if not patient_id:
                patient_id = 'unknown_patient'
            if not accession_number:
                accession_number = 'unknown_accession'
                
        except Exception as e:
            logger.warning(f"Could not read DICOM metadata: {e}, using fallback")
            patient_id = 'unknown_patient'
            accession_number = study_id_from_url or 'unknown_accession'
        
        # Structure: /patient_id/accession_number/hash.dcm
        file_path = f"{bucket_path}/{patient_id}_{accession_number}/{content_hash}.dcm"
        file_path = '/'.join(filter(None, file_path.split('/'))).lstrip('/')
        blob = bucket.blob(file_path)
        
        # Atomic upload-if-not-exists
        max_retries = 3
        for attempt in range(max_retries):
            try:
                blob.upload_from_string(
                    part.content, 
                    content_type='application/dicom',
                    if_generation_match=0
                )
                return True
                
            except gcs_exceptions.PreconditionFailed:
                logger.info(f"File already exists: {file_path}")
                return True
                
            except Exception as upload_error:
                logger.error(f"Upload error (attempt {attempt + 1}): {upload_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return False
                
    except Exception as e:
        logger.error(f"Error processing DICOM part: {e}")
        return False

# Add this at the top of your file
PROCESSED_URLS = set()
URL_LOCK = threading.Lock()

# Update the pubsub handler to use the improved function
@app.post("/push_handlers/receive_messages")
async def pubsub_push_handlers_receive(request: Request):
    global ACTIVE_REQUESTS
    
    # Check concurrency limit
    with CONCURRENCY_LOCK:
        if ACTIVE_REQUESTS >= MAX_CONCURRENT_REQUESTS:
            logger.warning(f"Concurrency limit reached ({ACTIVE_REQUESTS}/{MAX_CONCURRENT_REQUESTS}), returning 503")
            return Response(status_code=503)
        
        ACTIVE_REQUESTS += 1
    
    try:
        bearer_token = request.headers.get("Authorization")
        if not bearer_token:
            logger.warning("Missing Authorization header")
            return Response(status_code=HTTP_204_NO_CONTENT)  # Always 204!

        try:
            envelope = await request.json()
            if (
                isinstance(envelope, dict)
                and "message" in envelope
                and "data" in envelope["message"]
            ):
                # Decode the Pub/Sub message data
                data = base64.b64decode(envelope["message"]["data"]).decode("utf-8")
                dicom_url = data

                if not dicom_url:
                    logger.error("No DICOM URL found in payload")
                    return Response(status_code=HTTP_204_NO_CONTENT)  # Always 204!
                    
                with URL_LOCK:
                    if dicom_url in PROCESSED_URLS:
                        logger.info(f"SKIPPING DUPLICATE URL: {dicom_url}")
                        return Response(status_code=HTTP_204_NO_CONTENT)  # Always 204!
                    PROCESSED_URLS.add(dicom_url)
                
                # Get bucket information from environment variables
                bucket_name = os.environ.get("BUCKET_NAME", "")
                bucket_path = os.environ.get("BUCKET_PATH", "Downloads")               
                
                # Use the improved retrieval function
                success = await retrieve_and_store_dicom(dicom_url, bucket_name, bucket_path)
                
                if success:
                    logger.info(f"Successfully processed DICOM from {dicom_url}")
                else:
                    logger.error(f"Failed to process DICOM from {dicom_url}")
                
                # ALWAYS return 204 regardless of success/failure
                return Response(status_code=HTTP_204_NO_CONTENT)
                
            else:
                logger.warning("Invalid Pub/Sub message format")
                return Response(status_code=HTTP_204_NO_CONTENT)  # Always 204!

        except Exception as e:
            logger.exception(f"Error processing Pub/Sub message: {e}")
            # ALWAYS return 204, even on exceptions
            return Response(status_code=HTTP_204_NO_CONTENT)
    
    finally:
        # Always decrement the counter when the request completes
        with CONCURRENCY_LOCK:
            ACTIVE_REQUESTS -= 1

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)