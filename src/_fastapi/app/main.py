import logging
import io
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

# Dicom imports
import pydicom as dicom
from requests.structures import CaseInsensitiveDict
from requests_toolbelt import MultipartDecoder
from google.cloud import storage
import google.auth.transport.requests
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dicom-processor")

app = FastAPI()

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
        pool_connections=100,  # Increase connection pool size
        pool_maxsize=100,
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
            return JSONResponse(
                status_code=500, content={"success": False, "message": str(ex)}
            )

app.add_middleware(SimpleLoggingMiddleware)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def get_oauth2_token():
    """Retrieves an OAuth2 token with retry logic."""
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

async def retrieve_and_store_dicom_improved(url, bucket_name, bucket_path):
    """Improved version with better error handling and connection management"""
    study_id_from_url = url.split('/')[-1]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    headers = CaseInsensitiveDict()
    headers['Accept'] = 'multipart/related; type="application/dicom"; transfer-syntax=*'
    headers["Authorization"] = f"Bearer {get_oauth2_token()}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use streaming to handle large responses
            with http_session.get(url, headers=headers, timeout=600, stream=True) as response:
                
                if response.status_code != 200:
                    logger.error(f"Failed to retrieve DICOM: Status {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                
                content_type = response.headers.get('Content-Type', '')
                if 'multipart/related' not in content_type:
                    logger.error(f"Unsupported content type: {content_type}")
                    return False
                
                # Read response in chunks to avoid memory issues
                chunks = []
                chunk_size = 1024 * 1024  # 1MB chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        chunks.append(chunk)
                
                content = b''.join(chunks)
                decoder = MultipartDecoder(content, content_type)
                
                # Process parts with better error handling
                success_count = 0
                failed_count = 0
                
                # Use asyncio for concurrent processing instead of threads
                tasks = []
                for part in decoder.parts:
                    part_content_type = part.headers.get(b'Content-Type', b'').decode('utf-8')
                    
                    if 'application/dicom' not in part_content_type:
                        continue
                    
                    task = process_dicom_part_async(
                        part, bucket, bucket_path, study_id_from_url
                    )
                    tasks.append(task)
                
                # Process in batches to avoid overwhelming the system
                batch_size = 10
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
                
                logger.info(f"Processed {success_count} parts successfully, {failed_count} failed")
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

async def process_dicom_part_async(part, bucket, bucket_path, study_id_from_url):
    """Async version of DICOM part processing with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with io.BytesIO(part.content) as dicom_data:
                dcm = dicom.dcmread(dicom_data, force=True)
                
                series_uid = str(dcm.get('SeriesInstanceUID', 'unknown_series'))
                
                if 'SOPInstanceUID' in dcm:
                    instance_uid = str(dcm['SOPInstanceUID'].value)
                else:
                    instance_hash = hashlib.md5(part.content[:4096]).hexdigest()
                    instance_uid = f"unknown_uid_{instance_hash}"
                    logger.warning(f"No SOPInstanceUID found, using generated ID: {instance_uid}")
                
                file_path = f"{bucket_path}/{study_id_from_url}/{series_uid}/{instance_uid}.dcm"
                
                blob = bucket.blob(file_path)
                
                # Upload with retry
                for upload_attempt in range(3):
                    try:
                        blob.upload_from_string(part.content, content_type='application/dicom')
                        return True
                    except Exception as upload_error:
                        if upload_attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        raise upload_error
                
        except Exception as e:
            logger.error(f"Error processing DICOM part (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return False
    
    return False

# Update the pubsub handler to use the improved function
@app.post("/push_handlers/receive_messages")
async def pubsub_push_handlers_receive(request: Request):
    bearer_token = request.headers.get("Authorization")
    if not bearer_token:
        return JSONResponse(
            status_code=401, content={"message": "Missing Authorization header"}
        )

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
                return JSONResponse(
                    status_code=400, content={"message": "No DICOM URL in payload"}
                )
            
            # Get bucket information from environment variables
            bucket_name = os.environ.get("BUCKET_NAME", "")
            bucket_path = os.environ.get("BUCKET_PATH", "Downloads")               
            
            # Use the improved retrieval function
            success = await retrieve_and_store_dicom_improved(dicom_url, bucket_name, bucket_path)
            
            if success:
                logger.info(f"Successfully processed DICOM from {dicom_url}")
                return Response(status_code=HTTP_204_NO_CONTENT)
            else:
                logger.error(f"Failed to process DICOM from {dicom_url}")
                # Return 500 to trigger Pub/Sub retry
                return JSONResponse(
                    status_code=500, 
                    content={"message": f"Failed to process DICOM from {dicom_url}"}
                )
            
        else:
            logger.warning("Invalid Pub/Sub message format")
            return JSONResponse(
                status_code=400, 
                content={"message": "Invalid Pub/Sub message format"}
            )

    except Exception as e:
        logger.exception(f"Error processing Pub/Sub message: {e}")
        # Return 500 to trigger retry
        return JSONResponse(
            status_code=500, content={"message": "Error processing message"}
        )

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)