import logging
import io
import tempfile
import os
import sys
import base64
import json
from typing import Union
from typing import Dict
from fastapi import FastAPI
from starlette.status import HTTP_204_NO_CONTENT
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

# Dicom imports
import requests
import pydicom as dicom
from requests.structures import CaseInsensitiveDict
from requests_toolbelt import MultipartDecoder
from google.cloud import storage
import google.auth.transport.requests
import google.oauth2.id_token

# Configure standard Python logging (Cloud Run captures stdout/stderr automatically)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dicom-processor")

app = FastAPI()

# Simple exception handling middleware
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

@app.get("/")
async def read_root():
    logger.info("INFO MSG")
    logger.debug("DEBUG MSG")
    logger.warning("WARNING MSG")
    logger.error("ERROR MSG")
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


# Helper function to verify JWT
def verify_jwt(token: str) -> Dict:
    """Verifies a JWT token and returns the claims."""
    auth_req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.verify_oauth2_token(token, auth_req)


def get_oauth2_token():
    """Retrieves an OAuth2 token for accessing the Google Cloud Healthcare API."""
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

async def retrieve_and_store_dicom(url, bucket_name, bucket_path):
    """Retrieves and stores all DICOM instances from a DICOMweb study."""
    # Extract study ID from the URL
    study_id_from_url = url.split('/')[-1]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    headers = CaseInsensitiveDict()
    headers['Accept'] = 'multipart/related; type="application/dicom"; transfer-syntax=*'
    headers["Authorization"] = f"Bearer {get_oauth2_token()}"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve DICOM: Status {response.status_code}")
            return False
            
        content_type = response.headers.get('Content-Type', '')
        if 'multipart/related' not in content_type:
            logger.error(f"Unsupported content type: {content_type}")
            return False
        
        # Parse the multipart response
        decoder = MultipartDecoder(response.content, content_type)
        
        # Process each part (each part is a separate DICOM instance)
        instance_count = 0
        success_count = 0
        
        for part in decoder.parts:
            part_content_type = part.headers.get(b'Content-Type', b'').decode('utf-8')
            
            if 'application/dicom' not in part_content_type:
                logger.warning(f"Skipping non-DICOM part with content type: {part_content_type}")
                continue
            
            # Process this DICOM instance
            try:
                # Read the DICOM data
                with io.BytesIO(part.content) as dicom_data:
                    dcm = dicom.dcmread(dicom_data, force=True)
                    
                    # Extract metadata for logging (not using study_uid for path anymore)
                    series_uid = str(dcm.get('SeriesInstanceUID', 'unknown_series'))
                    
                    # Get instance UID
                    if 'SOPInstanceUID' in dcm:
                        instance_uid = str(dcm['SOPInstanceUID'].value)
                    else:
                        # Generate a fallback UID
                        import hashlib
                        instance_hash = hashlib.md5(part.content[:4096]).hexdigest()
                        instance_uid = f"unknown_uid_{instance_hash}"
                        logger.warning(f"No SOPInstanceUID found in DICOM, using generated ID: {instance_uid}")
                    
                    # Use study ID from URL instead of study_uid from DICOM
                    file_path = f"{bucket_path}/{study_id_from_url}/{series_uid}/{instance_uid}.dcm"
                    
                    # Upload to GCS
                    blob = bucket.blob(file_path)
                    blob.upload_from_string(part.content, content_type='application/dicom')
                    
                    success_count += 1
            
            except Exception as e:
                logger.exception(f"Error processing DICOM instance: {e}")
            
            instance_count += 1
        
        return success_count > 0
        
    except Exception as e:
        logger.exception(f"Error retrieving or processing DICOM study: {e}")
        return False
    
    
# Modify the Pub/Sub handler
@app.post("/push_handlers/receive_messages")
async def pubsub_push_handlers_receive(request: Request):
    bearer_token = request.headers.get("Authorization")
    if not bearer_token:
        return JSONResponse(
            status_code=401, content={"message": "Missing Authorization header"}
        )

    try:
        token = bearer_token.split(" ")[1]
        claim = verify_jwt(token)
        #logger.info(f"Processing request with JWT claim ID: {claim.get('sub', 'unknown')}")

        envelope = await request.json()
        if (
            isinstance(envelope, dict)
            and "message" in envelope
            and "data" in envelope["message"]
        ):
            # Decode the Pub/Sub message data
            data = base64.b64decode(envelope["message"]["data"]).decode("utf-8")
            dicom_url = data  # Now data is directly the URL instead of a JSON payload

            if not dicom_url:
                logger.error("No DICOM URL found in payload")
                return JSONResponse(
                    status_code=400, content={"message": "No DICOM URL in payload"}
                )
            
            # Get bucket information from environment variables with defaults
            bucket_name = os.environ.get("BUCKET_NAME", "")
            bucket_path = os.environ.get("BUCKET_PATH", "Downloads")               
            
            # Retrieve and store the DICOM image
            success = await retrieve_and_store_dicom(dicom_url, bucket_name, bucket_path)
            
            if not success:
                logger.warning(f"Failed to process DICOM from {dicom_url}")
            
        else:
            logger.warning("Invalid Pub/Sub message format")

        # Return a 204 to indicate a success, even if processing failed
        # This is standard practice for Pub/Sub to prevent retries
        return Response(status_code=HTTP_204_NO_CONTENT)
    except (ValueError, IndexError) as e:
        logger.error(f"Invalid Authorization header format: {e}")
        return JSONResponse(
            status_code=401, content={"message": "Invalid Authorization header"}
        )
    except Exception as e:
        logger.exception(f"Error processing Pub/Sub message: {e}")
        return JSONResponse(
            status_code=500, content={"message": "Error processing message"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)