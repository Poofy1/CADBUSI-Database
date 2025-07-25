import os
import requests
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

def download_samus_model():
    """
    Download SAMUS.pth model from Hugging Face using direct URL.
    """
    model_dir = Path(current_dir) / "models"
    model_path = model_dir / "SAMUS.pth"
    
    # Check if file already exists
    if model_path.exists():
        print(f"SAMUS.pth already exists")
        return str(model_path)
    
    print(f"Downloading SAMUS.pth from Hugging Face...")
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the direct
    url = "https://huggingface.co/poofy38/SAMUS/resolve/main/SAMUS.pth"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\nSuccessfully downloaded SAMUS.pth ({downloaded / (1024*1024):.1f} MB)")
        return str(model_path)
        
    except Exception as e:
        print(f"Error downloading SAMUS.pth: {e}")
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise
    
    
    

def download_yolo_model():
    """
    Download YOLO model from Hugging Face using direct URL.
    """
    model_dir = Path(current_dir) / "models"
    model_path = model_dir / "yolo_lesion_detect.pt"
    
    # Check if file already exists
    if model_path.exists():
        print(f"yolo_lesion_detect.pt already exists")
        return str(model_path)
    
    print(f"Downloading YOLO model from Hugging Face...")
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the direct URL
    url = "https://huggingface.co/poofy38/SAMUS/resolve/main/YOLO.pt"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\nSuccessfully downloaded YOLO model ({downloaded / (1024*1024):.1f} MB)")
        return str(model_path)
        
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise