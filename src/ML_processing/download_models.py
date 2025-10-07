import os
import requests
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

def download_models():
    """
    Download all required models from Hugging Face.
    """
    model_dir = Path(current_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all models to download
    models = {
        "SAMUS": {
            "url": "https://huggingface.co/poofy38/SAMUS/resolve/main/SAMUS.pth",
            "filename": "SAMUS.pth"
        },
        "YOLO": {
            "url": "https://huggingface.co/poofy38/SAMUS/resolve/main/YOLO.pt",
            "filename": "yolo_lesion_detect.pt"
        },
        "Caliper": {
            "url": "https://huggingface.co/poofy38/SAMUS/resolve/main/caliper_detect_10_7_25.pt",
            "filename": "caliper_detect_10_7_25.pt"
        }
    }
    
    for model_name, model_info in models.items():
        model_path = model_dir / model_info["filename"]
        
        # Check if file already exists
        if model_path.exists():
            print(f"✓ {model_info['filename']} already exists")
            continue
        
        print(f"\nDownloading {model_info['filename']} from Hugging Face...")
        
        try:
            response = requests.get(model_info["url"], stream=True, verify=False)
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
            
            print(f"\n✓ Successfully downloaded {model_info['filename']} ({downloaded / (1024*1024):.1f} MB)")
            
        except Exception as e:
            print(f"\n✗ Error downloading {model_info['filename']}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise
    
    print("\nAll models ready!")