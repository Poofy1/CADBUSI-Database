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
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/SAMUS.pth",
            "filename": "SAMUS.pth"
        },
        "YOLO": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/yolov11_lesion_detector_FP_control_2026_1_26.pt",
            "filename": "yolov11_lesion_detector_FP_control_2026_1_26.pt"
        },
        "Caliper": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_detect_10_7_25.pt",
            "filename": "caliper_detect_10_7_25.pt"
        },
        "Mask": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/mask_model.pt",
            "filename": "mask_model.pt"
        },
        "N2N": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/N2N_7.pth",
            "filename": "N2N_7.pth"
        },
        "Caliper Cropped CLF": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_cropped_clf_2_3_2026.pt",
            "filename": "caliper_pipeline_cropped_clf_2_3_2026.pt"
        },
        "Caliper Locator": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_locator_2_3_2026.pt",
            "filename": "caliper_pipeline_locator_2_3_2026.pt"
        },
        "Caliper Uncropped CLF": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/caliper_pipeline_uncropped_clf_2_3_2026.pt",
            "filename": "caliper_pipeline_uncropped_clf_2_3_2026.pt"
        },
        "US Region Cropper": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/us_region_2026_02_05.pth",
            "filename": "us_region_2026_02_05.pth"
        },
        "LOGIQE Orientation Yolo": {
            "url": "https://huggingface.co/poofy38/CADBUSI/resolve/main/LOGIQE_ori_yolo_2026_02_06.pt",
            "filename": "LOGIQE_ori_yolo_2026_02_06.pt"
        }
    }
    
    for model_name, model_info in models.items():
        model_path = model_dir / model_info["filename"]
        
        # Check if file already exists
        if model_path.exists():
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
            
            print(f"\nSuccessfully downloaded {model_info['filename']} ({downloaded / (1024*1024):.1f} MB)")
            
        except Exception as e:
            print(f"\nError downloading {model_info['filename']}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise