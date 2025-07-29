from ultralytics import YOLO
import torch

# Debug GPU availability
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

def train_yolo_model(data_yaml_path):
    """Train YOLO optimized for B&W ultrasound images"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolo11s.pt')  # Start with smaller model
    
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=5,
        
        # Ultrasound-appropriate augmentations
        hsv_h=0.0,        # No hue changes (B&W images)
        hsv_s=0.0,        # No saturation changes (B&W images)  
        hsv_v=0.3,        # Brightness/contrast changes (important for ultrasound)
        degrees=10.0,     # Small rotations (probe angle variations)
        translate=0.1,    # Translations (probe positioning)
        scale=0.3,        # Scaling (zoom variations)
        shear=0.0,        # No shearing
        perspective=0.0,  # No perspective changes
        flipud=0.0,       # No vertical flips (ultrasound orientation matters)
        fliplr=0.5,       # Horizontal flips (left/right lesions)
        mosaic=0.3,       # Reduced mosaic
        mixup=0.0,        # No mixup for medical data
        copy_paste=0.0,   # No copy-paste
        auto_augment='',  # Disable auto augment (designed for color images)
        erasing=0.1,      # Light random erasing
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Conservative learning rate for medical data
        lr0=0.0005,
        lrf=0.01,
    )
    
    # Validate the model
    metrics = model.val(plot=True)
    
    print(f"\nTraining completed!")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return model

# Train the model
if __name__ == "__main__":
    model = train_yolo_model(data_yaml_path="C:/Users/Tristan/Desktop/Yolo2/data.yaml")