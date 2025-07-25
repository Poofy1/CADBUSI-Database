from ultralytics import YOLO
import torch

# Debug GPU availability
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

def train_yolo_model(data_yaml_path="C:/Users/Tristan/Desktop/Yolo2/data.yaml"):
    """Train YOLO11m model on ultrasound lesion detection"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-trained YOLO11m model
    model = YOLO('yolo11s.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=32,
        device=device,
        project='ultrasound_lesion_detection',
        name='yolo11m_lesions',
        patience=5,
        save=True,
        plots=True,  # Enable plotting - saves train_batch images
        save_period=1,  # Save model checkpoint every epoch
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0
    )
    
    # Validate the model
    metrics = model.val(plot=True)
    
    print(f"\nTraining completed!")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return model

# Train the model
if __name__ == "__main__":
    model = train_yolo_model()