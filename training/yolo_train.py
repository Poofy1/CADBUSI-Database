from ultralytics import YOLO
import torch
import os

# Debug GPU availability
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
   print("CUDA device name:", torch.cuda.get_device_name(0))
   print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

def train_yolo_model(data_yaml_path, checkpoint_path=None, run_name="run", project_dir=None):
   """
   Train YOLO optimized for B&W ultrasound images
   
   Args:
       data_yaml_path: Path to data.yaml file
       checkpoint_path: Path to checkpoint file. If None or doesn't exist, starts fresh
       run_name: Name of this run
       project_dir: Directory where to save the training outputs
   """
   
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   # Check if checkpoint exists and load accordingly
   if checkpoint_path and os.path.exists(checkpoint_path):
       print(f"Resuming from checkpoint: {checkpoint_path}")
       model = YOLO(checkpoint_path)
       resume = True
   else:
       if checkpoint_path:
           print(f"Checkpoint not found: {checkpoint_path}")
       print("Starting fresh training with yolo11s.pt")
       model = YOLO('yolo11s.pt')
       resume = False
   
   results = model.train(
       data=data_yaml_path,
       epochs=100,
       imgsz=640,
       batch=16,
       device=device,
       patience=5,
       project=project_dir,    # Specify where to save outputs
       name=run_name,
       resume=resume,
       save_period=10,
       
       # Ultrasound-appropriate augmentations
       hsv_h=0.0,        # No hue changes (B&W images)
       hsv_s=0.0,        # No saturation changes (B&W images)  
       hsv_v=0.3,        # Brightness/contrast changes (important for ultrasound)
       #degrees=10.0,    # Small rotations (probe angle variations)
       degrees=0.0,      # Use this for ultrasound cropping
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

# Usage examples
if __name__ == "__main__":
   data_yaml_path = "C:/Users/Tristan/Desktop/Yolo_ultrasound1/data.yaml"
   project_directory = "C:/Users/Tristan/Desktop/Yolo_ultrasound1"
   
   # Start fresh training
   model = train_yolo_model(
       data_yaml_path, 
       run_name="train1",
       project_dir=project_directory
   )
   
   # Continue from checkpoint
   """model = train_yolo_model(
       data_yaml_path, 
       checkpoint_path="C:/Users/Tristan/Desktop/Yolo_ultrasound1/caliper_15pct/weights/best.pt",
       run_name="caliper_30pct",
       project_dir=project_directory
   )"""