from ultralytics import YOLO, settings
import mlflow
import torch
import os

# Function to select device (CPU or GPU)
def get_device():
    if torch.cuda.is_available():
        device = 0  # First GPU (index 0)
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"Device: cpu")
    
    return device

def evaluate_model(model_path, imgsz=640, batch_size=16):
    # Load model from .pt file
    model = YOLO(model_path)
    
    # Select device
    device = get_device()

    data_yaml_path = os.path.join(os.path.dirname(__file__), "data.yaml")
    
    # Evaluate model on test/validation set
    results = model.val(
        data=data_yaml_path,  # Data configuration file
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        split="test"  # Evaluate on test set
    )
    
    # Extract metrics from evaluation results
    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.p[0],  # Precision for the first class
        "recall": results.box.r[0],     # Recall for the first class
    }
    
    return metrics

# Disable MLflow auto-logging
mlflow.autolog(disable=True)
# Update a setting
settings.update({"mlflow": False})
