from ultralytics import YOLO
import os
import shutil
import glob
import torch

# Select device: GPU if available, else CPU
def get_device():
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        return 0
    print("Device: CPU")
    return "cpu"

# Get the next version name for the model
def get_next_version(model_name, base_dir="runs/detect"):
    pattern = os.path.join(base_dir, "train*", "weights", f"{model_name}_v*.pt")
    models = glob.glob(pattern)
    versions = [int(f.split("_v")[1].split(".pt")[0]) for f in models]
    next_version = max(versions) + 1 if versions else 1
    return f"v{next_version}"

# Get the most recent training directory
def get_latest_train_dir(base_dir="runs/detect"):
    train_dirs = glob.glob(os.path.join(base_dir, "train*"))
    if not train_dirs:
        return None
    return max(train_dirs, key=lambda d: int(os.path.basename(d).replace("train", "") or 1))

# Callback to log losses at the end of each epoch
training_metrics = []
def on_train_epoch_end(trainer):
    m = trainer.metrics
    training_metrics.append({
        "train_box_loss": m.get("train/box_loss", 0),
        "train_cls_loss": m.get("train/cls_loss", 0),
        "train_dfl_loss": m.get("train/dfl_loss", 0),
        "val_box_loss": m.get("val/box_loss", 0),
        "val_cls_loss": m.get("val/cls_loss", 0),
        "val_dfl_loss": m.get("val/dfl_loss", 0),
    })

# Main training function
def train(pretrained_model_path=None, freeze_layers=0, epochs=1, lr0=0.01):
    model_name = "yolov8n"
    version = get_next_version(model_name)
    full_model_name = f"{model_name}_{version}"
    
    model = YOLO(pretrained_model_path or f"{model_name}.pt")
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    device = get_device()
    data_yaml = os.path.join(os.path.dirname(__file__), "data.yaml")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        optimizer="SGD",
        lr0=lr0,
        freeze=freeze_layers
    )

    # Rename best.pt to model_version.pt
    latest_dir = get_latest_train_dir()
    if latest_dir:
        src = os.path.join(latest_dir, "weights", "best.pt")
        dst = os.path.join(latest_dir, "weights", f"{full_model_name}.pt")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Model saved as: {full_model_name}")
        else:
            print("best.pt not found!")
    else:
        print("No training directory found!")

    # Return training metrics
    metrics = {
        "val_mAP50": results.box.map50,
        "val_mAP50-95": results.box.map,
        "val_precision": results.box.p[0],
        "val_recall": results.box.r[0],
    }

    print("Training complete!")
    return full_model_name, {
        "epochs": epochs, "imgsz": 640, "batch": 16,
        "device": device, "lr0": lr0, "freeze_layers": freeze_layers
    }, metrics

if __name__ == "__main__":
    train()
