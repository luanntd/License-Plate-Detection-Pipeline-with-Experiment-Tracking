## 📦 Dataset Configuration

Organize your dataset to match the structure expected by YOLOv8. The standard directory layout should look like this:

```bash
dataset/
├── train/
│   ├── images/       # Training images (e.g., .jpg, .png)
│   └── labels/       # YOLO-format labels for training
├── valid/
│   ├── images/       # Validation images
│   └── labels/       # Corresponding validation labels
├── test/
│   ├── images/       # Test images
│   └── labels/       # Corresponding test labels
```

Or, you can configure the paths manually in the `data.yaml` file to match your custom dataset structure. Make sure to follow these rules to ensure YOLOv8 loads the data correctly:
- The `train`, `val`, and `test` paths should point to the directories containing image files (e.g., .jpg, .png).
- The corresponding `labels/` directories must be at the same level as `images/`.
- Labels should follow the YOLO format: one .txt file per image with bounding box annotations (class index, x_center, y_center, width, height – all normalized).