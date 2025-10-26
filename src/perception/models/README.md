# YOLO Models Directory

This directory contains YOLO model files for object detection.

## Downloading YOLO Models

### YOLOv8 Models

You can download pre-trained YOLOv8 models from the Ultralytics repository:

```bash
# Install ultralytics package
pip install ultralytics

# Download models (they will be automatically downloaded on first use)
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

### Model Sizes

- **yolov8n.pt** - Nano (smallest, fastest)
- **yolov8s.pt** - Small
- **yolov8m.pt** - Medium
- **yolov8l.pt** - Large
- **yolov8x.pt** - Extra Large (largest, most accurate)

### Custom Models

Place your custom-trained YOLO models in this directory and update the `model_path` parameter in the launch file or config file.

## Usage

Update the model path in `config/perception_params.yaml`:

```yaml
yolo_detector:
  ros__parameters:
    model_path: "/path/to/your/model.pt"
```

Or pass it as a launch argument:

```bash
ros2 launch perception perception.launch.py model_path:=/path/to/your/model.pt
```
