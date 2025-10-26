# Autonomous ROS Workspace

ROS2 workspace for autonomous vehicle development with planning, control, and perception capabilities.

## Overview

This workspace contains three main packages:

1. **Planning** - Path planning and trajectory generation
2. **Control** - Trajectory tracking and motion control
3. **Perception** - Camera processing and YOLO-based object detection

## Project Structure

```
autonomous-ros/
├── src/
│   ├── planning/
│   │   ├── scripts/
│   │   │   └── path_planner_node.py
│   │   ├── launch/
│   │   │   └── planning.launch.py
│   │   ├── config/
│   │   │   └── planning_params.yaml
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── control/
│   │   ├── scripts/
│   │   │   ├── trajectory_tracker_node.py
│   │   │   └── pid_controller_node.py
│   │   ├── launch/
│   │   │   └── control.launch.py
│   │   ├── config/
│   │   │   └── control_params.yaml
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   └── perception/
│       ├── scripts/
│       │   ├── yolo_detector_node.py
│       │   └── camera_processor_node.py
│       ├── launch/
│       │   └── perception.launch.py
│       ├── config/
│       │   └── perception_params.yaml
│       ├── models/
│       │   └── README.md
│       ├── package.xml
│       └── CMakeLists.txt
│
└── README.md
```

## Prerequisites

- ROS2 (Humble or later recommended)
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8 (for perception)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/autonomous-ros.git
cd autonomous-ros
```

### 2. Install dependencies

```bash
# Install ROS2 dependencies
sudo apt update
sudo apt install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-ackermann-msgs

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Build the workspace

```bash
cd autonomous-ros
colcon build
source install/setup.bash
```

## Usage

### Planning Package

Launch the path planner:

```bash
ros2 launch planning planning.launch.py
```

### Control Package

Launch the trajectory tracker:

```bash
ros2 launch control control.launch.py
```

### Perception Package

Launch the YOLO-based object detector:

```bash
# Make sure you have downloaded a YOLO model first
ros2 launch perception perception.launch.py model_path:=yolov8n.pt
```

### Running All Packages

You can create a master launch file or run each package in separate terminals:

```bash
# Terminal 1 - Perception
ros2 launch perception perception.launch.py

# Terminal 2 - Planning
ros2 launch planning planning.launch.py

# Terminal 3 - Control
ros2 launch control control.launch.py
```

## Package Details

### Planning

- **Path Planner Node**: Generates paths from current position to goal
- Topics:
  - Subscribes: `/goal_pose`, `/odom`
  - Publishes: `/planned_path`

### Control

- **Trajectory Tracker Node**: Tracks planned paths using Pure Pursuit algorithm
- **PID Controller Node**: Low-level velocity and steering control
- Topics:
  - Subscribes: `/planned_path`, `/odom`
  - Publishes: `/cmd_vel`

### Perception

- **YOLO Detector Node**: Detects objects in camera images using YOLOv8
- **Camera Processor Node**: Preprocesses camera images (undistortion, enhancement)
- Topics:
  - Subscribes: `/camera/image_raw`, `/camera/camera_info`
  - Publishes: `/detections`, `/detection_image`, `/camera/image_processed`

## Configuration

Each package has a `config` directory with YAML parameter files that can be modified to adjust behavior:

- `planning/config/planning_params.yaml`
- `control/config/control_params.yaml`
- `perception/config/perception_params.yaml`

## Development

### Adding New Features

1. Create new nodes in the appropriate package's `scripts/` directory
2. Update the `CMakeLists.txt` to install new scripts
3. Add launch files in the `launch/` directory
4. Update configuration files as needed

### Testing

```bash
# Build with tests
colcon build --cmake-args -DBUILD_TESTING=ON

# Run tests
colcon test
colcon test-result --verbose
```

## Troubleshooting

### YOLO Model Issues

If you encounter issues with the YOLO model:

1. Ensure ultralytics is installed: `pip install ultralytics`
2. Download the model manually and specify the full path
3. Check GPU availability if using CUDA

### ROS2 Communication Issues

- Verify all nodes are running: `ros2 node list`
- Check topic connections: `ros2 topic list` and `ros2 topic info <topic_name>`
- Monitor messages: `ros2 topic echo <topic_name>`

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue.
