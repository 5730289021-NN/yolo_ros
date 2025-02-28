# YOLO ROS

YOLO integration for ROS (Robot Operating System)

\![rqt_graph](docs/rqt_graph_yolov8.png)

## Features

- YOLO integration for ROS1 (converted from ROS2)
- Support for YOLOv5, YOLOv8, YOLOv9, YOLOv10, etc.
- Object detection, tracking, segmentation and pose estimation
- 3D detection
- Debug visualization
- OpenVINO acceleration support for CPU inference

## Usage

1. Clone this repository into your catkin workspace:
```bash
cd ~/catkin_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build your workspace:
```bash
cd ~/catkin_ws
catkin_make
```

4. Run YOLO:
```bash
source devel/setup.bash
roslaunch yolo_bringup yolov8.launch  # GPU inference
# or
roslaunch yolo_bringup yolo_openvino.launch  # CPU inference with OpenVINO
```

## Configuration

### Parameters

- `model_type`: YOLO model type (YOLO, World)
- `model`: YOLO model path
- `device`: Device to use (cuda:0, cpu)
- `openvino`: Whether to use OpenVINO for inference (true, false)
- `threshold`: Detection threshold
- `iou`: IoU threshold for Non-Maximum Suppression
- and more (see launch files)

### OpenVINO Acceleration

The package supports OpenVINO for optimized CPU inference:

1. Set `openvino: true` parameter
2. The node will automatically:
   - Convert your model to OpenVINO format if needed
   - Cache the converted model for future use
   - Use CPU for inference with OpenVINO optimizations

This is especially useful for deployment on devices without GPUs or when you need to reduce power consumption.

## License

This project is licensed under the GPL-3 license - see the [LICENSE](LICENSE) file for details.

## Credits

Created by Miguel Ángel González Santamarta

Converted to ROS1 from the original ROS2 package.
