#\!/bin/bash

# YOLO ROS Setup Script

echo "Setting up YOLO ROS..."

# Create scripts directory if it doesn't exist
mkdir -p /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/scripts

# Copy Python nodes to scripts directory
echo "Copying Python nodes to scripts directory..."
cp /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/yolo_ros/*.py /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/scripts/

# Make scripts executable
echo "Making scripts executable..."
chmod +x /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/scripts/*.py

# Create src directory and __init__.py file
mkdir -p /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/src/yolo_ros
touch /home/fibo/catkin_ws/src/yolo_ros/yolo_ros/src/yolo_ros/__init__.py

echo "Setup complete\! Now run:"
echo "  cd ~/catkin_ws"
echo "  catkin_make"
echo "  source devel/setup.bash"
echo "  roslaunch yolo_bringup yolov8.launch"
