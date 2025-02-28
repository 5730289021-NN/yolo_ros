cmake_minimum_required(VERSION 3.0.2)
project(yolo_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  std_srvs
  sensor_msgs
  yolo_msgs
)

catkin_package()

catkin_python_setup()

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/yolo_node.py
  scripts/tracking_node.py
  scripts/detect_3d_node.py
  scripts/debug_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
