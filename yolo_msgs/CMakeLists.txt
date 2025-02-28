cmake_minimum_required(VERSION 3.0.2)
project(yolo_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  Point2D.msg
  Vector2.msg
  Pose2D.msg
  BoundingBox2D.msg
  BoundingBox3D.msg
  Mask.msg
  KeyPoint2D.msg
  KeyPoint2DArray.msg
  KeyPoint3D.msg
  KeyPoint3DArray.msg
  Detection.msg
  DetectionArray.msg
)

add_service_files(
  FILES
  SetClasses.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    geometry_msgs
)
