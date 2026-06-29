cmake_minimum_required(VERSION 3.10)
project(isaac_ros_custom_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY isaac_ros_3/launch
  DESTINATION share/${PROJECT_NAME}
)

# Isaac ROS 4.4 (Jazzy) launch graphs (e.g. yolov8_foundationpose_isaac_sim.launch.py).
# Installs alongside the isaac_ros_3 launch files into share/${PROJECT_NAME}/launch.
install(DIRECTORY isaac_ros_4/launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
