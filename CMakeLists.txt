cmake_minimum_required(VERSION 3.10)
project(isaac_ros_custom_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
