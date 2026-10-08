cmake_minimum_required(VERSION 3.10)
project(isaac_ros_custom_bringup)

find_package(ament_cmake REQUIRED)

if("$ENV{ROS_DISTRO}" STREQUAL "lyrical")
  # Isaac ROS 5.0 (Lyrical): install only the 5.0 graphs. The 3.x/4.x launches use
  # NITROS packages that 5.0 removed, and the 4.x file has the same name as the 5.0 one.
  install(DIRECTORY isaac_ros_5/launch
    DESTINATION share/${PROJECT_NAME}
  )
else()
  install(DIRECTORY isaac_ros_3/launch
    DESTINATION share/${PROJECT_NAME}
  )

  # Isaac ROS 4.4 (Jazzy) launch graphs (e.g. yolov8_foundationpose_isaac_sim.launch.py).
  # Installs alongside the isaac_ros_3 launch files into share/${PROJECT_NAME}/launch.
  install(DIRECTORY isaac_ros_4/launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
