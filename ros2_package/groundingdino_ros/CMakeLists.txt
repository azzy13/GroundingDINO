cmake_minimum_required(VERSION 3.8)
project(groundingdino_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GroundingDINOTrack.msg"
  "msg/GroundingDINOTrackArray.msg"
  DEPENDENCIES std_msgs
)

# Install Python modules (only if not already installed)
if(NOT TARGET ament_cmake_python_symlink_${PROJECT_NAME})
  ament_python_install_package(${PROJECT_NAME})
endif()

# Install Python executables
install(PROGRAMS
  groundingdino_ros/groundingdino_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME groundingdino_node
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
