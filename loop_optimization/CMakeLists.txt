cmake_minimum_required(VERSION 3.0.2)
project(loop_opti)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-w)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  tf
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET)
set(GTSAM_DIR "/media/zhujun/0DFD06D20DFD06D2/ws_LTAOM_orig/src/LTAOM/thirdparty/lib/cmake/GTSAM")
find_package(GTSAM REQUIRED)
message(STATUS "Found GTSAM: ${GTSAM_DIR}")
find_package(Boost COMPONENTS timer REQUIRED QUIET)


catkin_package(
  CATKIN_DEPENDS roscpp
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  src
)
if (PCL_FOUND)
  message("Hint: PCL is found")
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINATIONS})
else()
  message("Hint: PCL is not found!!")
endif()


add_executable(loop_detection_node src/loop_detection.cpp)

target_link_libraries(loop_detection_node
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  gtsam
)



