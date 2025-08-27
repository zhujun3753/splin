cmake_minimum_required(VERSION 2.8.3)
project(splin)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14 -O3 -lboost_system -msse2 -msse3 -pthread -Wenum-compare") # -Wall
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-w)
add_definitions("-Wall -g") 

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  4")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
if(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif()

find_package(catkin REQUIRED COMPONENTS
                    roscpp
                    std_msgs
                    geometry_msgs
                    nav_msgs
                    tf
                    cv_bridge
                    livox_ros_driver
            )

# find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

### Find OpenMP #######
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	#cmake only check for separate OpenMP library on AppleClang 7+
	#https://github.com/Kitware/CMake/blob/42212f7539040139ecec092547b7d58ef12a4d72/Modules/FindOpenMP.cmake#L252
	if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang" AND (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0"))
		SET(OpenMP_LIBS ${OpenMP_libomp_LIBRARY})
		LIST(APPEND OpenMVS_EXTRA_LIBS ${OpenMP_LIBS})
	endif()
else()
	message("-- Can't find OpenMP. Continuing without it.")
endif()


# find_package(OpenMVS)
if(OpenMVS_FOUND)
	include_directories(${OpenMVS_INCLUDE_DIRS})
	add_definitions(${OpenMVS_DEFINITIONS})
endif()


include_directories(${catkin_INCLUDE_DIRS})

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${livox_ros_driver_INCLUDE_DIRS}
  ./src
  ./src/loam/include
  ./src/tools/
)


catkin_package()

message(STATUS "===== ${PROJECT_NAME}: OpenCV library status: =====")
message(STATUS "===== OpenCV ersion: ${OpenCV_VERSION}  =====")
message(STATUS "===== OpenCV libraries: ${OpenCV_LIBS} =====")
message(STATUS "===== OpenCV include path: ${OpenCV_INCLUDE_DIRS} =====")

add_executable(poslam_LiDAR_front_end src/loam/LiDAR_front_end.cpp)
target_link_libraries(poslam_LiDAR_front_end ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(poslam_mapping src/splin.cpp 
                src/poslam_lio.cpp
                src/loam/IMU_Processing.cpp
              )
target_link_libraries(poslam_mapping 
                          ${catkin_LIBRARIES}
                          ${Boost_LIBRARIES}
                          ${Boost_FILESYSTEM_LIBRARY}
                          ${Boost_SERIALIZATION_LIBRARY} # serialization
                          ${OpenCV_LIBRARIES}
                          pcl_common 
                          pcl_io
                          )







