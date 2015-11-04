cmake_minimum_required(VERSION 2.8.3)
project(raspi_headset)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg sensor_msgs cv_bridge image_transport OpenCV)
#find_package(OpenCV REQUIRED)

## Declare ROS messages and services
#add_message_files(FILES Num.msg)
#add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

## Build the project nodes
include_directories(include ${catkin_INCLUDE_DIRS})

## Build the pupil node
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.7)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()
add_executable(pupil src/pupil/pupil_tracker.cpp)
add_library (pupil_lib src/pupil/lib/PupilTracker.cpp src/pupil/lib/cvx.cpp src/pupil/lib/utils.cpp)
target_link_libraries(pupil ${catkin_LIBRARIES} ${OpenCV_LIBS} pupil_lib tbb)
add_dependencies(pupil raspi_headset_generate_messages_cpp)

## Build the eyecam node
add_executable(eyecam src/eyecam/main.cpp)
target_link_libraries(eyecam ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(eyecam raspi_headset_generate_messages_cpp)
          
