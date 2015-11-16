
# Set the minimum version for cmake 
cmake_minimum_required(VERSION 2.8)
# Set the name of the project
project(trial)

# Find the OpenCV Package
find_package(OpenCV REQUIRED)

# Set Flags for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Set the source files
set(SOURCE_FILES main.cpp)

# Set the executable
add_executable(trial ${SOURCE_FILES})

# target OpenCV libraries
target_link_libraries(trial ${OpenCV_LIBS})

# If all else fails just run this command in the command line
# g++ main.cpp -o main `pkg-config --cflags --libs opencv`
