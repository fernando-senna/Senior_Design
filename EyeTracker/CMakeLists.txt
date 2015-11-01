# Bhattarai, Krishna
# OCT 31, 2015
# http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html#linux-gcc-usage

# cmake_minimum_required(VERSION 3.3)    # Commented this out
cmake_minimum_required(VERSION 2.8)      # Added this version
project(EyeTracker)
find_package(OpenCV REQUIRED)            # Added this

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp)
add_executable(EyeTracker ${SOURCE_FILES} )

target_link_libraries(EyeTracker ${OpenCV_LIBS})


# Now here is how i can supply arguments and compile by myself
# Change directory to the project folder
# and do the following
# $ cmake .
# $ make

# That creates an executable with the filename specified and
# Now I just need to run it by supplying some arguments
# $ ./executable_name argument
# For example for this program do:
# $ ./EyeTracker test.jpg
