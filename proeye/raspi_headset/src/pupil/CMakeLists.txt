project (pupiltracker)
cmake_minimum_required(VERSION 2.8)

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

find_package (OpenCV REQUIRED)

add_executable (pupil_tracker main.cpp)
add_library (pupil_lib lib/PupilTracker.cpp lib/cvx.cpp lib/utils.cpp)
target_link_libraries(pupil_tracker pupil_lib ${OpenCV_LIBS} tbb)
