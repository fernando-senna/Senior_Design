project (pupiltracker)
cmake_minimum_required(VERSION 2.8)

# configure gcc compiler flags
IF(UNIX)
    IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        IF(GCC_VERSION VERSION_LESS 4.7)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        ELSE()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        ENDIF()
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    ENDIF()
ENDIF(UNIX)

# configure boost
IF(WIN32)
    find_package(Boost 1.55.0 COMPONENTS random) 
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ELSE(WIN32)
    find_package(Boost REQUIRED)
ENDIF(WIN32)

# configure OpenCV
IF(WIN32)
    message("Windows platform detected, generating OpenCV configuration...")
    
    set(OPENCV_BUILD_DIR "C:/OpenCV/build/x64/vc12")
    set(OPENCV_INCLUDE_DIRS "C:/OpenCV/build/include;C:/OpenCV/build/include/opencv;C:/OpenCV/build/include/opencv2")
    set(OPENCV_LIB_DIRS "C:/OpenCV/build/x64/vc12/lib")

    # find all of the subdirectories in OPENCV_LIB_DIRS to include
    include_directories(${OPENCV_BUILD_DIR})
    include_directories(${OPENCV_INCLUDE_DIRS})
    link_directories(${OPENCV_LIB_DIRS})
    
    # find all of the libs in OPENCV_LIB_DIRS
    file(GLOB OpenCV_LIBS RELATIVE ${OPENCV_LIB_DIRS} ${OPENCV_LIB_DIRS}/*2410.lib)
    message("Found OpenCV libs...")
    message(${OpenCV_LIBS})
ELSE(WIN32)
    find_package(OpenCV REQUIRED)
ENDIF(WIN32)

add_executable(swirski_tracker swirski_main.cpp)
add_library(swirski_lib swirski_pupil/PupilTracker.cpp swirski_pupil/cvx.cpp swirski_pupil/utils.cpp)
target_link_libraries(swirski_tracker swirski_lib ${OpenCV_LIBS} tbb)

add_executable(canny_tracker canny_main.cpp canny_pupil/PupilTracker.cpp)
target_link_libraries(canny_tracker ${OpenCV_LIBS})
