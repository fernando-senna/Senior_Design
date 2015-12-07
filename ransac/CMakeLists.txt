cmake_minimum_required(VERSION 2.8)
project(ransac)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
#IF( NOT EIGEN3_INCLUDE_DIR )
#    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
#ENDIF()
#INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")




set(SOURCE_FILES main.cpp)
add_executable(ransac ${SOURCE_FILES})

# target OpenCV libraries
target_link_libraries(ransac ${OpenCV_LIBS})