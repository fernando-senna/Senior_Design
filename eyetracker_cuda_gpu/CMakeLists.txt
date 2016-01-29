cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
#file (GLOB sources ./&.cpp ./*.h*)
# add this line instead of next if you have .hpp or .h files
file(GLOB sources ./*.cpp)
cuda_add_executable(main  ${sources})
target_link_libraries(main ${OpenCV_LIBS})
