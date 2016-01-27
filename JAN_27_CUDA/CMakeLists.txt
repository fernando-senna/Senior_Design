cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
# I don't know what the flags are but I am going to use them
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
file (GLOB cu *.cu)
cuda_add_executable(main main.cpp ${cu})
target_link_libraries(main ${OpenCV_LIBS})