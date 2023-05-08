cmake_minimum_required(VERSION 3.14)
project(Integral C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

set(SRC
    hcore.cpp
    JKmat.cpp
    HartreeFock_Rys.cpp
    # hcore.cu
    JKmat.cu
    HartreeFock_Rys_gpu.cu
)

add_library(integral ${SRC})
install(TARGETS integral DESTINATION lib)
target_link_libraries(integral basis setup  ${CUDA_LIBRARIES})
