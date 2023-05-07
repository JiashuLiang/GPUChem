cmake_minimum_required(VERSION 3.14)
project(BASIS C CXX)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(SRC
    AO.cpp
    util.cpp
    basis_set.cpp
    molecule_basis.cpp
    molecule_basis.cu
)

add_library(basis ${SRC})
install(TARGETS basis DESTINATION lib)
target_link_libraries(basis setup ${CUDA_LIBRARIES}) 