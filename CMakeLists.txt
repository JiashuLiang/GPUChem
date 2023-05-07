cmake_minimum_required(VERSION 3.14)
project(GPUChem LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    message(WARNING
            "The present compiler is ${CMAKE_C_COMPILER_ID}"
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()



if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}"
        CACHE PATH "Default install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# set(ARMADILLO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/armadillo-12.2.0")
find_package(Armadillo REQUIRED PATHS ${ARMADILLO_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})
find_package(BLAS)
find_package(LAPACK)
enable_testing()

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(Setup)
add_subdirectory(basis)
add_subdirectory(integral)
add_subdirectory(SCF)
add_subdirectory(tests)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set(CMAKE_CUDA_ARCHITECTURES 80)