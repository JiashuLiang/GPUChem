cmake_minimum_required(VERSION 3.14)
project(BASIS C CXX)

set(SRC
    AO.cpp
    util.cpp
    basis_set.cpp
    molecule_basis.cpp
)

add_library(basis ${SRC})
install(TARGETS basis DESTINATION lib)
target_link_libraries(basis setup) 