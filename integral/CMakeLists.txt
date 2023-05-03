cmake_minimum_required(VERSION 3.14)
project(Integral C CXX)

set(SRC
    hcore.cpp
    JKmat.cpp
    HartreeFock_Rys.cpp
)

add_library(integral ${SRC})
install(TARGETS integral DESTINATION lib)
target_link_libraries(integral basis setup)