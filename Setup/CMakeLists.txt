cmake_minimum_required(VERSION 3.14)
project(SETUP C CXX)

set(SRC
    Jobinfo.cpp
    molecule.cpp
)

add_library(setup ${SRC})
install(TARGETS setup DESTINATION lib)