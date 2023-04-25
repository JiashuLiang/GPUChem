cmake_minimum_required(VERSION 3.14)
project(SCF C CXX)

set(SRC
    SCF.cpp
)

add_library(scf ${SRC})
install(TARGETS scf DESTINATION lib)
target_link_libraries(scf basis setup integral)