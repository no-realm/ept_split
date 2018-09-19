cmake_minimum_required(VERSION 3.6)
project(ept_monitor C CXX)

include(${SOURCE_CMAKE_DIR}/project.cmake)
init_project()

add_executable(ept_monitor
    ept_monitor.cpp
    vmcall.asm)

target_link_static_libraries(ept_monitor bfintrinsics)

install(TARGETS ept_monitor DESTINATION bin)
