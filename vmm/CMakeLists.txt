cmake_minimum_required(VERSION 3.6)
project(ept_split_vmm C CXX)

include(${SOURCE_CMAKE_DIR}/project.cmake)
init_project()

eapis_add_vmm_executable(ept_split_vmm
    SOURCES ept_split_vcpu_factory.cpp
)
