cmake_minimum_required(VERSION 3.16)
project(kv260)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig)
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_search_module(gstreamer-base REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_search_module(opencv REQUIRED IMPORTED_TARGET opencv4)

add_executable(kv260-ai main.cc)
target_include_directories(kv260-ai PRIVATE

)
target_link_libraries(kv260-ai
    PkgConfig::gstreamer
    PkgConfig::gstreamer-base
    PkgConfig::opencv
    xilinxopencl
    pthread
    rt
    glog
    vitis_ai_library-dpu_task
    xir
    vart-runner
)
