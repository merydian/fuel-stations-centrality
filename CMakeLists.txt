cmake_minimum_required(VERSION 3.12)
project(fuel_stations_stats)

set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(pybind11 REQUIRED)

# Add the pybind11 module
pybind11_add_module(stats_cpp 
    stats_pybind.cpp 
    stats.cpp
    # Add other required source files here:
    # graph.cpp
    # config.cpp
    # centrality.cpp
)

# Compiler-specific options
target_compile_definitions(stats_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
