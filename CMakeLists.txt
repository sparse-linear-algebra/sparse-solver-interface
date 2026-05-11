cmake_minimum_required(VERSION 3.16)

project(
  sparse_solver_interface
  VERSION 0.1.0
  LANGUAGES C CXX
)

add_library(sparse_solver_interface SHARED)

target_sources(
  sparse_solver_interface
  PRIVATE
    src/sparse_solver_interface.c
)

target_include_directories(
  sparse_solver_interface
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_compile_features(sparse_solver_interface PUBLIC c_std_99)

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
