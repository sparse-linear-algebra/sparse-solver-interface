cmake_minimum_required(VERSION 3.16)

project(
  sparse_solver_interface
  VERSION 0.1.0
  LANGUAGES C CXX
)

add_library(sparse_solver_interface INTERFACE)

target_include_directories(
  sparse_solver_interface
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_compile_features(sparse_solver_interface INTERFACE cxx_std_20)

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
