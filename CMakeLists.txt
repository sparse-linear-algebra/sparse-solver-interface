cmake_minimum_required(VERSION 3.16)

project(
  sparse_solver_interface
  VERSION 1.0.0
  LANGUAGES C CXX
)

include(GNUInstallDirs)

add_library(sparse_solver_interface INTERFACE)

target_include_directories(
  sparse_solver_interface
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_compile_features(sparse_solver_interface INTERFACE cxx_std_20)

install(TARGETS sparse_solver_interface)

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
