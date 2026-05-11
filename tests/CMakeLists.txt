add_executable(sparse_solver_interface_tests test_sparse_solver_interface.c)

target_include_directories(
  sparse_solver_interface_tests
  PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_link_libraries(
  sparse_solver_interface_tests
  PRIVATE
    sparse_solver_interface
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(
  sparse_solver_interface_tests
  PRIVATE
    SSI_TEST_LIBRARY_PATH="$<TARGET_FILE:sparse_solver_interface>"
)

add_test(NAME sparse_solver_interface_tests COMMAND sparse_solver_interface_tests)

add_executable(sparse_solver_interface_cpp_tests test_sparse_solver_interface_cpp.cpp)

target_include_directories(
  sparse_solver_interface_cpp_tests
  PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_link_libraries(
  sparse_solver_interface_cpp_tests
  PRIVATE
    sparse_solver_interface
)

target_compile_features(sparse_solver_interface_cpp_tests PRIVATE cxx_std_11)

add_test(NAME sparse_solver_interface_cpp_tests COMMAND sparse_solver_interface_cpp_tests)
