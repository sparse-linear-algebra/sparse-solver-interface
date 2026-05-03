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
)

add_test(NAME sparse_solver_interface_tests COMMAND sparse_solver_interface_tests)
