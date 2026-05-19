add_executable(sparse_solver_interface_tests test_sparse_solver_interface.cpp)

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

add_library(ssi_test_plugin SHARED test_plugin.cpp)

target_link_libraries(
  ssi_test_plugin
  PRIVATE
    sparse_solver_interface
)

add_executable(ssi_test_plugin_loader test_plugin_loader.cpp)

target_include_directories(
  ssi_test_plugin_loader
  PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_compile_definitions(
  ssi_test_plugin_loader
  PRIVATE
    SSI_TEST_PLUGIN_PATH="$<TARGET_FILE:ssi_test_plugin>"
)

target_link_libraries(
  ssi_test_plugin_loader
  PRIVATE
    sparse_solver_interface
    ${CMAKE_DL_LIBS}
)

add_test(NAME ssi_test_plugin_loader COMMAND ssi_test_plugin_loader)
