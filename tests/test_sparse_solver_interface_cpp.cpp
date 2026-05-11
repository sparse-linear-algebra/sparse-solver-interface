#include "acutest.h"
#include "sparse_solver_interface.hpp"

class toy_solver final : public ssi::solver {
public:
  ssi_status_t create_context(const ssi_config_t*, ssi_context_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_context(ssi_context_t**) noexcept override {}

  ssi_status_t create_matrix(ssi_context_t*, const ssi_matrix_desc_t&, ssi_matrix_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  ssi_status_t build_matrix(ssi_context_t*, ssi_matrix_t*, ssi::matrix_builder&) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  ssi_status_t read_matrix(ssi_context_t*, const ssi_matrix_t*, ssi::matrix_reader&) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_matrix(ssi_context_t*, ssi_matrix_t**) noexcept override {}

  ssi_status_t create_sparse_graph(ssi_context_t*, const ssi_sparse_graph_desc_t&, ssi_sparse_graph_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  ssi_status_t build_sparse_graph(ssi_context_t*, ssi_sparse_graph_t*, ssi::sparse_graph_builder&) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_sparse_graph(ssi_context_t*, ssi_sparse_graph_t**) noexcept override {}

  ssi_status_t create_sparse_matrix(ssi_context_t*, ssi_sparse_graph_t*, const ssi_sparse_matrix_desc_t&, ssi_sparse_matrix_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  ssi_status_t build_sparse_matrix(ssi_context_t*, ssi_sparse_matrix_t*, ssi::sparse_matrix_builder&) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_sparse_matrix(ssi_context_t*, ssi_sparse_matrix_t**) noexcept override {}

  ssi_status_t create_symbolic_factorization(ssi_context_t*, ssi_sparse_graph_t*, ssi_symbolic_factorization_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_symbolic_factorization(ssi_context_t*, ssi_symbolic_factorization_t**) noexcept override {}

  ssi_status_t create_numeric_factorization(ssi_context_t*, ssi_symbolic_factorization_t*, ssi_sparse_matrix_t*, ssi_numeric_factorization_t**) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  void destroy_numeric_factorization(ssi_context_t*, ssi_numeric_factorization_t**) noexcept override {}

  ssi_status_t solve(ssi_context_t*, const ssi_numeric_factorization_t*, const ssi_matrix_t*, ssi_matrix_t*) noexcept override { return SSI_ERROR_NOT_IMPLEMENTED; }
  ssi_status_t await(ssi_context_t*) noexcept override { return SSI_SUCCESS; }
  const char* last_error(ssi_context_t*) noexcept override { return "not implemented"; }
};

static void test_cpp_matrix_view(void)
{
  double values[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  ssi::matrix_view<double> view(values, 2, 3, 3, SSI_MATRIX_ROW_MAJOR);

  TEST_CHECK(view.rows() == 2);
  TEST_CHECK(view.columns() == 3);
  TEST_CHECK(view(1, 2) == 5.0);
  view(0, 1) = 7.0;
  TEST_CHECK(values[1] == 7.0);
}

static void test_cpp_api_view_load(void)
{
  ssi::api_view api;

  TEST_CHECK(ssi::api_view::load(ssi_get_api_v1, &api) == SSI_SUCCESS);
  TEST_CHECK(api.get().abi_version == SSI_ABI_VERSION);
}

static void test_cpp_export_api(void)
{
  ssi_api_v1_t api = {};

  TEST_CHECK(ssi::export_api<toy_solver>(&api, sizeof(api)) == SSI_SUCCESS);
  TEST_CHECK(api.abi_version == SSI_ABI_VERSION);
  TEST_CHECK(api.create_context != 0);
  TEST_CHECK(api.create_context(0, 0) == SSI_ERROR_NOT_IMPLEMENTED);
}

TEST_LIST = {
  { "cpp_matrix_view", test_cpp_matrix_view },
  { "cpp_api_view_load", test_cpp_api_view_load },
  { "cpp_export_api", test_cpp_export_api },
  { 0, 0 }
};
