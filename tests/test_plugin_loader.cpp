#include "acutest.h"
#include "sparse_solver_interface_plugin.hpp"

static void test_load_plugin_context(void)
{
  auto context = ssi::load_context_from_shared_object(SSI_TEST_PLUGIN_PATH);
  auto matrix = context->make_matrix(ssi::dtype_t::fp64);

  TEST_CHECK(matrix->dtype() == ssi::dtype_t::fp64);
  TEST_CHECK(matrix->nrows() == 0);
  TEST_CHECK(matrix->ncols() == 0);

  matrix->preallocate(4,5);

  TEST_CHECK(matrix->nrows() == 4);
  TEST_CHECK(matrix->ncols() == 5);

  ssi::sparse_problem_properties_t properties;
  properties.nrows = 5;
  properties.ncols = 4;
  properties.itype = ssi::itype_t::i32;
  properties.dtype = ssi::dtype_t::fp32;
  properties.full_column_rank = ssi::property_state_t::known_true;

  auto support = context->check_support(properties);
  TEST_CHECK(support.supported());

  auto unsupported = properties;
  unsupported.dtype = ssi::dtype_t::c128;
  auto unsupported_result = context->check_support(unsupported);
  TEST_CHECK(unsupported_result.status == ssi::status_t::unsupported);
  TEST_CHECK(unsupported_result.reason == "c128 is not supported by the test plugin");

  auto problem = context->make_sparse_problem(properties);
  TEST_CHECK(problem->properties().nrows == 5);
  TEST_CHECK(problem->properties().ncols == 4);
  TEST_CHECK(problem->properties().itype == ssi::itype_t::i32);
  TEST_CHECK(problem->properties().dtype == ssi::dtype_t::fp32);
  TEST_CHECK(
    problem->properties().full_column_rank ==
    ssi::property_state_t::known_true);

  auto sparse_matrix = problem->make_sparse_matrix();
  auto symbolic = problem->make_symbolic_analysis();
  auto factorization = symbolic->make_numeric_factorization(sparse_matrix);
  auto rhs = context->make_matrix(ssi::dtype_t::fp64);
  auto solution = context->make_matrix(ssi::dtype_t::fp64);
  auto solve_result = factorization->solve(*rhs,*solution);

  TEST_CHECK(solve_result.success());
  TEST_CHECK(solve_result.iterations == 7);
  TEST_CHECK(solve_result.refinement_steps == 2);
  TEST_CHECK(solve_result.residual_norm == 1.0e-12);
  TEST_CHECK(solve_result.relative_residual_norm == 2.0e-12);
  TEST_CHECK(solve_result.backward_error == 3.0e-12);
  TEST_CHECK(solve_result.reason == "plugin solve");
}

TEST_LIST = {
  { "load_plugin_context", test_load_plugin_context },
  { 0, 0 }
};
