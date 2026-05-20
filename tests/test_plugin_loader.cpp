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
  properties.nrows = 4;
  properties.ncols = 5;
  properties.itype = ssi::itype_t::i32;
  properties.dtype = ssi::dtype_t::fp32;
  properties.full_column_rank = ssi::property_state_t::known_true;

  auto problem = context->make_sparse_problem(properties);
  TEST_CHECK(problem->properties().nrows == 4);
  TEST_CHECK(problem->properties().ncols == 5);
  TEST_CHECK(problem->properties().itype == ssi::itype_t::i32);
  TEST_CHECK(problem->properties().dtype == ssi::dtype_t::fp32);
  TEST_CHECK(
    problem->properties().full_column_rank ==
    ssi::property_state_t::known_true);
}

TEST_LIST = {
  { "load_plugin_context", test_load_plugin_context },
  { 0, 0 }
};
