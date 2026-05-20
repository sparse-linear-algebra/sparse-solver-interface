#include "acutest.h"
#include "sparse_solver_interface.hpp"
#include "sparse_solver_interface_plugin.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class test_graph_t final : public ssi::graph_t{
  public:
    test_graph_t(std::shared_ptr<ssi::context_t> context,ssi::itype_t itype) :
      ssi::graph_t(std::move(context)),
      itype_(itype) {}

    ssi::itype_t itype() const override{
      return itype_;
    }

    int64_t nrows() const override{
      return 0;
    }

    int64_t ncols() const override{
      return 0;
    }

    int64_t nedges() const override{
      return 0;
    }

    void build_from_host(
      int64_t,
      int64_t,
      ssi::graph_orientation_t,
      std::function<void(ssi::graph_count_builder_t&)>&,
      std::function<void(ssi::graph_edge_builder_t&)>&) override{
      throw std::runtime_error("test graph does not build");
    }

    void borrow_compressed_graph_view(
      const ssi::compressed_graph_view_t&) override{
      throw std::runtime_error("test graph does not borrow");
    }

    std::shared_ptr<ssi::sparse_matrix_t> make_sparse_matrix() override{
      throw std::runtime_error("test graph does not make matrices");
    }

    std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override{
      throw std::runtime_error("test graph does not make symbolic analyses");
    }

  private:
    ssi::itype_t itype_;
};

class test_context_t final :
  public ssi::context_t,
  public std::enable_shared_from_this<test_context_t>{
  public:
    std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t) override{
      throw std::runtime_error("test context does not make dense matrices");
    }

    std::shared_ptr<ssi::graph_t> make_graph(ssi::itype_t itype) override{
      return std::make_shared<test_graph_t>(shared_from_this(),itype);
    }

    std::shared_ptr<ssi::sparse_problem_t> make_sparse_problem(
      const ssi::sparse_problem_properties_t& properties) override{
      return std::make_shared<ssi::default_sparse_problem_t>(
        shared_from_this(),
        properties);
    }
};

}  // namespace

static void test_sparse_problem_properties(void)
{
  ssi::sparse_problem_properties_t properties;
  properties.nrows = 5;
  properties.ncols = 5;
  properties.itype = ssi::itype_t::i32;
  properties.dtype = ssi::dtype_t::fp32;
  properties.structurally_symmetric = ssi::property_state_t::known_true;
  properties.numerically_symmetric = ssi::property_state_t::known_true;
  properties.positive_definite = ssi::property_state_t::known_true;
  properties.strong_hall = ssi::property_state_t::known_false;
  properties.symmetric_storage = ssi::symmetric_storage_t::lower;

  auto context = std::make_shared<test_context_t>();
  auto problem = context->make_sparse_problem(properties);
  TEST_CHECK(problem->properties().nrows == 5);
  TEST_CHECK(problem->properties().ncols == 5);
  TEST_CHECK(problem->properties().itype == ssi::itype_t::i32);
  TEST_CHECK(problem->properties().dtype == ssi::dtype_t::fp32);
  TEST_CHECK(
    problem->properties().structurally_symmetric ==
    ssi::property_state_t::known_true);
  TEST_CHECK(
    problem->properties().numerically_symmetric ==
    ssi::property_state_t::known_true);
  TEST_CHECK(
    problem->properties().positive_definite ==
    ssi::property_state_t::known_true);
  TEST_CHECK(problem->properties().strong_hall == ssi::property_state_t::known_false);
  TEST_CHECK(problem->properties().symmetric_storage == ssi::symmetric_storage_t::lower);

  auto graph = problem->make_graph();
  TEST_CHECK(graph->itype() == ssi::itype_t::i32);
}

static void test_c_abi_version(void)
{
  ssi_plugin_api_t api{};

  TEST_CHECK(SSI_ABI_VERSION_MAJOR == 0u);
  TEST_CHECK(SSI_ABI_VERSION_MINOR == 1u);
  TEST_CHECK(api.struct_size == 0u);
}

static void test_graph_count_builder_i32(void)
{
  std::vector<ssi::int32_t> counts(3,0);
  ssi::graph_count_builder_t builder{
    ssi::graph_orientation_t::row,
    ssi::itype_t::i32,
    5,
    7,
    1,
    4,
    {.i32 = counts.data()}
  };

  builder.set_count(1,2);
  builder.set_count(2,0);
  builder.set_count(3,4);

  TEST_CHECK(counts[0] == 2);
  TEST_CHECK(counts[1] == 0);
  TEST_CHECK(counts[2] == 4);
  TEST_EXCEPTION(builder.set_count(4,1),std::out_of_range);
  TEST_EXCEPTION(builder.set_count(1,-1),std::invalid_argument);
}

static void test_graph_edge_builder_row_oriented(void)
{
  std::vector<ssi::int64_t> offsets{0,2,2,5};
  std::vector<ssi::int64_t> ids(5,-1);
  ssi::graph_edge_builder_t builder{
    ssi::graph_orientation_t::row,
    ssi::itype_t::i64,
    5,
    7,
    1,
    4,
    {.i64 = offsets.data()},
    {.i64 = ids.data()}
  };

  TEST_CHECK(builder.degree(1) == 2);
  TEST_CHECK(builder.degree(2) == 0);
  TEST_CHECK(builder.degree(3) == 3);
  TEST_CHECK(builder.count(3) == 3);

  builder.set_edge(1,0,6);
  builder.set_edge(1,1,0);
  builder.set_edge(3,2,4);

  TEST_CHECK(ids[0] == 6);
  TEST_CHECK(ids[1] == 0);
  TEST_CHECK(ids[4] == 4);

  TEST_EXCEPTION(builder.set_edge(2,0,1),std::out_of_range);
  TEST_EXCEPTION(builder.set_edge(1,0,7),std::out_of_range);
}

static void test_graph_edge_builder_column_oriented(void)
{
  std::vector<ssi::int32_t> offsets{0,1,3};
  std::vector<ssi::int32_t> ids(3,-1);
  ssi::graph_edge_builder_t builder{
    ssi::graph_orientation_t::column,
    ssi::itype_t::i32,
    4,
    6,
    2,
    4,
    {.i32 = offsets.data()},
    {.i32 = ids.data()}
  };

  builder.set_edge(2,0,3);
  builder.set_edge(3,1,0);

  TEST_CHECK(ids[0] == 3);
  TEST_CHECK(ids[2] == 0);
  TEST_EXCEPTION(builder.set_edge(2,0,4),std::out_of_range);
}

static void test_sparse_value_builder_complex64(void)
{
  std::vector<ssi::int64_t> offsets{0,2,3};
  std::vector<ssi::int64_t> ids{1,3,0};
  std::vector<ssi::complex64_t> values(3);
  ssi::sparse_value_builder_t builder{
    ssi::graph_orientation_t::row,
    ssi::itype_t::i64,
    ssi::dtype_t::c64,
    4,
    5,
    1,
    3,
    {.i64 = offsets.data()},
    {.i64 = ids.data()},
    {.c64 = values.data()}
  };

  TEST_CHECK(builder.degree(1) == 2);
  TEST_CHECK(builder.degree(2) == 1);
  TEST_CHECK(builder.edge_id(1,1) == 3);

  builder.set_value<ssi::complex64_t>(1,0,{1.0f,2.0f});
  builder.value_mut<ssi::complex64_t>(2,0) = {3.0f,4.0f};

  TEST_CHECK(values[0] == ssi::complex64_t(1.0f,2.0f));
  TEST_CHECK(builder.value<ssi::complex64_t>(2,0) == ssi::complex64_t(3.0f,4.0f));
  TEST_EXCEPTION(builder.set_value<ssi::complex64_t>(3,0,{0.0f,0.0f}),std::out_of_range);
  TEST_EXCEPTION(builder.edge_id(1,2),std::out_of_range);
}

static void test_compressed_graph_view_i32(void)
{
  std::vector<ssi::int32_t> offsets{0,1,3};
  std::vector<ssi::int32_t> ids{3,0,2};
  ssi::compressed_graph_view_t view{
    ssi::graph_orientation_t::column,
    ssi::itype_t::i32,
    4,
    6,
    2,
    4,
    {.i32 = offsets.data()},
    {.i32 = ids.data()}
  };

  TEST_CHECK(view.extent() == 2);
  TEST_CHECK(view.offset(2) == 3);
  TEST_CHECK(view.degree(2) == 1);
  TEST_CHECK(view.degree(3) == 2);
  TEST_CHECK(view.edge_id(3,1) == 2);
  TEST_EXCEPTION(view.edge_id(3,2),std::out_of_range);
}

static void test_sparse_values_view_float32(void)
{
  std::vector<ssi::float32_t> values{1.0f,2.0f,3.0f};
  ssi::sparse_values_view_t view{
    ssi::dtype_t::fp32,
    3,
    {.fp32 = values.data()}
  };

  TEST_CHECK(view.value<ssi::float32_t>(0) == 1.0f);
  TEST_CHECK(view.value<ssi::float32_t>(2) == 3.0f);
  TEST_EXCEPTION(view.value<ssi::float32_t>(3),std::out_of_range);
}

TEST_LIST = {
  { "sparse_problem_properties", test_sparse_problem_properties },
  { "c_abi_version", test_c_abi_version },
  { "graph_count_builder_i32", test_graph_count_builder_i32 },
  { "graph_edge_builder_row_oriented", test_graph_edge_builder_row_oriented },
  { "graph_edge_builder_column_oriented", test_graph_edge_builder_column_oriented },
  { "sparse_value_builder_complex64", test_sparse_value_builder_complex64 },
  { "compressed_graph_view_i32", test_compressed_graph_view_i32 },
  { "sparse_values_view_float32", test_sparse_values_view_float32 },
  { 0, 0 }
};
