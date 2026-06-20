#include "acutest.h"
#include "sparse_solver_interface.hpp"
#include "sparse_solver_interface_plugin.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class test_matrix_t final : public ssi::matrix_t{
  public:
    test_matrix_t(std::shared_ptr<ssi::context_t> context,ssi::dtype_t dtype) :
      ssi::matrix_t(std::move(context)),
      dtype_(dtype) {}

    int64_t nrows() const override{
      return nrows_;
    }

    int64_t ncols() const override{
      return ncols_;
    }

    ssi::dtype_t dtype() const override{
      return dtype_;
    }

    void preallocate(int64_t nrows,int64_t ncols) override{
      nrows_ = nrows;
      ncols_ = ncols;
    }

    void borrow_matrix_view(
      const ssi::placement_t&,
      const ssi::matrix_view_t& view) override{
      nrows_ = view.rend - view.rbeg;
      ncols_ = view.cend - view.cbeg;
      dtype_ = view.dtype;
    }

    void build_from_host(std::function<void(ssi::matrix_view_t&)>&) override{
      throw std::runtime_error("test matrix does not build");
    }

    void build_from_placement(
      std::function<void(const ssi::placement_t&,ssi::matrix_view_t&)>&) override{
      throw std::runtime_error("test matrix does not build from placement");
    }

    void read_to_host(std::function<void(const ssi::matrix_view_t&)>&) const override{
      throw std::runtime_error("test matrix does not read");
    }

    void read_to_placement(
      const ssi::placement_t&,
      std::function<void(const ssi::matrix_view_t&)>&) const override{
      throw std::runtime_error("test matrix does not read to placement");
    }

  private:
    int64_t nrows_ = 0;
    int64_t ncols_ = 0;
    ssi::dtype_t dtype_;
};

class test_sparse_matrix_t final : public ssi::sparse_matrix_t{
  public:
    test_sparse_matrix_t(std::shared_ptr<ssi::graph_t> graph,ssi::dtype_t dtype) :
      ssi::sparse_matrix_t(std::move(graph)),
      dtype_(dtype) {}

    int64_t nrows() const override{
      return graph().nrows();
    }

    int64_t ncols() const override{
      return graph().ncols();
    }

    ssi::dtype_t dtype() const override{
      return dtype_;
    }

    void build_from_host(
      ssi::dtype_t dtype,
      ssi::graph_orientation_t,
      std::function<void(ssi::sparse_value_builder_t&)>&) override{
      dtype_ = dtype;
    }

    void read_to_host(
      ssi::graph_orientation_t,
      std::function<void(const ssi::sparse_value_builder_t&)>&) const override{
      throw std::runtime_error("test sparse matrix does not read");
    }

    void borrow_sparse_values_view(const ssi::sparse_values_view_t& view) override{
      dtype_ = view.dtype;
    }

  private:
    ssi::dtype_t dtype_;
};

class test_numeric_factorization_t final : public ssi::numeric_factorization_t{
  public:
    test_numeric_factorization_t(
      std::shared_ptr<ssi::symbolic_t> symbolic,
      std::shared_ptr<ssi::sparse_matrix_t> matrix) :
      ssi::numeric_factorization_t(std::move(symbolic),std::move(matrix)) {}

    ssi::dtype_t dtype() const override{
      return matrix().dtype();
    }

    ssi::solve_result_t solve(const ssi::matrix_t&,ssi::matrix_t&) const override{
      return {
        ssi::status_t::ok,
        true,
        0,
        1,
        0.0,
        0.0,
        0.0,
        "direct solve"
      };
    }
};

class test_symbolic_t final :
  public ssi::symbolic_t,
  public std::enable_shared_from_this<test_symbolic_t>{
  public:
    explicit test_symbolic_t(std::shared_ptr<ssi::graph_t> graph) :
      ssi::symbolic_t(std::move(graph)) {}

    std::shared_ptr<ssi::numeric_factorization_t>
    make_numeric_factorization(std::shared_ptr<ssi::sparse_matrix_t> matrix) override{
      return std::make_shared<test_numeric_factorization_t>(
        std::static_pointer_cast<ssi::symbolic_t>(shared_from_this()),
        std::move(matrix));
    }
};

class test_graph_t final :
  public ssi::graph_t,
  public std::enable_shared_from_this<test_graph_t>{
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
      return std::make_shared<test_sparse_matrix_t>(
        std::static_pointer_cast<ssi::graph_t>(shared_from_this()),
        ssi::dtype_t::fp64);
    }

    std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override{
      return std::make_shared<test_symbolic_t>(
        std::static_pointer_cast<ssi::graph_t>(shared_from_this()));
    }

  private:
    ssi::itype_t itype_;
};

class test_context_t final :
  public ssi::context_t,
  public std::enable_shared_from_this<test_context_t>{
  public:
    std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t dtype) override{
      return std::make_shared<test_matrix_t>(shared_from_this(),dtype);
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

  TEST_CHECK(SSI_ABI_VERSION_MAJOR == 1u);
  TEST_CHECK(SSI_ABI_VERSION_MINOR == 0u);
  TEST_CHECK(api.struct_size == 0u);
}

static void test_status_exception_mapping(void)
{
  ssi_plugin_api_t api{};
  api.last_error_message = ssi::plugin::detail::last_error_message;

  auto unsupported_status = ssi::plugin::detail::guard([]{
    throw ssi::unsupported_error_t("unsupported dtype");
  });
  TEST_CHECK(unsupported_status == SSI_STATUS_UNSUPPORTED);
  TEST_EXCEPTION(
    ssi::plugin::detail::check_status(api,unsupported_status),
    ssi::unsupported_error_t);

  auto singular_status = ssi::plugin::detail::guard([]{
    throw ssi::singular_error_t("zero diagonal pivot");
  });
  TEST_CHECK(singular_status == SSI_STATUS_SINGULAR);
  TEST_EXCEPTION(
    ssi::plugin::detail::check_status(api,singular_status),
    ssi::singular_error_t);
}

static void test_sparse_problem_property_consistency(void)
{
  ssi::sparse_problem_properties_t properties;
  properties.nrows = 5;
  properties.ncols = 5;
  properties.itype = ssi::itype_t::i64;
  properties.dtype = ssi::dtype_t::fp64;
  properties.nonsingular = ssi::property_state_t::known_true;

  auto context = std::make_shared<test_context_t>();
  auto problem = context->make_sparse_problem(properties);

  auto refined = properties;
  refined.strong_hall = ssi::property_state_t::known_true;
  problem->assert_properties(refined);
  TEST_CHECK(problem->properties().strong_hall == ssi::property_state_t::known_true);

  auto contradicted = problem->properties();
  contradicted.nonsingular = ssi::property_state_t::known_false;
  TEST_EXCEPTION(problem->assert_properties(contradicted),std::invalid_argument);

  (void)problem->make_graph();
  auto changed_dtype = problem->properties();
  changed_dtype.dtype = ssi::dtype_t::fp32;
  TEST_EXCEPTION(problem->assert_properties(changed_dtype),std::invalid_argument);

  auto impossible = properties;
  impossible.nrows = 3;
  impossible.ncols = 4;
  impossible.nonsingular = ssi::property_state_t::known_true;
  TEST_EXCEPTION(context->make_sparse_problem(impossible),std::invalid_argument);
}

static void test_default_support_query(void)
{
  auto context = std::make_shared<test_context_t>();

  ssi::sparse_problem_properties_t properties;
  properties.nrows = 4;
  properties.ncols = 4;
  properties.dtype = ssi::dtype_t::fp64;

  auto result = context->check_support(properties);
  TEST_CHECK(result.supported());
  TEST_CHECK(context->supports(properties));
}

static void test_solve_result_metadata(void)
{
  auto context = std::make_shared<test_context_t>();

  ssi::sparse_problem_properties_t properties;
  properties.nrows = 3;
  properties.ncols = 3;
  properties.dtype = ssi::dtype_t::fp64;

  auto problem = context->make_sparse_problem(properties);
  auto matrix = problem->make_sparse_matrix();
  auto symbolic = problem->make_symbolic_analysis();
  auto factorization = symbolic->make_numeric_factorization(matrix);
  auto rhs = context->make_matrix(ssi::dtype_t::fp64);
  auto solution = context->make_matrix(ssi::dtype_t::fp64);

  auto result = factorization->solve(*rhs,*solution);
  TEST_CHECK(result.success());
  TEST_CHECK(result.converged);
  TEST_CHECK(result.iterations == 0);
  TEST_CHECK(result.refinement_steps == 1);
  TEST_CHECK(result.residual_norm == 0.0);
  TEST_CHECK(result.reason == "direct solve");
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
  { "status_exception_mapping", test_status_exception_mapping },
  { "sparse_problem_property_consistency", test_sparse_problem_property_consistency },
  { "default_support_query", test_default_support_query },
  { "solve_result_metadata", test_solve_result_metadata },
  { "graph_count_builder_i32", test_graph_count_builder_i32 },
  { "graph_edge_builder_row_oriented", test_graph_edge_builder_row_oriented },
  { "graph_edge_builder_column_oriented", test_graph_edge_builder_column_oriented },
  { "sparse_value_builder_complex64", test_sparse_value_builder_complex64 },
  { "compressed_graph_view_i32", test_compressed_graph_view_i32 },
  { "sparse_values_view_float32", test_sparse_values_view_float32 },
  { 0, 0 }
};
