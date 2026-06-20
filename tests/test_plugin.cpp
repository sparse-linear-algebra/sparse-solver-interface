#include "sparse_solver_interface_plugin.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace {

class test_context_t;

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
      throw std::runtime_error("test plugin does not implement sparse matrix read");
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
        7,
        2,
        1.0e-12,
        2.0e-12,
        3.0e-12,
        "plugin solve"
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
      throw std::runtime_error("test plugin does not implement graph build");
    }

    void borrow_compressed_graph_view(const ssi::compressed_graph_view_t&) override{
      throw std::runtime_error("test plugin does not implement graph borrow");
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
      throw std::runtime_error("test plugin does not implement matrix build");
    }

    void build_from_placement(
      std::function<void(const ssi::placement_t&,ssi::matrix_view_t&)>&) override{
      throw std::runtime_error("test plugin does not implement placement build");
    }

    void read_to_host(std::function<void(const ssi::matrix_view_t&)>&) const override{
      throw std::runtime_error("test plugin does not implement matrix read");
    }

    void read_to_placement(
      const ssi::placement_t&,
      std::function<void(const ssi::matrix_view_t&)>&) const override{
      throw std::runtime_error("test plugin does not implement placement read");
    }

  private:
    int64_t nrows_ = 0;
    int64_t ncols_ = 0;
    ssi::dtype_t dtype_;
};

class test_context_t final :
  public ssi::context_t,
  public std::enable_shared_from_this<test_context_t>{
  public:
    std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t dtype) override{
      return std::make_shared<test_matrix_t>(shared_from_this(),dtype);
    }

    ssi::support_result_t check_support(
      const ssi::sparse_problem_properties_t& properties) const override{
      if(properties.dtype == ssi::dtype_t::c128){
        return {
          ssi::status_t::unsupported,
          "c128 is not supported by the test plugin"
        };
      }
      return ssi::context_t::check_support(properties);
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

std::shared_ptr<ssi::context_t> make_test_context(){
  return std::make_shared<test_context_t>();
}

}  // namespace

SSI_EXPORT_PLUGIN(make_test_context)
