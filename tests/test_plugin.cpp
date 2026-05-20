#include "sparse_solver_interface_plugin.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace {

class test_context_t;

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

    std::shared_ptr<ssi::graph_t> make_graph(ssi::itype_t) override{
      throw std::runtime_error("test plugin does not implement graphs");
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
