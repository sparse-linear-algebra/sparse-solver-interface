#ifndef SPARSE_SOLVER_INTERFACE_PLUGIN_HPP
#define SPARSE_SOLVER_INTERFACE_PLUGIN_HPP

#include "sparse_solver_interface.hpp"
#include "sparse_solver_interface_c.h"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

struct ssi_context_t{
  std::shared_ptr<ssi::context_t> ptr;
};
struct ssi_matrix_t{
  std::shared_ptr<ssi::matrix_t> ptr;
};
struct ssi_sparse_problem_t{
  std::shared_ptr<ssi::sparse_problem_t> ptr;
};
struct ssi_graph_t{
  std::shared_ptr<ssi::graph_t> ptr;
};
struct ssi_sparse_matrix_t{
  std::shared_ptr<ssi::sparse_matrix_t> ptr;
};
struct ssi_symbolic_t{
  std::shared_ptr<ssi::symbolic_t> ptr;
};
struct ssi_numeric_factorization_t{
  std::shared_ptr<ssi::numeric_factorization_t> ptr;
};

namespace ssi::plugin {

namespace detail {

inline thread_local std::string last_error;

inline const char* last_error_message(){
  return last_error.c_str();
}

inline const char* store_last_error(const std::string& message){
  last_error = message;
  return last_error.empty() ? nullptr : last_error.c_str();
}

inline ssi_status_t to_c(status_t status){
  switch(status){
    case status_t::ok: return SSI_STATUS_OK;
    case status_t::invalid_argument: return SSI_STATUS_INVALID_ARGUMENT;
    case status_t::out_of_range: return SSI_STATUS_OUT_OF_RANGE;
    case status_t::unsupported: return SSI_STATUS_UNSUPPORTED;
    case status_t::singular: return SSI_STATUS_SINGULAR;
    case status_t::rank_deficient: return SSI_STATUS_RANK_DEFICIENT;
    case status_t::indefinite: return SSI_STATUS_INDEFINITE;
    case status_t::zero_pivot: return SSI_STATUS_ZERO_PIVOT;
    case status_t::breakdown: return SSI_STATUS_BREAKDOWN;
    case status_t::not_converged: return SSI_STATUS_NOT_CONVERGED;
    case status_t::exception: return SSI_STATUS_EXCEPTION;
  }
  return SSI_STATUS_EXCEPTION;
}

inline status_t from_c(ssi_status_t status){
  switch(status){
    case SSI_STATUS_OK: return status_t::ok;
    case SSI_STATUS_INVALID_ARGUMENT: return status_t::invalid_argument;
    case SSI_STATUS_OUT_OF_RANGE: return status_t::out_of_range;
    case SSI_STATUS_UNSUPPORTED: return status_t::unsupported;
    case SSI_STATUS_SINGULAR: return status_t::singular;
    case SSI_STATUS_RANK_DEFICIENT: return status_t::rank_deficient;
    case SSI_STATUS_INDEFINITE: return status_t::indefinite;
    case SSI_STATUS_ZERO_PIVOT: return status_t::zero_pivot;
    case SSI_STATUS_BREAKDOWN: return status_t::breakdown;
    case SSI_STATUS_NOT_CONVERGED: return status_t::not_converged;
    case SSI_STATUS_EXCEPTION: return status_t::exception;
  }
  return status_t::exception;
}

inline ssi_status_t status_from_exception(){
  try{
    throw;
  }catch(const error_t& e){
    last_error = e.what();
    return to_c(e.status());
  }catch(const std::invalid_argument& e){
    last_error = e.what();
    return SSI_STATUS_INVALID_ARGUMENT;
  }catch(const std::out_of_range& e){
    last_error = e.what();
    return SSI_STATUS_OUT_OF_RANGE;
  }catch(const std::exception& e){
    last_error = e.what();
    return SSI_STATUS_EXCEPTION;
  }catch(...){
    last_error = "unknown exception";
    return SSI_STATUS_EXCEPTION;
  }
}

template<typename F>
ssi_status_t guard(F&& f){
  try{
    last_error.clear();
    f();
    return SSI_STATUS_OK;
  }catch(...){
    return status_from_exception();
  }
}

inline void check_status(const ssi_plugin_api_t& api,ssi_status_t status){
  if(status == SSI_STATUS_OK){
    return;
  }
  const char* message = api.last_error_message ? api.last_error_message() : nullptr;
  if(message == nullptr || message[0] == '\0'){
    message = "sparse solver plugin call failed";
  }
  if(status == SSI_STATUS_INVALID_ARGUMENT){
    throw std::invalid_argument(message);
  }
  if(status == SSI_STATUS_OUT_OF_RANGE){
    throw std::out_of_range(message);
  }
  if(status == SSI_STATUS_UNSUPPORTED){
    throw unsupported_error_t(message);
  }
  if(status == SSI_STATUS_SINGULAR){
    throw singular_error_t(message);
  }
  if(status == SSI_STATUS_RANK_DEFICIENT){
    throw rank_deficient_error_t(message);
  }
  if(status == SSI_STATUS_INDEFINITE){
    throw indefinite_error_t(message);
  }
  if(status == SSI_STATUS_ZERO_PIVOT){
    throw zero_pivot_error_t(message);
  }
  if(status == SSI_STATUS_BREAKDOWN){
    throw breakdown_error_t(message);
  }
  if(status == SSI_STATUS_NOT_CONVERGED){
    throw not_converged_error_t(message);
  }
  throw std::runtime_error(message);
}

inline void check_callback_status(ssi_status_t status){
  ssi_plugin_api_t api{};
  api.last_error_message = last_error_message;
  check_status(api,status);
}

inline ssi_dtype_t to_c(dtype_t dtype){
  return static_cast<ssi_dtype_t>(dtype);
}

inline dtype_t from_c(ssi_dtype_t dtype){
  return static_cast<dtype_t>(dtype);
}

inline ssi_itype_t to_c(itype_t itype){
  return static_cast<ssi_itype_t>(itype);
}

inline itype_t from_c(ssi_itype_t itype){
  return static_cast<itype_t>(itype);
}

inline ssi_matrix_order_t to_c(matrix_order_t order){
  return static_cast<ssi_matrix_order_t>(order);
}

inline matrix_order_t from_c(ssi_matrix_order_t order){
  return static_cast<matrix_order_t>(order);
}

inline ssi_graph_orientation_t to_c(graph_orientation_t orientation){
  return static_cast<ssi_graph_orientation_t>(orientation);
}

inline graph_orientation_t from_c(ssi_graph_orientation_t orientation){
  return static_cast<graph_orientation_t>(orientation);
}

inline ssi_property_state_t to_c(property_state_t state){
  return static_cast<ssi_property_state_t>(state);
}

inline property_state_t from_c(ssi_property_state_t state){
  return static_cast<property_state_t>(state);
}

inline ssi_symmetric_storage_t to_c(symmetric_storage_t storage){
  return static_cast<ssi_symmetric_storage_t>(storage);
}

inline symmetric_storage_t from_c(ssi_symmetric_storage_t storage){
  return static_cast<symmetric_storage_t>(storage);
}

inline ssi_sparse_problem_properties_t to_c(
  const sparse_problem_properties_t& properties){
  return {
    properties.nrows,
    properties.ncols,
    to_c(properties.orientation),
    to_c(properties.itype),
    to_c(properties.dtype),
    to_c(properties.structurally_symmetric),
    to_c(properties.numerically_symmetric),
    to_c(properties.positive_definite),
    to_c(properties.negative_definite),
    to_c(properties.full_column_rank),
    to_c(properties.full_row_rank),
    to_c(properties.nonsingular),
    to_c(properties.strong_hall),
    to_c(properties.symmetric_storage)
  };
}

inline sparse_problem_properties_t from_c(
  const ssi_sparse_problem_properties_t& properties){
  return {
    properties.nrows,
    properties.ncols,
    from_c(properties.orientation),
    from_c(properties.itype),
    from_c(properties.dtype),
    from_c(properties.structurally_symmetric),
    from_c(properties.numerically_symmetric),
    from_c(properties.positive_definite),
    from_c(properties.negative_definite),
    from_c(properties.full_column_rank),
    from_c(properties.full_row_rank),
    from_c(properties.nonsingular),
    from_c(properties.strong_hall),
    from_c(properties.symmetric_storage)
  };
}

inline ssi_support_result_t to_c(const support_result_t& result){
  return {
    to_c(result.status),
    result.reason.empty() ? nullptr : store_last_error(result.reason)
  };
}

inline support_result_t from_c(const ssi_support_result_t& result){
  return {
    from_c(result.status),
    result.reason == nullptr ? std::string{} : std::string(result.reason)
  };
}

inline ssi_solve_result_t to_c(const solve_result_t& result){
  return {
    to_c(result.status),
    result.converged ? 1 : 0,
    result.iterations,
    result.refinement_steps,
    result.residual_norm,
    result.relative_residual_norm,
    result.backward_error,
    result.reason.empty() ? nullptr : store_last_error(result.reason)
  };
}

inline solve_result_t from_c(const ssi_solve_result_t& result){
  return {
    from_c(result.status),
    result.converged != 0,
    result.iterations,
    result.refinement_steps,
    result.residual_norm,
    result.relative_residual_norm,
    result.backward_error,
    result.reason == nullptr ? std::string{} : std::string(result.reason)
  };
}

inline void* data_pointer(matrix_view_t& view){
  if(view.dtype == dtype_t::fp32) return view.d.fp32;
  if(view.dtype == dtype_t::fp64) return view.d.fp64;
  if(view.dtype == dtype_t::c64) return view.d.c64;
  if(view.dtype == dtype_t::c128) return view.d.c128;
  return nullptr;
}

inline const void* data_pointer(const matrix_view_t& view){
  if(view.dtype == dtype_t::fp32) return view.d.fp32;
  if(view.dtype == dtype_t::fp64) return view.d.fp64;
  if(view.dtype == dtype_t::c64) return view.d.c64;
  if(view.dtype == dtype_t::c128) return view.d.c128;
  return nullptr;
}

inline matrix_view_t from_c(const ssi_matrix_view_t& view){
  matrix_view_t out{
    from_c(view.order),
    from_c(view.dtype),
    view.rbeg,
    view.rend,
    view.cbeg,
    view.cend,
    view.ld,
    {.fp32 = nullptr}
  };
  if(out.dtype == dtype_t::fp32) out.d.fp32 = static_cast<float32_t*>(view.data);
  if(out.dtype == dtype_t::fp64) out.d.fp64 = static_cast<float64_t*>(view.data);
  if(out.dtype == dtype_t::c64) out.d.c64 = static_cast<complex64_t*>(view.data);
  if(out.dtype == dtype_t::c128) out.d.c128 = static_cast<complex128_t*>(view.data);
  return out;
}

inline ssi_matrix_view_t to_c(matrix_view_t& view){
  return {
    to_c(view.order),
    to_c(view.dtype),
    view.rbeg,
    view.rend,
    view.cbeg,
    view.cend,
    view.ld,
    data_pointer(view)
  };
}

inline ssi_matrix_view_t to_c(const matrix_view_t& view){
  return {
    to_c(view.order),
    to_c(view.dtype),
    view.rbeg,
    view.rend,
    view.cbeg,
    view.cend,
    view.ld,
    const_cast<void*>(data_pointer(view))
  };
}

inline graph_count_builder_t from_c(const ssi_graph_count_builder_t& builder){
  graph_count_builder_t out{
    from_c(builder.orientation),
    from_c(builder.itype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    {.i32 = nullptr}
  };
  if(out.itype == itype_t::i32) out.counts.i32 = static_cast<int32_t*>(builder.counts);
  if(out.itype == itype_t::i64) out.counts.i64 = static_cast<int64_t*>(builder.counts);
  return out;
}

inline ssi_graph_count_builder_t to_c(graph_count_builder_t& builder){
  void* counts = builder.itype == itype_t::i32 ?
    static_cast<void*>(builder.counts.i32) :
    static_cast<void*>(builder.counts.i64);
  return {
    to_c(builder.orientation),
    to_c(builder.itype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    counts
  };
}

inline graph_edge_builder_t from_c(const ssi_graph_edge_builder_t& builder){
  graph_edge_builder_t out{
    from_c(builder.orientation),
    from_c(builder.itype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    {.i32 = nullptr},
    {.i32 = nullptr}
  };
  if(out.itype == itype_t::i32){
    out.offsets.i32 = static_cast<const int32_t*>(builder.offsets);
    out.ids.i32 = static_cast<int32_t*>(builder.ids);
  }
  if(out.itype == itype_t::i64){
    out.offsets.i64 = static_cast<const int64_t*>(builder.offsets);
    out.ids.i64 = static_cast<int64_t*>(builder.ids);
  }
  return out;
}

inline ssi_graph_edge_builder_t to_c(graph_edge_builder_t& builder){
  const void* offsets = builder.itype == itype_t::i32 ?
    static_cast<const void*>(builder.offsets.i32) :
    static_cast<const void*>(builder.offsets.i64);
  void* ids = builder.itype == itype_t::i32 ?
    static_cast<void*>(builder.ids.i32) :
    static_cast<void*>(builder.ids.i64);
  return {
    to_c(builder.orientation),
    to_c(builder.itype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    offsets,
    ids
  };
}

inline compressed_graph_view_t from_c(const ssi_compressed_graph_view_t& view){
  compressed_graph_view_t out{
    from_c(view.orientation),
    from_c(view.itype),
    view.nrows,
    view.ncols,
    view.beg,
    view.end,
    {.i32 = nullptr},
    {.i32 = nullptr}
  };
  if(out.itype == itype_t::i32){
    out.offsets.i32 = static_cast<const int32_t*>(view.offsets);
    out.ids.i32 = static_cast<const int32_t*>(view.ids);
  }
  if(out.itype == itype_t::i64){
    out.offsets.i64 = static_cast<const int64_t*>(view.offsets);
    out.ids.i64 = static_cast<const int64_t*>(view.ids);
  }
  return out;
}

inline ssi_compressed_graph_view_t to_c(const compressed_graph_view_t& view){
  const void* offsets = view.itype == itype_t::i32 ?
    static_cast<const void*>(view.offsets.i32) :
    static_cast<const void*>(view.offsets.i64);
  const void* ids = view.itype == itype_t::i32 ?
    static_cast<const void*>(view.ids.i32) :
    static_cast<const void*>(view.ids.i64);
  return {
    to_c(view.orientation),
    to_c(view.itype),
    view.nrows,
    view.ncols,
    view.beg,
    view.end,
    offsets,
    ids
  };
}

inline sparse_values_view_t from_c(const ssi_sparse_values_view_t& view){
  sparse_values_view_t out{from_c(view.dtype),view.nedges,{.fp32 = nullptr}};
  if(out.dtype == dtype_t::fp32) out.values.fp32 = static_cast<const float32_t*>(view.values);
  if(out.dtype == dtype_t::fp64) out.values.fp64 = static_cast<const float64_t*>(view.values);
  if(out.dtype == dtype_t::c64) out.values.c64 = static_cast<const complex64_t*>(view.values);
  if(out.dtype == dtype_t::c128) out.values.c128 = static_cast<const complex128_t*>(view.values);
  return out;
}

inline ssi_sparse_values_view_t to_c(const sparse_values_view_t& view){
  const void* values = nullptr;
  if(view.dtype == dtype_t::fp32) values = view.values.fp32;
  if(view.dtype == dtype_t::fp64) values = view.values.fp64;
  if(view.dtype == dtype_t::c64) values = view.values.c64;
  if(view.dtype == dtype_t::c128) values = view.values.c128;
  return {to_c(view.dtype),view.nedges,values};
}

inline sparse_value_builder_t from_c(const ssi_sparse_value_builder_t& builder){
  sparse_value_builder_t out{
    from_c(builder.orientation),
    from_c(builder.itype),
    from_c(builder.dtype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    {.i32 = nullptr},
    {.i32 = nullptr},
    {.fp32 = nullptr}
  };
  if(out.itype == itype_t::i32){
    out.offsets.i32 = static_cast<const int32_t*>(builder.offsets);
    out.ids.i32 = static_cast<const int32_t*>(builder.ids);
  }
  if(out.itype == itype_t::i64){
    out.offsets.i64 = static_cast<const int64_t*>(builder.offsets);
    out.ids.i64 = static_cast<const int64_t*>(builder.ids);
  }
  if(out.dtype == dtype_t::fp32) out.values.fp32 = static_cast<float32_t*>(builder.values);
  if(out.dtype == dtype_t::fp64) out.values.fp64 = static_cast<float64_t*>(builder.values);
  if(out.dtype == dtype_t::c64) out.values.c64 = static_cast<complex64_t*>(builder.values);
  if(out.dtype == dtype_t::c128) out.values.c128 = static_cast<complex128_t*>(builder.values);
  return out;
}

inline ssi_sparse_value_builder_t to_c(sparse_value_builder_t& builder){
  const void* offsets = builder.itype == itype_t::i32 ?
    static_cast<const void*>(builder.offsets.i32) :
    static_cast<const void*>(builder.offsets.i64);
  const void* ids = builder.itype == itype_t::i32 ?
    static_cast<const void*>(builder.ids.i32) :
    static_cast<const void*>(builder.ids.i64);
  void* values = nullptr;
  if(builder.dtype == dtype_t::fp32) values = builder.values.fp32;
  if(builder.dtype == dtype_t::fp64) values = builder.values.fp64;
  if(builder.dtype == dtype_t::c64) values = builder.values.c64;
  if(builder.dtype == dtype_t::c128) values = builder.values.c128;
  return {
    to_c(builder.orientation),
    to_c(builder.itype),
    to_c(builder.dtype),
    builder.nrows,
    builder.ncols,
    builder.beg,
    builder.end,
    offsets,
    ids,
    values
  };
}

inline ssi_sparse_value_builder_t to_c(const sparse_value_builder_t& builder){
  auto copy = builder;
  return to_c(copy);
}

}  // namespace detail

using create_context_fn = std::shared_ptr<context_t> (*)();

namespace detail {

inline ssi_status_t create_context(create_context_fn create,ssi_context_h* out_context){
  return guard([&]{
    if(out_context == nullptr){
      throw std::invalid_argument("null output context");
    }
    *out_context = new ssi_context_t{create()};
  });
}

inline create_context_fn& exported_create_context_function(){
  static create_context_fn create = nullptr;
  return create;
}

inline ssi_status_t exported_create_context(ssi_context_h* out_context){
  return create_context(exported_create_context_function(),out_context);
}

inline void context_release(ssi_context_h context){
  delete context;
}

inline ssi_status_t context_make_matrix(
  ssi_context_h context,
  ssi_dtype_t dtype,
  ssi_matrix_h* out_matrix){
  return guard([&]{
    if(context == nullptr || out_matrix == nullptr){
      throw std::invalid_argument("null context or output matrix");
    }
    *out_matrix = new ssi_matrix_t{context->ptr->make_matrix(from_c(dtype))};
  });
}

inline ssi_status_t context_make_graph(
  ssi_context_h context,
  ssi_itype_t itype,
  ssi_graph_h* out_graph){
  return guard([&]{
    if(context == nullptr || out_graph == nullptr){
      throw std::invalid_argument("null context or output graph");
    }
    *out_graph = new ssi_graph_t{context->ptr->make_graph(from_c(itype))};
  });
}

inline ssi_status_t context_make_sparse_problem(
  ssi_context_h context,
  const ssi_sparse_problem_properties_t* properties,
  ssi_sparse_problem_h* out_problem){
  return guard([&]{
    if(context == nullptr || properties == nullptr || out_problem == nullptr){
      throw std::invalid_argument("null context, properties, or output problem");
    }
    *out_problem = new ssi_sparse_problem_t{
      context->ptr->make_sparse_problem(from_c(*properties))
    };
  });
}

inline ssi_status_t context_check_support(
  ssi_context_h context,
  const ssi_sparse_problem_properties_t* properties,
  ssi_support_result_t* out_result){
  return guard([&]{
    if(context == nullptr || properties == nullptr || out_result == nullptr){
      throw std::invalid_argument("null context, properties, or support result");
    }
    *out_result = to_c(context->ptr->check_support(from_c(*properties)));
  });
}

inline void matrix_release(ssi_matrix_h matrix){ delete matrix; }
inline void sparse_problem_release(ssi_sparse_problem_h problem){ delete problem; }
inline void graph_release(ssi_graph_h graph){ delete graph; }
inline void sparse_matrix_release(ssi_sparse_matrix_h matrix){ delete matrix; }
inline void symbolic_release(ssi_symbolic_h symbolic){ delete symbolic; }
inline void numeric_factorization_release(ssi_numeric_factorization_h factorization){
  delete factorization;
}

inline ssi_status_t matrix_nrows(ssi_matrix_h matrix,int64_t* out){
  return guard([&]{ *out = matrix->ptr->nrows(); });
}
inline ssi_status_t matrix_ncols(ssi_matrix_h matrix,int64_t* out){
  return guard([&]{ *out = matrix->ptr->ncols(); });
}
inline ssi_status_t matrix_dtype(ssi_matrix_h matrix,ssi_dtype_t* out){
  return guard([&]{ *out = to_c(matrix->ptr->dtype()); });
}
inline ssi_status_t matrix_preallocate(ssi_matrix_h matrix,int64_t nrows,int64_t ncols){
  return guard([&]{ matrix->ptr->preallocate(nrows,ncols); });
}
inline ssi_status_t matrix_borrow_matrix_view(
  ssi_matrix_h matrix,
  const ssi_matrix_view_t* view){
  return guard([&]{
    if(view == nullptr){
      throw std::invalid_argument("null matrix view");
    }
    matrix_view_t cpp_view = from_c(*view);
    class host_placement_t final : public placement_t {};
    static host_placement_t host_placement;
    matrix->ptr->borrow_matrix_view(host_placement,cpp_view);
  });
}

struct matrix_build_callback_state{
  ssi_matrix_view_callback_t callback;
  void* user_data;
};

inline ssi_status_t matrix_build_from_host(
  ssi_matrix_h matrix,
  ssi_matrix_view_callback_t builder,
  void* user_data){
  return guard([&]{
    matrix_build_callback_state state{builder,user_data};
    std::function<void(matrix_view_t&)> cpp_builder = [&](matrix_view_t& view){
      ssi_matrix_view_t c_view = to_c(view);
      check_callback_status(state.callback(&c_view,state.user_data));
    };
    matrix->ptr->build_from_host(cpp_builder);
  });
}

struct matrix_read_callback_state{
  ssi_const_matrix_view_callback_t callback;
  void* user_data;
};

inline ssi_status_t matrix_read_to_host(
  ssi_matrix_h matrix,
  ssi_const_matrix_view_callback_t reader,
  void* user_data){
  return guard([&]{
    matrix_read_callback_state state{reader,user_data};
    std::function<void(const matrix_view_t&)> cpp_reader = [&](const matrix_view_t& view){
      ssi_matrix_view_t c_view = to_c(view);
      check_callback_status(state.callback(&c_view,state.user_data));
    };
    matrix->ptr->read_to_host(cpp_reader);
  });
}

inline ssi_status_t sparse_problem_properties(
  ssi_sparse_problem_h problem,
  ssi_sparse_problem_properties_t* out){
  return guard([&]{ *out = to_c(problem->ptr->properties()); });
}

inline ssi_status_t sparse_problem_assert_properties(
  ssi_sparse_problem_h problem,
  const ssi_sparse_problem_properties_t* properties){
  return guard([&]{
    if(properties == nullptr){
      throw std::invalid_argument("null sparse problem properties");
    }
    problem->ptr->assert_properties(from_c(*properties));
  });
}

inline ssi_status_t sparse_problem_compute_missing_properties(
  ssi_sparse_problem_h problem){
  return guard([&]{ problem->ptr->compute_missing_properties(); });
}

inline ssi_status_t sparse_problem_make_graph(
  ssi_sparse_problem_h problem,
  ssi_graph_h* out_graph){
  return guard([&]{ *out_graph = new ssi_graph_t{problem->ptr->make_graph()}; });
}

inline ssi_status_t sparse_problem_make_sparse_matrix(
  ssi_sparse_problem_h problem,
  ssi_sparse_matrix_h* out_matrix){
  return guard([&]{
    *out_matrix = new ssi_sparse_matrix_t{problem->ptr->make_sparse_matrix()};
  });
}

inline ssi_status_t sparse_problem_make_symbolic_analysis(
  ssi_sparse_problem_h problem,
  ssi_symbolic_h* out_symbolic){
  return guard([&]{
    *out_symbolic = new ssi_symbolic_t{problem->ptr->make_symbolic_analysis()};
  });
}

inline ssi_status_t graph_itype(ssi_graph_h graph,ssi_itype_t* out){
  return guard([&]{ *out = to_c(graph->ptr->itype()); });
}
inline ssi_status_t graph_nrows(ssi_graph_h graph,int64_t* out){
  return guard([&]{ *out = graph->ptr->nrows(); });
}
inline ssi_status_t graph_ncols(ssi_graph_h graph,int64_t* out){
  return guard([&]{ *out = graph->ptr->ncols(); });
}
inline ssi_status_t graph_nedges(ssi_graph_h graph,int64_t* out){
  return guard([&]{ *out = graph->ptr->nedges(); });
}

inline ssi_status_t graph_build_from_host(
  ssi_graph_h graph,
  int64_t nrows,
  int64_t ncols,
  ssi_graph_orientation_t orientation,
  ssi_graph_count_callback_t count_builder,
  void* count_user_data,
  ssi_graph_edge_callback_t edge_builder,
  void* edge_user_data){
  return guard([&]{
    std::function<void(graph_count_builder_t&)> cpp_count = [&](graph_count_builder_t& builder){
      ssi_graph_count_builder_t c_builder = to_c(builder);
      check_callback_status(count_builder(&c_builder,count_user_data));
    };
    std::function<void(graph_edge_builder_t&)> cpp_edge = [&](graph_edge_builder_t& builder){
      ssi_graph_edge_builder_t c_builder = to_c(builder);
      check_callback_status(edge_builder(&c_builder,edge_user_data));
    };
    graph->ptr->build_from_host(nrows,ncols,from_c(orientation),cpp_count,cpp_edge);
  });
}

inline ssi_status_t graph_borrow_compressed_graph_view(
  ssi_graph_h graph,
  const ssi_compressed_graph_view_t* view){
  return guard([&]{
    if(view == nullptr){
      throw std::invalid_argument("null compressed graph view");
    }
    graph->ptr->borrow_compressed_graph_view(from_c(*view));
  });
}

inline ssi_status_t graph_make_sparse_matrix(
  ssi_graph_h graph,
  ssi_sparse_matrix_h* out_matrix){
  return guard([&]{ *out_matrix = new ssi_sparse_matrix_t{graph->ptr->make_sparse_matrix()}; });
}
inline ssi_status_t graph_make_symbolic_analysis(
  ssi_graph_h graph,
  ssi_symbolic_h* out_symbolic){
  return guard([&]{ *out_symbolic = new ssi_symbolic_t{graph->ptr->make_symbolic_analysis()}; });
}

inline ssi_status_t sparse_matrix_nrows(ssi_sparse_matrix_h matrix,int64_t* out){
  return guard([&]{ *out = matrix->ptr->nrows(); });
}
inline ssi_status_t sparse_matrix_ncols(ssi_sparse_matrix_h matrix,int64_t* out){
  return guard([&]{ *out = matrix->ptr->ncols(); });
}
inline ssi_status_t sparse_matrix_dtype(ssi_sparse_matrix_h matrix,ssi_dtype_t* out){
  return guard([&]{ *out = to_c(matrix->ptr->dtype()); });
}
inline ssi_status_t sparse_matrix_build_from_host(
  ssi_sparse_matrix_h matrix,
  ssi_dtype_t dtype,
  ssi_graph_orientation_t orientation,
  ssi_sparse_value_callback_t builder,
  void* user_data){
  return guard([&]{
    std::function<void(sparse_value_builder_t&)> cpp_builder =
      [&](sparse_value_builder_t& cpp_view){
        ssi_sparse_value_builder_t c_view = to_c(cpp_view);
        check_callback_status(builder(&c_view,user_data));
      };
    matrix->ptr->build_from_host(from_c(dtype),from_c(orientation),cpp_builder);
  });
}
inline ssi_status_t sparse_matrix_read_to_host(
  ssi_sparse_matrix_h matrix,
  ssi_graph_orientation_t orientation,
  ssi_const_sparse_value_callback_t reader,
  void* user_data){
  return guard([&]{
    std::function<void(const sparse_value_builder_t&)> cpp_reader =
      [&](const sparse_value_builder_t& cpp_view){
        ssi_sparse_value_builder_t c_view = to_c(cpp_view);
        check_callback_status(reader(&c_view,user_data));
      };
    matrix->ptr->read_to_host(from_c(orientation),cpp_reader);
  });
}
inline ssi_status_t sparse_matrix_borrow_sparse_values_view(
  ssi_sparse_matrix_h matrix,
  const ssi_sparse_values_view_t* view){
  return guard([&]{
    if(view == nullptr){
      throw std::invalid_argument("null sparse values view");
    }
    matrix->ptr->borrow_sparse_values_view(from_c(*view));
  });
}

inline ssi_status_t symbolic_make_numeric_factorization(
  ssi_symbolic_h symbolic,
  ssi_sparse_matrix_h matrix,
  ssi_numeric_factorization_h* out_factorization){
  return guard([&]{
    *out_factorization = new ssi_numeric_factorization_t{
      symbolic->ptr->make_numeric_factorization(matrix->ptr)
    };
  });
}

inline ssi_status_t numeric_factorization_dtype(
  ssi_numeric_factorization_h factorization,
  ssi_dtype_t* out){
  return guard([&]{ *out = to_c(factorization->ptr->dtype()); });
}

inline ssi_status_t numeric_factorization_solve(
  ssi_numeric_factorization_h factorization,
  ssi_matrix_h rhs,
  ssi_matrix_h solution,
  ssi_solve_result_t* out_result){
  return guard([&]{
    if(factorization == nullptr || rhs == nullptr || solution == nullptr ||
       out_result == nullptr){
      throw std::invalid_argument("null factorization, rhs, solution, or solve result");
    }
    *out_result = to_c(factorization->ptr->solve(*rhs->ptr,*solution->ptr));
  });
}

inline void fill_export_api(ssi_plugin_api_t* out_api,create_context_fn create){
  exported_create_context_function() = create;
  std::memset(out_api,0,sizeof(*out_api));
  out_api->abi_version_major = SSI_ABI_VERSION_MAJOR;
  out_api->abi_version_minor = SSI_ABI_VERSION_MINOR;
  out_api->struct_size = sizeof(ssi_plugin_api_t);
  out_api->last_error_message = last_error_message;
  out_api->create_context = exported_create_context;
  out_api->context_release = context_release;
  out_api->context_make_matrix = context_make_matrix;
  out_api->context_make_graph = context_make_graph;
  out_api->context_make_sparse_problem = context_make_sparse_problem;
  out_api->context_check_support = context_check_support;
  out_api->matrix_release = matrix_release;
  out_api->matrix_nrows = matrix_nrows;
  out_api->matrix_ncols = matrix_ncols;
  out_api->matrix_dtype = matrix_dtype;
  out_api->matrix_preallocate = matrix_preallocate;
  out_api->matrix_borrow_matrix_view = matrix_borrow_matrix_view;
  out_api->matrix_build_from_host = matrix_build_from_host;
  out_api->matrix_read_to_host = matrix_read_to_host;
  out_api->sparse_problem_release = sparse_problem_release;
  out_api->sparse_problem_properties = sparse_problem_properties;
  out_api->sparse_problem_assert_properties = sparse_problem_assert_properties;
  out_api->sparse_problem_compute_missing_properties =
    sparse_problem_compute_missing_properties;
  out_api->sparse_problem_make_graph = sparse_problem_make_graph;
  out_api->sparse_problem_make_sparse_matrix = sparse_problem_make_sparse_matrix;
  out_api->sparse_problem_make_symbolic_analysis =
    sparse_problem_make_symbolic_analysis;
  out_api->graph_release = graph_release;
  out_api->graph_itype = graph_itype;
  out_api->graph_nrows = graph_nrows;
  out_api->graph_ncols = graph_ncols;
  out_api->graph_nedges = graph_nedges;
  out_api->graph_build_from_host = graph_build_from_host;
  out_api->graph_borrow_compressed_graph_view = graph_borrow_compressed_graph_view;
  out_api->graph_make_sparse_matrix = graph_make_sparse_matrix;
  out_api->graph_make_symbolic_analysis = graph_make_symbolic_analysis;
  out_api->sparse_matrix_release = sparse_matrix_release;
  out_api->sparse_matrix_nrows = sparse_matrix_nrows;
  out_api->sparse_matrix_ncols = sparse_matrix_ncols;
  out_api->sparse_matrix_dtype = sparse_matrix_dtype;
  out_api->sparse_matrix_build_from_host = sparse_matrix_build_from_host;
  out_api->sparse_matrix_read_to_host = sparse_matrix_read_to_host;
  out_api->sparse_matrix_borrow_sparse_values_view = sparse_matrix_borrow_sparse_values_view;
  out_api->symbolic_release = symbolic_release;
  out_api->symbolic_make_numeric_factorization = symbolic_make_numeric_factorization;
  out_api->numeric_factorization_release = numeric_factorization_release;
  out_api->numeric_factorization_dtype = numeric_factorization_dtype;
  out_api->numeric_factorization_solve = numeric_factorization_solve;
}

}  // namespace detail

inline ssi_status_t export_plugin(ssi_plugin_api_t* out_api,create_context_fn create){
  return detail::guard([&]{
    if(out_api == nullptr){
      throw std::invalid_argument("null plugin api");
    }
    detail::fill_export_api(out_api,create);
  });
}

}  // namespace ssi::plugin

#define SSI_EXPORT_PLUGIN(CREATE_CONTEXT_FUNCTION) \
  extern "C" SSI_EXPORT ssi_status_t ssi_get_plugin(ssi_plugin_api_t* out_api){ \
    return ::ssi::plugin::export_plugin(out_api,CREATE_CONTEXT_FUNCTION); \
  }

namespace ssi {

namespace plugin::detail {

struct imported_plugin_t{
  ssi_plugin_api_t api{};
  void* library = nullptr;

  ~imported_plugin_t(){
#if defined(__unix__) || defined(__APPLE__)
    if(library != nullptr){
      dlclose(library);
    }
#endif
  }
};

class imported_context_t;
class imported_matrix_t;
class imported_sparse_problem_t;
class imported_graph_t;
class imported_sparse_matrix_t;
class imported_symbolic_t;
class imported_numeric_factorization_t;

class imported_matrix_t final : public matrix_t{
  public:
    imported_matrix_t(
      std::shared_ptr<context_t> context,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_matrix_h handle) :
      matrix_t(std::move(context)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_matrix_t() override{
      if(handle_ != nullptr){
        plugin_->api.matrix_release(handle_);
      }
    }

    int64_t nrows() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.matrix_nrows(handle_,&out));
      return out;
    }
    int64_t ncols() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.matrix_ncols(handle_,&out));
      return out;
    }
    dtype_t dtype() const override{
      ssi_dtype_t out = SSI_DTYPE_FP64;
      check_status(plugin_->api,plugin_->api.matrix_dtype(handle_,&out));
      return from_c(out);
    }
    void preallocate(int64_t nrows,int64_t ncols) override{
      check_status(plugin_->api,plugin_->api.matrix_preallocate(handle_,nrows,ncols));
    }
    void borrow_matrix_view(const placement_t&,const matrix_view_t& view) override{
      ssi_matrix_view_t c_view = to_c(view);
      check_status(plugin_->api,plugin_->api.matrix_borrow_matrix_view(handle_,&c_view));
    }
    void build_from_host(std::function<void(matrix_view_t&)>& builder) override{
      auto callback = [](ssi_matrix_view_t* view,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(matrix_view_t&)>*>(user_data);
          matrix_view_t cpp_view = from_c(*view);
          (*fn)(cpp_view);
        });
      };
      check_status(
        plugin_->api,
        plugin_->api.matrix_build_from_host(handle_,callback,&builder));
    }
    void build_from_placement(
      std::function<void(const placement_t&,matrix_view_t&)>&) override{
      throw std::runtime_error("placement matrix builds are not exposed by the C ABI yet");
    }
    void read_to_host(std::function<void(const matrix_view_t&)>& reader) const override{
      auto callback = [](const ssi_matrix_view_t* view,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(const matrix_view_t&)>*>(user_data);
          matrix_view_t cpp_view = from_c(*view);
          (*fn)(cpp_view);
        });
      };
      check_status(
        plugin_->api,
        plugin_->api.matrix_read_to_host(handle_,callback,&reader));
    }
    void read_to_placement(
      const placement_t&,
      std::function<void(const matrix_view_t&)>&) const override{
      throw std::runtime_error("placement matrix reads are not exposed by the C ABI yet");
    }

    ssi_matrix_h handle() const{
      return handle_;
    }

  private:
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_matrix_h handle_;
};

class imported_graph_t final :
  public graph_t,
  public std::enable_shared_from_this<imported_graph_t>{
  public:
    imported_graph_t(
      std::shared_ptr<context_t> context,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_graph_h handle) :
      graph_t(std::move(context)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_graph_t() override{
      if(handle_ != nullptr){
        plugin_->api.graph_release(handle_);
      }
    }

    itype_t itype() const override{
      ssi_itype_t out = SSI_ITYPE_I64;
      check_status(plugin_->api,plugin_->api.graph_itype(handle_,&out));
      return from_c(out);
    }
    int64_t nrows() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.graph_nrows(handle_,&out));
      return out;
    }
    int64_t ncols() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.graph_ncols(handle_,&out));
      return out;
    }
    int64_t nedges() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.graph_nedges(handle_,&out));
      return out;
    }
    void build_from_host(
      int64_t nrows,
      int64_t ncols,
      graph_orientation_t orientation,
      std::function<void(graph_count_builder_t&)>& count_builder,
      std::function<void(graph_edge_builder_t&)>& edge_builder) override{
      auto count_callback = [](ssi_graph_count_builder_t* builder,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(graph_count_builder_t&)>*>(user_data);
          graph_count_builder_t cpp_builder = from_c(*builder);
          (*fn)(cpp_builder);
        });
      };
      auto edge_callback = [](ssi_graph_edge_builder_t* builder,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(graph_edge_builder_t&)>*>(user_data);
          graph_edge_builder_t cpp_builder = from_c(*builder);
          (*fn)(cpp_builder);
        });
      };
      check_status(
        plugin_->api,
        plugin_->api.graph_build_from_host(
          handle_,
          nrows,
          ncols,
          to_c(orientation),
          count_callback,
          &count_builder,
          edge_callback,
          &edge_builder));
    }
    void borrow_compressed_graph_view(const compressed_graph_view_t& view) override{
      ssi_compressed_graph_view_t c_view = to_c(view);
      check_status(
        plugin_->api,
        plugin_->api.graph_borrow_compressed_graph_view(handle_,&c_view));
    }
    std::shared_ptr<sparse_matrix_t> make_sparse_matrix() override;
    std::shared_ptr<symbolic_t> make_symbolic_analysis() override;

    ssi_graph_h handle() const{
      return handle_;
    }

  private:
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_graph_h handle_;
};

class imported_sparse_matrix_t final : public sparse_matrix_t{
  public:
    imported_sparse_matrix_t(
      std::shared_ptr<imported_graph_t> graph,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_sparse_matrix_h handle) :
      sparse_matrix_t(graph),
      graph_(std::move(graph)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_sparse_matrix_t() override{
      if(handle_ != nullptr){
        plugin_->api.sparse_matrix_release(handle_);
      }
    }

    int64_t nrows() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.sparse_matrix_nrows(handle_,&out));
      return out;
    }
    int64_t ncols() const override{
      int64_t out = 0;
      check_status(plugin_->api,plugin_->api.sparse_matrix_ncols(handle_,&out));
      return out;
    }
    dtype_t dtype() const override{
      ssi_dtype_t out = SSI_DTYPE_FP64;
      check_status(plugin_->api,plugin_->api.sparse_matrix_dtype(handle_,&out));
      return from_c(out);
    }
    void build_from_host(
      dtype_t dtype,
      graph_orientation_t orientation,
      std::function<void(sparse_value_builder_t&)>& builder) override{
      auto callback = [](ssi_sparse_value_builder_t* view,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(sparse_value_builder_t&)>*>(user_data);
          sparse_value_builder_t cpp_view = from_c(*view);
          (*fn)(cpp_view);
        });
      };
      check_status(
        plugin_->api,
        plugin_->api.sparse_matrix_build_from_host(
          handle_,
          to_c(dtype),
          to_c(orientation),
          callback,
          &builder));
    }
    void read_to_host(
      graph_orientation_t orientation,
      std::function<void(const sparse_value_builder_t&)>& reader) const override{
      auto callback = [](const ssi_sparse_value_builder_t* view,void* user_data) -> ssi_status_t{
        return guard([&]{
          auto* fn = static_cast<std::function<void(const sparse_value_builder_t&)>*>(user_data);
          sparse_value_builder_t cpp_view = from_c(*view);
          (*fn)(cpp_view);
        });
      };
      check_status(
        plugin_->api,
        plugin_->api.sparse_matrix_read_to_host(
          handle_,
          to_c(orientation),
          callback,
          &reader));
    }
    void borrow_sparse_values_view(const sparse_values_view_t& view) override{
      ssi_sparse_values_view_t c_view = to_c(view);
      check_status(
        plugin_->api,
        plugin_->api.sparse_matrix_borrow_sparse_values_view(handle_,&c_view));
    }

    ssi_sparse_matrix_h handle() const{
      return handle_;
    }

  private:
    std::shared_ptr<imported_graph_t> graph_;
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_sparse_matrix_h handle_;
};

class imported_symbolic_t final :
  public symbolic_t,
  public std::enable_shared_from_this<imported_symbolic_t>{
  public:
    imported_symbolic_t(
      std::shared_ptr<imported_graph_t> graph,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_symbolic_h handle) :
      symbolic_t(graph),
      graph_(std::move(graph)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_symbolic_t() override{
      if(handle_ != nullptr){
        plugin_->api.symbolic_release(handle_);
      }
    }

    std::shared_ptr<numeric_factorization_t>
    make_numeric_factorization(std::shared_ptr<sparse_matrix_t> matrix) override;

    ssi_symbolic_h handle() const{
      return handle_;
    }

  private:
    std::shared_ptr<imported_graph_t> graph_;
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_symbolic_h handle_;
};

class imported_numeric_factorization_t final : public numeric_factorization_t{
  public:
    imported_numeric_factorization_t(
      std::shared_ptr<imported_symbolic_t> symbolic,
      std::shared_ptr<imported_sparse_matrix_t> matrix,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_numeric_factorization_h handle) :
      numeric_factorization_t(symbolic,matrix),
      symbolic_(std::move(symbolic)),
      matrix_(std::move(matrix)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_numeric_factorization_t() override{
      if(handle_ != nullptr){
        plugin_->api.numeric_factorization_release(handle_);
      }
    }

    dtype_t dtype() const override{
      ssi_dtype_t out = SSI_DTYPE_FP64;
      check_status(plugin_->api,plugin_->api.numeric_factorization_dtype(handle_,&out));
      return from_c(out);
    }
    solve_result_t solve(const matrix_t& rhs,matrix_t& solution) const override{
      const auto* c_rhs = dynamic_cast<const imported_matrix_t*>(&rhs);
      auto* c_solution = dynamic_cast<imported_matrix_t*>(&solution);
      if(c_rhs == nullptr || c_solution == nullptr){
        throw std::invalid_argument("C ABI factorization can only solve with C ABI matrices");
      }
      ssi_solve_result_t result{};
      check_status(
        plugin_->api,
        plugin_->api.numeric_factorization_solve(
          handle_,
          c_rhs->handle(),
          c_solution->handle(),
          &result));
      return from_c(result);
    }

  private:
    std::shared_ptr<imported_symbolic_t> symbolic_;
    std::shared_ptr<imported_sparse_matrix_t> matrix_;
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_numeric_factorization_h handle_;
};

class imported_sparse_problem_t final : public sparse_problem_t{
  public:
    imported_sparse_problem_t(
      std::shared_ptr<context_t> context,
      std::shared_ptr<imported_plugin_t> plugin,
      ssi_sparse_problem_h handle,
      sparse_problem_properties_t properties) :
      sparse_problem_t(context,properties),
      context_(std::move(context)),
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_sparse_problem_t() override{
      if(handle_ != nullptr){
        plugin_->api.sparse_problem_release(handle_);
      }
    }

    sparse_problem_properties_t properties() const override{
      ssi_sparse_problem_properties_t out{};
      check_status(
        plugin_->api,
        plugin_->api.sparse_problem_properties(handle_,&out));
      return from_c(out);
    }

    void assert_properties(const sparse_problem_properties_t& properties) override{
      ssi_sparse_problem_properties_t c_properties = to_c(properties);
      check_status(
        plugin_->api,
        plugin_->api.sparse_problem_assert_properties(handle_,&c_properties));
    }

    void compute_missing_properties() override{
      check_status(
        plugin_->api,
        plugin_->api.sparse_problem_compute_missing_properties(handle_));
    }

    std::shared_ptr<graph_t> make_graph() override;
    std::shared_ptr<sparse_matrix_t> make_sparse_matrix() override;
    std::shared_ptr<symbolic_t> make_symbolic_analysis() override;

  private:
    std::shared_ptr<context_t> context_;
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_sparse_problem_h handle_;
    std::shared_ptr<imported_graph_t> graph_;
    std::shared_ptr<imported_sparse_matrix_t> matrix_;
    std::shared_ptr<imported_symbolic_t> symbolic_;
};

class imported_context_t final :
  public context_t,
  public std::enable_shared_from_this<imported_context_t>{
  public:
    imported_context_t(std::shared_ptr<imported_plugin_t> plugin,ssi_context_h handle) :
      plugin_(std::move(plugin)),
      handle_(handle) {}
    ~imported_context_t() override{
      if(handle_ != nullptr){
        plugin_->api.context_release(handle_);
      }
    }

    support_result_t check_support(
      const sparse_problem_properties_t& properties) const override{
      ssi_sparse_problem_properties_t c_properties = to_c(properties);
      ssi_support_result_t result{};
      check_status(
        plugin_->api,
        plugin_->api.context_check_support(handle_,&c_properties,&result));
      return from_c(result);
    }

    std::shared_ptr<matrix_t> make_matrix(dtype_t dtype) override{
      ssi_matrix_h out = nullptr;
      check_status(
        plugin_->api,
        plugin_->api.context_make_matrix(handle_,to_c(dtype),&out));
      return std::make_shared<imported_matrix_t>(shared_from_this(),plugin_,out);
    }
    std::shared_ptr<graph_t> make_graph(itype_t itype) override{
      ssi_graph_h out = nullptr;
      check_status(
        plugin_->api,
        plugin_->api.context_make_graph(handle_,to_c(itype),&out));
      return std::make_shared<imported_graph_t>(shared_from_this(),plugin_,out);
    }
    std::shared_ptr<sparse_problem_t> make_sparse_problem(
      const sparse_problem_properties_t& properties) override{
      ssi_sparse_problem_h out = nullptr;
      ssi_sparse_problem_properties_t c_properties = to_c(properties);
      check_status(
        plugin_->api,
        plugin_->api.context_make_sparse_problem(handle_,&c_properties,&out));
      return std::make_shared<imported_sparse_problem_t>(
        shared_from_this(),
        plugin_,
        out,
        properties);
    }

  private:
    std::shared_ptr<imported_plugin_t> plugin_;
    ssi_context_h handle_;
};

inline std::shared_ptr<sparse_matrix_t> imported_graph_t::make_sparse_matrix(){
  ssi_sparse_matrix_h out = nullptr;
  check_status(plugin_->api,plugin_->api.graph_make_sparse_matrix(handle_,&out));
  return std::make_shared<imported_sparse_matrix_t>(
    shared_from_this(),
    plugin_,
    out);
}

inline std::shared_ptr<symbolic_t> imported_graph_t::make_symbolic_analysis(){
  ssi_symbolic_h out = nullptr;
  check_status(plugin_->api,plugin_->api.graph_make_symbolic_analysis(handle_,&out));
  return std::make_shared<imported_symbolic_t>(
    shared_from_this(),
    plugin_,
    out);
}

inline std::shared_ptr<graph_t> imported_sparse_problem_t::make_graph(){
  if(graph_ == nullptr){
    ssi_graph_h out = nullptr;
    check_status(
      plugin_->api,
      plugin_->api.sparse_problem_make_graph(handle_,&out));
    graph_ = std::make_shared<imported_graph_t>(context_,plugin_,out);
  }
  return graph_;
}

inline std::shared_ptr<sparse_matrix_t> imported_sparse_problem_t::make_sparse_matrix(){
  if(matrix_ == nullptr){
    ssi_sparse_matrix_h out = nullptr;
    check_status(
      plugin_->api,
      plugin_->api.sparse_problem_make_sparse_matrix(handle_,&out));
    matrix_ = std::make_shared<imported_sparse_matrix_t>(
      std::static_pointer_cast<imported_graph_t>(make_graph()),
      plugin_,
      out);
  }
  return matrix_;
}

inline std::shared_ptr<symbolic_t> imported_sparse_problem_t::make_symbolic_analysis(){
  if(symbolic_ == nullptr){
    ssi_symbolic_h out = nullptr;
    check_status(
      plugin_->api,
      plugin_->api.sparse_problem_make_symbolic_analysis(handle_,&out));
    symbolic_ = std::make_shared<imported_symbolic_t>(
      std::static_pointer_cast<imported_graph_t>(make_graph()),
      plugin_,
      out);
  }
  return symbolic_;
}

inline std::shared_ptr<numeric_factorization_t>
imported_symbolic_t::make_numeric_factorization(std::shared_ptr<sparse_matrix_t> matrix){
  auto imported_matrix = std::dynamic_pointer_cast<imported_sparse_matrix_t>(matrix);
  if(imported_matrix == nullptr){
    throw std::invalid_argument(
      "C ABI symbolic analysis can only factor C ABI sparse matrices");
  }
  ssi_numeric_factorization_h out = nullptr;
  check_status(
    plugin_->api,
    plugin_->api.symbolic_make_numeric_factorization(
      handle_,
      imported_matrix->handle(),
      &out));
  return std::make_shared<imported_numeric_factorization_t>(
    std::static_pointer_cast<imported_symbolic_t>(shared_from_this()),
    imported_matrix,
    plugin_,
    out);
}

}  // namespace plugin::detail

inline std::shared_ptr<context_t> load_context_from_shared_object(const char* path){
#if defined(__unix__) || defined(__APPLE__)
  void* library = dlopen(path,RTLD_NOW | RTLD_LOCAL);
  if(library == nullptr){
    throw std::runtime_error(dlerror());
  }
  struct dlcloser_t{
    void operator()(void* handle) const{
      if(handle != nullptr){
        dlclose(handle);
      }
    }
  };
  auto close_on_error = std::unique_ptr<void,dlcloser_t>(library);
  auto get_plugin = reinterpret_cast<ssi_get_plugin_fn>(dlsym(library,"ssi_get_plugin"));
  if(get_plugin == nullptr){
    throw std::runtime_error("shared object does not export ssi_get_plugin");
  }
  auto plugin = std::make_shared<plugin::detail::imported_plugin_t>();
  plugin->library = library;
  plugin->api.struct_size = sizeof(ssi_plugin_api_t);
  plugin::detail::check_status(plugin->api,get_plugin(&plugin->api));
  if(plugin->api.abi_version_major != SSI_ABI_VERSION_MAJOR){
    throw std::runtime_error("incompatible sparse solver interface ABI major version");
  }
  if(plugin->api.struct_size > sizeof(ssi_plugin_api_t)){
    throw std::runtime_error("plugin ABI table is newer than this header");
  }
  ssi_context_h context = nullptr;
  plugin::detail::check_status(plugin->api,plugin->api.create_context(&context));
  close_on_error.release();
  return std::make_shared<plugin::detail::imported_context_t>(plugin,context);
#else
  (void)path;
  throw std::runtime_error("dlopen-based plugin loading is unavailable on this platform");
#endif
}

}  // namespace ssi

#endif
