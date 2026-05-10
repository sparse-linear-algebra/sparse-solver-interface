#ifndef SPARSE_SOLVER_INTERFACE_HPP
#define SPARSE_SOLVER_INTERFACE_HPP

#include "sparse_solver_interface.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ssi {

class error : public std::runtime_error {
public:
  explicit error(const char* message) : std::runtime_error(message) {}
};

inline void check(ssi_status_t status)
{
  if (status != ssi_success) {
    throw error("ssi operation failed");
  }
}

template <typename T>
struct dtype;

template <>
struct dtype<float> {
  static constexpr ssi_dtype_t value = ssi_f32;
};

template <>
struct dtype<double> {
  static constexpr ssi_dtype_t value = ssi_f64;
};

template <>
struct dtype<ssi_complex64_t> {
  static constexpr ssi_dtype_t value = ssi_c64;
};

template <>
struct dtype<ssi_complex128_t> {
  static constexpr ssi_dtype_t value = ssi_c128;
};

template <typename T>
struct itype;

template <>
struct itype<ssi_int32_t> {
  static constexpr ssi_itype_t value = ssi_i32;
};

template <>
struct itype<ssi_int64_t> {
  static constexpr ssi_itype_t value = ssi_i64;
};

template <typename Callback>
ssi_status_t build_graph(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_graph_t* graph,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_graph_builder_t builder = {};
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_graph_build_request_t* request,
                     ssi_graph_build_buffer_t* buffer,
                     void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.build_logical_graph(context, graph, &builder);
}

template <typename Callback>
ssi_status_t build_numeric(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_numeric_t* numeric,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_numeric_builder_t builder = {};
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_numeric_build_request_t* request,
                     ssi_numeric_build_buffer_t* buffer,
                     void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.build_logical_numeric(context, numeric, &builder);
}

template <typename Callback>
ssi_status_t build_dense_matrix(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_dense_matrix_t* matrix,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_dense_matrix_builder_t builder = {};
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_dense_matrix_build_request_t* request,
                     ssi_dense_matrix_build_buffer_t* buffer,
                     void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.build_logical_dense_matrix(context, matrix, &builder);
}

template <typename Callback>
ssi_status_t read_graph(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_graph_t* graph,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_graph_reader_t reader = {};
  reader.user_data = &callback_ref;
  reader.read = [](const ssi_graph_read_request_t* request,
                   const ssi_graph_read_buffer_t* buffer,
                   void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.read_logical_graph(context, graph, &reader);
}

template <typename Callback>
ssi_status_t read_numeric(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_numeric_t* numeric,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_numeric_reader_t reader = {};
  reader.user_data = &callback_ref;
  reader.read = [](const ssi_numeric_read_request_t* request,
                   const ssi_numeric_read_buffer_t* buffer,
                   void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.read_logical_numeric(context, numeric, &reader);
}

template <typename Callback>
ssi_status_t read_dense_matrix(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_dense_matrix_t* matrix,
    Callback&& callback)
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_dense_matrix_reader_t reader = {};
  reader.user_data = &callback_ref;
  reader.read = [](const ssi_dense_matrix_read_request_t* request,
                   const ssi_dense_matrix_read_buffer_t* buffer,
                   void* user_data) -> ssi_status_t {
    auto& fn = *static_cast<callback_type*>(user_data);
    return fn(*request, *buffer);
  };

  return api.read_logical_dense_matrix(context, matrix, &reader);
}

template <typename Callback>
void checked_build_graph(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_graph_t* graph,
    Callback&& callback)
{
  check(build_graph(api, context, graph, std::forward<Callback>(callback)));
}

template <typename Callback>
void checked_build_numeric(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_numeric_t* numeric,
    Callback&& callback)
{
  check(build_numeric(api, context, numeric, std::forward<Callback>(callback)));
}

template <typename Callback>
void checked_build_dense_matrix(
    const ssi_api_t& api,
    ssi_context_t* context,
    ssi_logical_dense_matrix_t* matrix,
    Callback&& callback)
{
  check(build_dense_matrix(api, context, matrix, std::forward<Callback>(callback)));
}

template <typename Callback>
void checked_read_graph(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_graph_t* graph,
    Callback&& callback)
{
  check(read_graph(api, context, graph, std::forward<Callback>(callback)));
}

template <typename Callback>
void checked_read_numeric(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_numeric_t* numeric,
    Callback&& callback)
{
  check(read_numeric(api, context, numeric, std::forward<Callback>(callback)));
}

template <typename Callback>
void checked_read_dense_matrix(
    const ssi_api_t& api,
    ssi_context_t* context,
    const ssi_logical_dense_matrix_t* matrix,
    Callback&& callback)
{
  check(read_dense_matrix(api, context, matrix, std::forward<Callback>(callback)));
}

}  // namespace ssi

#endif
