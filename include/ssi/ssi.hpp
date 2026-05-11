#ifndef SSI_SSI_HPP
#define SSI_SSI_HPP

#include "../sparse_solver_interface.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ssi {

class error : public std::runtime_error {
public:
  explicit error(const char* message) : std::runtime_error(message ? message : "ssi error") {}
};

inline void check(ssi_status_t status)
{
  if (status != SSI_SUCCESS) {
    throw error("ssi operation failed");
  }
}

template <typename T>
struct dtype;

template <>
struct dtype<float> {
  static constexpr ssi_dtype_t value = SSI_DTYPE_F32;
};

template <>
struct dtype<double> {
  static constexpr ssi_dtype_t value = SSI_DTYPE_F64;
};

template <>
struct dtype<ssi_complex64_t> {
  static constexpr ssi_dtype_t value = SSI_DTYPE_C64;
};

template <>
struct dtype<ssi_complex128_t> {
  static constexpr ssi_dtype_t value = SSI_DTYPE_C128;
};

template <typename T>
struct itype;

template <>
struct itype<std::int32_t> {
  static constexpr ssi_itype_t value = SSI_ITYPE_I32;
};

template <>
struct itype<std::int64_t> {
  static constexpr ssi_itype_t value = SSI_ITYPE_I64;
};

template <typename T>
class matrix_view {
public:
  matrix_view(T* values, std::int64_t rows, std::int64_t columns, std::int64_t leading_dimension, ssi_matrix_order_t order) noexcept
      : values_(values), rows_(rows), columns_(columns), leading_dimension_(leading_dimension), order_(order)
  {
  }

  T& operator()(std::int64_t row, std::int64_t column) noexcept
  {
    return values_[offset(row, column)];
  }

  const T& operator()(std::int64_t row, std::int64_t column) const noexcept
  {
    return values_[offset(row, column)];
  }

  std::int64_t rows() const noexcept { return rows_; }
  std::int64_t columns() const noexcept { return columns_; }
  std::int64_t leading_dimension() const noexcept { return leading_dimension_; }
  ssi_matrix_order_t order() const noexcept { return order_; }
  T* data() const noexcept { return values_; }

private:
  std::int64_t offset(std::int64_t row, std::int64_t column) const noexcept
  {
    return order_ == SSI_MATRIX_ROW_MAJOR ? row * leading_dimension_ + column : column * leading_dimension_ + row;
  }

  T* values_;
  std::int64_t rows_;
  std::int64_t columns_;
  std::int64_t leading_dimension_;
  ssi_matrix_order_t order_;
};

class matrix_builder {
public:
  virtual ~matrix_builder() = default;
  virtual ssi_status_t build(const ssi_matrix_build_request_t& request, ssi_matrix_build_buffer_t& buffer) noexcept = 0;
};

class sparse_graph_builder {
public:
  virtual ~sparse_graph_builder() = default;
  virtual ssi_status_t build(const ssi_sparse_graph_build_request_t& request, ssi_sparse_graph_build_buffer_t& buffer) noexcept = 0;
};

class sparse_matrix_builder {
public:
  virtual ~sparse_matrix_builder() = default;
  virtual ssi_status_t build(const ssi_sparse_matrix_build_request_t& request, ssi_sparse_matrix_build_buffer_t& buffer) noexcept = 0;
};

class matrix_reader {
public:
  virtual ~matrix_reader() = default;
  virtual ssi_status_t read(const ssi_matrix_read_request_t& request, const ssi_matrix_read_buffer_t& buffer) noexcept = 0;
};

class solver {
public:
  virtual ~solver() = default;

  virtual ssi_status_t create_context(const ssi_config_t* config, ssi_context_t** out_context) noexcept = 0;
  virtual void destroy_context(ssi_context_t** context) noexcept = 0;

  virtual ssi_status_t create_matrix(ssi_context_t* context, const ssi_matrix_desc_t& desc, ssi_matrix_t** out_matrix) noexcept = 0;
  virtual ssi_status_t build_matrix(ssi_context_t* context, ssi_matrix_t* matrix, matrix_builder& builder) noexcept = 0;
  virtual ssi_status_t read_matrix(ssi_context_t* context, const ssi_matrix_t* matrix, matrix_reader& reader) noexcept = 0;
  virtual void destroy_matrix(ssi_context_t* context, ssi_matrix_t** matrix) noexcept = 0;

  virtual ssi_status_t create_sparse_graph(ssi_context_t* context, const ssi_sparse_graph_desc_t& desc, ssi_sparse_graph_t** out_graph) noexcept = 0;
  virtual ssi_status_t build_sparse_graph(ssi_context_t* context, ssi_sparse_graph_t* graph, sparse_graph_builder& builder) noexcept = 0;
  virtual void destroy_sparse_graph(ssi_context_t* context, ssi_sparse_graph_t** graph) noexcept = 0;

  virtual ssi_status_t create_sparse_matrix(ssi_context_t* context, ssi_sparse_graph_t* graph, const ssi_sparse_matrix_desc_t& desc, ssi_sparse_matrix_t** out_matrix) noexcept = 0;
  virtual ssi_status_t build_sparse_matrix(ssi_context_t* context, ssi_sparse_matrix_t* matrix, sparse_matrix_builder& builder) noexcept = 0;
  virtual void destroy_sparse_matrix(ssi_context_t* context, ssi_sparse_matrix_t** matrix) noexcept = 0;

  virtual ssi_status_t create_symbolic_factorization(ssi_context_t* context, ssi_sparse_graph_t* graph, ssi_symbolic_factorization_t** out_factorization) noexcept = 0;
  virtual void destroy_symbolic_factorization(ssi_context_t* context, ssi_symbolic_factorization_t** factorization) noexcept = 0;

  virtual ssi_status_t create_numeric_factorization(ssi_context_t* context, ssi_symbolic_factorization_t* symbolic, ssi_sparse_matrix_t* matrix, ssi_numeric_factorization_t** out_factorization) noexcept = 0;
  virtual void destroy_numeric_factorization(ssi_context_t* context, ssi_numeric_factorization_t** factorization) noexcept = 0;

  virtual ssi_status_t solve(ssi_context_t* context, const ssi_numeric_factorization_t* factorization, const ssi_matrix_t* input, ssi_matrix_t* output) noexcept = 0;
  virtual ssi_status_t await(ssi_context_t* context) noexcept = 0;
  virtual const char* last_error(ssi_context_t* context) noexcept = 0;
};

namespace detail {

template <typename Callback, typename Request, typename Buffer>
class builder_adapter;

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_matrix_build_request_t& request, ssi_matrix_build_buffer_t& buffer) noexcept;

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_sparse_graph_build_request_t& request, ssi_sparse_graph_build_buffer_t& buffer) noexcept;

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_sparse_matrix_build_request_t& request, ssi_sparse_matrix_build_buffer_t& buffer) noexcept;

template <typename Callback>
class builder_adapter<Callback, ssi_matrix_build_request_t, ssi_matrix_build_buffer_t> final : public matrix_builder {
public:
  explicit builder_adapter(Callback& callback) : callback_(callback) {}

  ssi_status_t build(const ssi_matrix_build_request_t& request, ssi_matrix_build_buffer_t& buffer) noexcept override
  {
    return invoke(callback_, request, buffer);
  }

private:
  Callback& callback_;
};

template <typename Callback>
class builder_adapter<Callback, ssi_sparse_graph_build_request_t, ssi_sparse_graph_build_buffer_t> final : public sparse_graph_builder {
public:
  explicit builder_adapter(Callback& callback) : callback_(callback) {}

  ssi_status_t build(const ssi_sparse_graph_build_request_t& request, ssi_sparse_graph_build_buffer_t& buffer) noexcept override
  {
    return invoke(callback_, request, buffer);
  }

private:
  Callback& callback_;
};

template <typename Callback>
class builder_adapter<Callback, ssi_sparse_matrix_build_request_t, ssi_sparse_matrix_build_buffer_t> final : public sparse_matrix_builder {
public:
  explicit builder_adapter(Callback& callback) : callback_(callback) {}

  ssi_status_t build(const ssi_sparse_matrix_build_request_t& request, ssi_sparse_matrix_build_buffer_t& buffer) noexcept override
  {
    return invoke(callback_, request, buffer);
  }

private:
  Callback& callback_;
};

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_matrix_build_request_t& request, ssi_matrix_build_buffer_t& buffer) noexcept
{
  try {
    return callback(request, buffer);
  } catch (...) {
    return SSI_ERROR;
  }
}

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_sparse_graph_build_request_t& request, ssi_sparse_graph_build_buffer_t& buffer) noexcept
{
  try {
    return callback(request, buffer);
  } catch (...) {
    return SSI_ERROR;
  }
}

template <typename Callback>
ssi_status_t invoke(Callback& callback, const ssi_sparse_matrix_build_request_t& request, ssi_sparse_matrix_build_buffer_t& buffer) noexcept
{
  try {
    return callback(request, buffer);
  } catch (...) {
    return SSI_ERROR;
  }
}

}  // namespace detail

template <typename Callback>
ssi_status_t build_matrix(solver& implementation, ssi_context_t* context, ssi_matrix_t* matrix, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;
  detail::builder_adapter<callback_type, ssi_matrix_build_request_t, ssi_matrix_build_buffer_t> builder(callback_ref);
  return implementation.build_matrix(context, matrix, builder);
}

template <typename Callback>
ssi_status_t build_sparse_graph(solver& implementation, ssi_context_t* context, ssi_sparse_graph_t* graph, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;
  detail::builder_adapter<callback_type, ssi_sparse_graph_build_request_t, ssi_sparse_graph_build_buffer_t> builder(callback_ref);
  return implementation.build_sparse_graph(context, graph, builder);
}

template <typename Callback>
ssi_status_t build_sparse_matrix(solver& implementation, ssi_context_t* context, ssi_sparse_matrix_t* matrix, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;
  detail::builder_adapter<callback_type, ssi_sparse_matrix_build_request_t, ssi_sparse_matrix_build_buffer_t> builder(callback_ref);
  return implementation.build_sparse_matrix(context, matrix, builder);
}

}  // namespace ssi

#include "client.hpp"
#include "export_c_api.hpp"

#endif
