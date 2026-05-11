#ifndef SSI_EXPORT_C_API_HPP
#define SSI_EXPORT_C_API_HPP

#include <new>

namespace ssi {

namespace detail {

template <typename Function>
ssi_status_t catch_status(Function&& function) noexcept
{
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return SSI_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return SSI_ERROR;
  }
}

class c_matrix_builder final : public matrix_builder {
public:
  explicit c_matrix_builder(const ssi_matrix_builder_t& builder) : builder_(builder) {}

  ssi_status_t build(const ssi_matrix_build_request_t& request, ssi_matrix_build_buffer_t& buffer) noexcept override
  {
    if (builder_.build == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    return builder_.build(&request, &buffer, builder_.user_data);
  }

private:
  const ssi_matrix_builder_t& builder_;
};

class c_sparse_graph_builder final : public sparse_graph_builder {
public:
  explicit c_sparse_graph_builder(const ssi_sparse_graph_builder_t& builder) : builder_(builder) {}

  ssi_status_t build(const ssi_sparse_graph_build_request_t& request, ssi_sparse_graph_build_buffer_t& buffer) noexcept override
  {
    if (builder_.build == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    return builder_.build(&request, &buffer, builder_.user_data);
  }

private:
  const ssi_sparse_graph_builder_t& builder_;
};

class c_sparse_matrix_builder final : public sparse_matrix_builder {
public:
  explicit c_sparse_matrix_builder(const ssi_sparse_matrix_builder_t& builder) : builder_(builder) {}

  ssi_status_t build(const ssi_sparse_matrix_build_request_t& request, ssi_sparse_matrix_build_buffer_t& buffer) noexcept override
  {
    if (builder_.build == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    return builder_.build(&request, &buffer, builder_.user_data);
  }

private:
  const ssi_sparse_matrix_builder_t& builder_;
};

class c_matrix_reader final : public matrix_reader {
public:
  explicit c_matrix_reader(const ssi_matrix_reader_t& reader) : reader_(reader) {}

  ssi_status_t read(const ssi_matrix_read_request_t& request, const ssi_matrix_read_buffer_t& buffer) noexcept override
  {
    if (reader_.read == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    return reader_.read(&request, &buffer, reader_.user_data);
  }

private:
  const ssi_matrix_reader_t& reader_;
};

template <typename Solver>
Solver& instance() noexcept
{
  static Solver solver;
  return solver;
}

template <typename Solver>
ssi_status_t create_context(const ssi_config_t* config, ssi_context_t** out_context) noexcept
{
  return catch_status([&] { return instance<Solver>().create_context(config, out_context); });
}

template <typename Solver>
void destroy_context(ssi_context_t** context) noexcept
{
  instance<Solver>().destroy_context(context);
}

template <typename Solver>
ssi_status_t create_matrix(ssi_context_t* context, const ssi_matrix_desc_t* desc, ssi_matrix_t** out_matrix) noexcept
{
  if (desc == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] { return instance<Solver>().create_matrix(context, *desc, out_matrix); });
}

template <typename Solver>
ssi_status_t build_matrix(ssi_context_t* context, ssi_matrix_t* matrix, const ssi_matrix_builder_t* builder) noexcept
{
  if (builder == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] {
    c_matrix_builder wrapped(*builder);
    return instance<Solver>().build_matrix(context, matrix, wrapped);
  });
}

template <typename Solver>
ssi_status_t read_matrix(ssi_context_t* context, const ssi_matrix_t* matrix, const ssi_matrix_reader_t* reader) noexcept
{
  if (reader == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] {
    c_matrix_reader wrapped(*reader);
    return instance<Solver>().read_matrix(context, matrix, wrapped);
  });
}

template <typename Solver>
void destroy_matrix(ssi_context_t* context, ssi_matrix_t** matrix) noexcept
{
  instance<Solver>().destroy_matrix(context, matrix);
}

template <typename Solver>
ssi_status_t create_sparse_graph(ssi_context_t* context, const ssi_sparse_graph_desc_t* desc, ssi_sparse_graph_t** out_graph) noexcept
{
  if (desc == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] { return instance<Solver>().create_sparse_graph(context, *desc, out_graph); });
}

template <typename Solver>
ssi_status_t build_sparse_graph(ssi_context_t* context, ssi_sparse_graph_t* graph, const ssi_sparse_graph_builder_t* builder) noexcept
{
  if (builder == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] {
    c_sparse_graph_builder wrapped(*builder);
    return instance<Solver>().build_sparse_graph(context, graph, wrapped);
  });
}

template <typename Solver>
void destroy_sparse_graph(ssi_context_t* context, ssi_sparse_graph_t** graph) noexcept
{
  instance<Solver>().destroy_sparse_graph(context, graph);
}

template <typename Solver>
ssi_status_t create_sparse_matrix(ssi_context_t* context, ssi_sparse_graph_t* graph, const ssi_sparse_matrix_desc_t* desc, ssi_sparse_matrix_t** out_matrix) noexcept
{
  if (desc == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] { return instance<Solver>().create_sparse_matrix(context, graph, *desc, out_matrix); });
}

template <typename Solver>
ssi_status_t build_sparse_matrix(ssi_context_t* context, ssi_sparse_matrix_t* matrix, const ssi_sparse_matrix_builder_t* builder) noexcept
{
  if (builder == nullptr) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }
  return catch_status([&] {
    c_sparse_matrix_builder wrapped(*builder);
    return instance<Solver>().build_sparse_matrix(context, matrix, wrapped);
  });
}

template <typename Solver>
void destroy_sparse_matrix(ssi_context_t* context, ssi_sparse_matrix_t** matrix) noexcept
{
  instance<Solver>().destroy_sparse_matrix(context, matrix);
}

template <typename Solver>
ssi_status_t create_symbolic_factorization(ssi_context_t* context, ssi_sparse_graph_t* graph, ssi_symbolic_factorization_t** out_factorization) noexcept
{
  return catch_status([&] { return instance<Solver>().create_symbolic_factorization(context, graph, out_factorization); });
}

template <typename Solver>
void destroy_symbolic_factorization(ssi_context_t* context, ssi_symbolic_factorization_t** factorization) noexcept
{
  instance<Solver>().destroy_symbolic_factorization(context, factorization);
}

template <typename Solver>
ssi_status_t create_numeric_factorization(ssi_context_t* context, ssi_symbolic_factorization_t* symbolic, ssi_sparse_matrix_t* matrix, ssi_numeric_factorization_t** out_factorization) noexcept
{
  return catch_status([&] { return instance<Solver>().create_numeric_factorization(context, symbolic, matrix, out_factorization); });
}

template <typename Solver>
void destroy_numeric_factorization(ssi_context_t* context, ssi_numeric_factorization_t** factorization) noexcept
{
  instance<Solver>().destroy_numeric_factorization(context, factorization);
}

template <typename Solver>
ssi_status_t solve(ssi_context_t* context, const ssi_numeric_factorization_t* factorization, const ssi_matrix_t* input, ssi_matrix_t* output) noexcept
{
  return catch_status([&] { return instance<Solver>().solve(context, factorization, input, output); });
}

template <typename Solver>
ssi_status_t await(ssi_context_t* context) noexcept
{
  return catch_status([&] { return instance<Solver>().await(context); });
}

template <typename Solver>
const char* last_error(ssi_context_t* context) noexcept
{
  return instance<Solver>().last_error(context);
}

}  // namespace detail

template <typename Solver>
ssi_status_t export_api(ssi_api_v1_t* out_api, size_t out_size) noexcept
{
  if (out_api == nullptr || out_size < offsetof(ssi_api_v1_t, reserved)) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }

  ssi_api_v1_t api = {};
  api.size = sizeof(api);
  api.abi_version = SSI_ABI_VERSION;
  api.capabilities = 0;

  api.create_context = detail::create_context<Solver>;
  api.destroy_context = detail::destroy_context<Solver>;
  api.create_matrix = detail::create_matrix<Solver>;
  api.build_matrix = detail::build_matrix<Solver>;
  api.read_matrix = detail::read_matrix<Solver>;
  api.destroy_matrix = detail::destroy_matrix<Solver>;
  api.create_sparse_graph = detail::create_sparse_graph<Solver>;
  api.build_sparse_graph = detail::build_sparse_graph<Solver>;
  api.destroy_sparse_graph = detail::destroy_sparse_graph<Solver>;
  api.create_sparse_matrix = detail::create_sparse_matrix<Solver>;
  api.build_sparse_matrix = detail::build_sparse_matrix<Solver>;
  api.destroy_sparse_matrix = detail::destroy_sparse_matrix<Solver>;
  api.create_symbolic_factorization = detail::create_symbolic_factorization<Solver>;
  api.destroy_symbolic_factorization = detail::destroy_symbolic_factorization<Solver>;
  api.create_numeric_factorization = detail::create_numeric_factorization<Solver>;
  api.destroy_numeric_factorization = detail::destroy_numeric_factorization<Solver>;
  api.solve = detail::solve<Solver>;
  api.await = detail::await<Solver>;
  api.last_error = detail::last_error<Solver>;

  const size_t copy_size = out_size < sizeof(api) ? out_size : sizeof(api);
  unsigned char* dst = reinterpret_cast<unsigned char*>(out_api);
  const unsigned char* src = reinterpret_cast<const unsigned char*>(&api);
  for (size_t i = 0; i < copy_size; ++i) {
    dst[i] = src[i];
  }
  return SSI_SUCCESS;
}

}  // namespace ssi

#endif
