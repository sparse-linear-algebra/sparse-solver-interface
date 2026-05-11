#ifndef SSI_CLIENT_HPP
#define SSI_CLIENT_HPP

namespace ssi {

class api_view {
public:
  api_view() noexcept : api_{} {}
  explicit api_view(const ssi_api_v1_t& api) noexcept : api_(api) {}

  const ssi_api_v1_t& get() const noexcept { return api_; }
  const ssi_api_v1_t* operator->() const noexcept { return &api_; }

  static ssi_status_t load(ssi_get_api_v1_fn get_api, api_view* out) noexcept
  {
    if (get_api == nullptr || out == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }

    ssi_api_v1_t api = {};
    api.size = sizeof(api);
    const ssi_status_t status = get_api(&api, sizeof(api));
    if (status != SSI_SUCCESS) {
      return status;
    }
    if (api.abi_version != SSI_ABI_VERSION) {
      return SSI_ERROR_UNSUPPORTED;
    }

    *out = api_view(api);
    return SSI_SUCCESS;
  }

private:
  ssi_api_v1_t api_;
};

template <typename Callback>
ssi_status_t build_matrix(const api_view& api, ssi_context_t* context, ssi_matrix_t* matrix, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_matrix_builder_t builder = {};
  builder.size = sizeof(builder);
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_matrix_build_request_t* request, ssi_matrix_build_buffer_t* buffer, void* user_data) -> ssi_status_t {
    if (request == nullptr || buffer == nullptr || user_data == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    try {
      auto& fn = *static_cast<callback_type*>(user_data);
      return fn(*request, *buffer);
    } catch (...) {
      return SSI_ERROR;
    }
  };

  return api->build_matrix(context, matrix, &builder);
}

template <typename Callback>
ssi_status_t build_sparse_graph(const api_view& api, ssi_context_t* context, ssi_sparse_graph_t* graph, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_sparse_graph_builder_t builder = {};
  builder.size = sizeof(builder);
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_sparse_graph_build_request_t* request, ssi_sparse_graph_build_buffer_t* buffer, void* user_data) -> ssi_status_t {
    if (request == nullptr || buffer == nullptr || user_data == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    try {
      auto& fn = *static_cast<callback_type*>(user_data);
      return fn(*request, *buffer);
    } catch (...) {
      return SSI_ERROR;
    }
  };

  return api->build_sparse_graph(context, graph, &builder);
}

template <typename Callback>
ssi_status_t build_sparse_matrix(const api_view& api, ssi_context_t* context, ssi_sparse_matrix_t* matrix, Callback&& callback) noexcept
{
  using callback_type = typename std::remove_reference<Callback>::type;
  callback_type& callback_ref = callback;

  ssi_sparse_matrix_builder_t builder = {};
  builder.size = sizeof(builder);
  builder.user_data = &callback_ref;
  builder.build = [](const ssi_sparse_matrix_build_request_t* request, ssi_sparse_matrix_build_buffer_t* buffer, void* user_data) -> ssi_status_t {
    if (request == nullptr || buffer == nullptr || user_data == nullptr) {
      return SSI_ERROR_INVALID_ARGUMENT;
    }
    try {
      auto& fn = *static_cast<callback_type*>(user_data);
      return fn(*request, *buffer);
    } catch (...) {
      return SSI_ERROR;
    }
  };

  return api->build_sparse_matrix(context, matrix, &builder);
}

}  // namespace ssi

#endif
