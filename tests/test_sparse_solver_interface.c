#include "acutest.h"
#include "sparse_solver_interface.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

static void test_build_link_and_run(void)
{
  ssi_api_v1_t api = {0};

  TEST_CHECK(ssi_version() != 0);
  TEST_CHECK(ssi_get_api_v1(&api, sizeof(api)) == SSI_SUCCESS);
  TEST_CHECK(api.size == sizeof(api));
  TEST_CHECK(api.abi_version == SSI_ABI_VERSION);
}

static void test_rejects_too_small_api_buffer(void)
{
  ssi_api_v1_t api = {0};

  TEST_CHECK(ssi_get_api_v1(&api, 1) == SSI_ERROR_INVALID_ARGUMENT);
}

static void test_dlopen_get_api(void)
{
#ifdef _WIN32
  TEST_SKIP("dlopen test is POSIX-only");
#else
  void* library = dlopen(SSI_TEST_LIBRARY_PATH, RTLD_NOW | RTLD_LOCAL);
  ssi_get_api_v1_fn get_api;
  ssi_api_v1_t api = {0};

  TEST_CHECK(library != 0);
  if (library == 0) {
    return;
  }

  get_api = (ssi_get_api_v1_fn)dlsym(library, "ssi_get_api_v1");
  TEST_CHECK(get_api != 0);
  TEST_CHECK(get_api(&api, sizeof(api)) == SSI_SUCCESS);
  TEST_CHECK(api.abi_version == SSI_ABI_VERSION);

  dlclose(library);
#endif
}

static ssi_status_t count_then_fill_sparse_graph(
    const ssi_sparse_graph_build_request_t* request,
    ssi_sparse_graph_build_buffer_t* buffer,
    void* user_data)
{
  (void)user_data;

  if (request == 0 || buffer == 0) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }

  if (request->phase == SSI_BUILD_COUNT) {
    buffer->required_offset_count = 3;
    buffer->required_index_count = 2;
    return SSI_SUCCESS;
  }

  if (request->phase == SSI_BUILD_FILL) {
    if (buffer->offset_capacity < 3 || buffer->index_capacity < 2) {
      buffer->required_offset_count = 3;
      buffer->required_index_count = 2;
      return SSI_ERROR_INSUFFICIENT_CAPACITY;
    }

    buffer->offsets.i64[0] = 0;
    buffer->offsets.i64[1] = 1;
    buffer->offsets.i64[2] = 2;
    buffer->column_indices.i64[0] = 0;
    buffer->column_indices.i64[1] = 1;
    buffer->offset_count = 3;
    buffer->index_count = 2;
    return SSI_SUCCESS;
  }

  return SSI_ERROR_INVALID_ARGUMENT;
}

static void test_sparse_graph_builder_count_and_fill(void)
{
  ssi_sparse_graph_build_request_t request = {0};
  ssi_sparse_graph_build_buffer_t buffer = {0};
  int64_t offsets[3] = {0};
  int64_t columns[2] = {0};

  request.size = sizeof(request);
  request.phase = SSI_BUILD_COUNT;
  request.graph.size = sizeof(request.graph);
  request.graph.rows = 2;
  request.graph.columns = 2;
  request.graph.nonzeros = 2;
  request.graph.itype = SSI_ITYPE_I64;
  request.format = SSI_SPARSE_FORMAT_CSR;
  request.range_kind = SSI_RANGE_ROWS;
  request.range.rows.begin = 0;
  request.range.rows.end = 2;

  buffer.size = sizeof(buffer);
  buffer.phase = SSI_BUILD_COUNT;
  buffer.graph = request.graph;
  buffer.format = request.format;
  buffer.range_kind = request.range_kind;

  TEST_CHECK(count_then_fill_sparse_graph(&request, &buffer, 0) == SSI_SUCCESS);
  TEST_CHECK(buffer.required_offset_count == 3);
  TEST_CHECK(buffer.required_index_count == 2);

  request.phase = SSI_BUILD_FILL;
  buffer.phase = SSI_BUILD_FILL;
  buffer.offset_capacity = 3;
  buffer.index_capacity = 2;
  buffer.offsets.i64 = offsets;
  buffer.column_indices.i64 = columns;

  TEST_CHECK(count_then_fill_sparse_graph(&request, &buffer, 0) == SSI_SUCCESS);
  TEST_CHECK(buffer.offset_count == 3);
  TEST_CHECK(buffer.index_count == 2);
  TEST_CHECK(offsets[2] == 2);
  TEST_CHECK(columns[1] == 1);
}

TEST_LIST = {
  { "build_link_and_run", test_build_link_and_run },
  { "rejects_too_small_api_buffer", test_rejects_too_small_api_buffer },
  { "dlopen_get_api", test_dlopen_get_api },
  { "sparse_graph_builder_count_and_fill", test_sparse_graph_builder_count_and_fill },
  { 0, 0 }
};
