# Implementing an SSI Solver

SSI has one binary contract: the C ABI in `include/sparse_solver_interface.h`.
The C++ headers are header-only helpers for implementing or consuming that C
ABI; C++ types must not cross the shared-library boundary.

## Direct C Implementation

A compatible shared object exports these C symbols:

```c
#define SSI_API SSI_EXPORT
#include "sparse_solver_interface.h"

const char* ssi_version(void);
ssi_status_t ssi_get_api_v1(ssi_api_v1_t* out_api, size_t out_size);
```

`ssi_get_api_v1` fills an `ssi_api_v1_t` table with function pointers owned by
the solver library:

```c
static ssi_status_t my_create_context(
    const ssi_config_t* config,
    ssi_context_t** out_context)
{
  (void)config;
  if (out_context == 0) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }

  *out_context = (ssi_context_t*)my_context_create();
  return *out_context ? SSI_SUCCESS : SSI_ERROR_OUT_OF_MEMORY;
}

ssi_status_t ssi_get_api_v1(ssi_api_v1_t* out_api, size_t out_size)
{
  if (out_api == 0 || out_size < offsetof(ssi_api_v1_t, reserved)) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }

  memset(out_api, 0, out_size);
  out_api->size = sizeof(*out_api);
  out_api->abi_version = SSI_ABI_VERSION;
  out_api->create_context = my_create_context;
  out_api->destroy_context = my_destroy_context;
  out_api->create_sparse_graph = my_create_sparse_graph;
  out_api->build_sparse_graph = my_build_sparse_graph;
  return SSI_SUCCESS;
}
```

Build it as a shared library and ensure `ssi_get_api_v1` is visible. On ELF
platforms, a consumer should be able to find it with:

```c
void* library = dlopen("./libmy_solver.so", RTLD_NOW | RTLD_LOCAL);
ssi_get_api_v1_fn get_api = (ssi_get_api_v1_fn)dlsym(library, "ssi_get_api_v1");
```

## Sparse Graph Builders

Every implementation should support the portable host-addressable builder path.
The solver drives construction by calling the builder over logical global
ranges. Sparse builders use two phases.

In `SSI_BUILD_COUNT`, the solver requests a range with null or zero-capacity
buffers. The callback reports required storage:

```c
buffer->required_offset_count = row_count + 1;
buffer->required_index_count = nonzeros_in_rows;
return SSI_SUCCESS;
```

In `SSI_BUILD_FILL`, the solver provides buffers with enough capacity. The
callback writes offsets and indices, then reports actual counts:

```c
buffer->offset_count = row_count + 1;
buffer->index_count = nonzeros_in_rows;
return SSI_SUCCESS;
```

If a fill buffer is too small, return `SSI_ERROR_INSUFFICIENT_CAPACITY` and set
the required counts.

For `SSI_SPARSE_FORMAT_CSR` with `SSI_RANGE_ROWS`:

- The range is global rows `[begin, end)`.
- `offsets` has `end - begin + 1` entries.
- `offsets[0]` is zero.
- `column_indices` contains global zero-based column indices.
- `row_indices` is unused.

For `SSI_SPARSE_FORMAT_CSC` with `SSI_RANGE_COLUMNS`:

- The range is global columns `[begin, end)`.
- `offsets` has `end - begin + 1` entries.
- `row_indices` contains global zero-based row indices.
- `column_indices` is unused.

For `SSI_SPARSE_FORMAT_COO` with `SSI_RANGE_ENTRIES`:

- The range is logical entries `[begin, end)`.
- `offsets` is unused.
- `row_indices` and `column_indices` contain global zero-based coordinates.

All ranges are half-open. Dimensions and counts use `int64_t`; index arrays use
the graph descriptor's `itype`.

## Sparse Matrix Builders

Sparse matrix builders follow the same count/fill pattern. In the count phase,
set `required_value_count`. In the fill phase, write values and set
`value_count`.

Sparse matrix values should follow the graph entry order chosen by the
implementation for the requested range and format.

## C++ Implementation

The easier C++ path is to implement `ssi::solver` from `include/ssi/ssi.hpp`
inside your shared library and export the C ABI with `ssi::export_api`.

```cpp
#define SSI_API SSI_EXPORT
#include "sparse_solver_interface.hpp"

class my_solver final : public ssi::solver {
public:
  ssi_status_t create_context(
      const ssi_config_t* config,
      ssi_context_t** out_context) noexcept override;

  void destroy_context(ssi_context_t** context) noexcept override;

  ssi_status_t create_sparse_graph(
      ssi_context_t* context,
      const ssi_sparse_graph_desc_t& desc,
      ssi_sparse_graph_t** out_graph) noexcept override;

  ssi_status_t build_sparse_graph(
      ssi_context_t* context,
      ssi_sparse_graph_t* graph,
      ssi::sparse_graph_builder& builder) noexcept override;

  const char* last_error(ssi_context_t* context) noexcept override;

  /* Implement the remaining pure virtual methods. */
};

extern "C" SSI_EXPORT const char* ssi_version(void)
{
  return "0.1.0";
}

extern "C" SSI_EXPORT ssi_status_t ssi_get_api_v1(
    ssi_api_v1_t* out_api,
    size_t out_size)
{
  return ssi::export_api<my_solver>(out_api, out_size);
}
```

The C++ adapter catches exceptions before they cross the C ABI and translates
them to `ssi_status_t`. Implementations should still prefer `noexcept` methods
and use `last_error` for diagnostic text.

The header-only helper means implementors may use their own C++ compiler and
standard library internally. Consumers only see the stable C entry points and
the `ssi_api_v1_t` function table.
