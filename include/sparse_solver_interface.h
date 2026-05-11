#ifndef SPARSE_SOLVER_INTERFACE_H
#define SPARSE_SOLVER_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #define SSI_EXPORT __declspec(dllexport)
  #define SSI_IMPORT __declspec(dllimport)
#else
  #define SSI_EXPORT __attribute__((visibility("default")))
  #define SSI_IMPORT
#endif

#ifndef SSI_API
  #define SSI_API SSI_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SSI_ABI_VERSION_MAJOR 1u
#define SSI_ABI_VERSION_MINOR 0u
#define SSI_ABI_VERSION ((SSI_ABI_VERSION_MAJOR << 16u) | SSI_ABI_VERSION_MINOR)

typedef int32_t ssi_status_t;

enum {
  SSI_SUCCESS = 0,
  SSI_ERROR = 1,
  SSI_ERROR_INVALID_ARGUMENT = 2,
  SSI_ERROR_UNSUPPORTED = 3,
  SSI_ERROR_OUT_OF_MEMORY = 4,
  SSI_ERROR_INSUFFICIENT_CAPACITY = 5,
  SSI_ERROR_NUMERIC_FAILURE = 6,
  SSI_ERROR_SINGULAR = 7,
  SSI_ERROR_NOT_IMPLEMENTED = 8
};

typedef uint32_t ssi_dtype_t;
enum {
  SSI_DTYPE_F32 = 0,
  SSI_DTYPE_F64 = 1,
  SSI_DTYPE_C64 = 2,
  SSI_DTYPE_C128 = 3
};

typedef uint32_t ssi_itype_t;
enum {
  SSI_ITYPE_I32 = 0,
  SSI_ITYPE_I64 = 1
};

typedef uint32_t ssi_matrix_order_t;
enum {
  SSI_MATRIX_ROW_MAJOR = 0,
  SSI_MATRIX_COLUMN_MAJOR = 1
};

typedef uint32_t ssi_sparse_format_t;
enum {
  SSI_SPARSE_FORMAT_COO = 0,
  SSI_SPARSE_FORMAT_CSR = 1,
  SSI_SPARSE_FORMAT_CSC = 2
};

typedef uint32_t ssi_range_kind_t;
enum {
  SSI_RANGE_ROWS = 0,
  SSI_RANGE_COLUMNS = 1,
  SSI_RANGE_ENTRIES = 2
};

typedef uint32_t ssi_build_phase_t;
enum {
  SSI_BUILD_COUNT = 0,
  SSI_BUILD_FILL = 1
};

typedef uint32_t ssi_assembly_t;
enum {
  SSI_ASSEMBLY_AUTO = 0,
  SSI_ASSEMBLY_BY_ROW = 1,
  SSI_ASSEMBLY_BY_COLUMN = 2,
  SSI_ASSEMBLY_BY_ENTRY = 3
};

typedef uint32_t ssi_structural_properties_t;
enum {
  SSI_STRUCTURAL_PROPERTY_NONE = 0,
  SSI_STRUCTURAL_PROPERTY_STRONG_HALL = 1u << 0,
  SSI_STRUCTURAL_PROPERTY_SYMMETRIC = 1u << 1
};

typedef uint32_t ssi_numeric_properties_t;
enum {
  SSI_NUMERIC_PROPERTY_NONE = 0,
  SSI_NUMERIC_PROPERTY_SYMMETRIC = 1u << 0,
  SSI_NUMERIC_PROPERTY_HERMITIAN = 1u << 1,
  SSI_NUMERIC_PROPERTY_POSITIVE_DEFINITE = 1u << 2
};

typedef struct {
  float re;
  float im;
} ssi_complex64_t;

typedef struct {
  double re;
  double im;
} ssi_complex128_t;

typedef struct {
  int64_t begin;
  int64_t end;
} ssi_index_range_t;

typedef struct {
  int64_t row_begin;
  int64_t row_end;
  int64_t column_begin;
  int64_t column_end;
} ssi_matrix_range_t;

typedef struct {
  size_t size;
  const void* user_data;
} ssi_config_t;

typedef struct {
  size_t size;
  int64_t rows;
  int64_t columns;
  ssi_dtype_t dtype;
  ssi_matrix_order_t preferred_order;
} ssi_matrix_desc_t;

typedef struct {
  size_t size;
  int64_t rows;
  int64_t columns;
  int64_t nonzeros;
  ssi_itype_t itype;
  ssi_sparse_format_t preferred_format;
  ssi_structural_properties_t properties;
  ssi_assembly_t preferred_assembly;
} ssi_sparse_graph_desc_t;

typedef struct {
  size_t size;
  int64_t nonzeros;
  ssi_dtype_t dtype;
  ssi_numeric_properties_t properties;
  ssi_assembly_t preferred_assembly;
} ssi_sparse_matrix_desc_t;

typedef struct ssi_context ssi_context_t;
typedef struct ssi_matrix ssi_matrix_t;
typedef struct ssi_sparse_graph ssi_sparse_graph_t;
typedef struct ssi_sparse_matrix ssi_sparse_matrix_t;
typedef struct ssi_symbolic_factorization ssi_symbolic_factorization_t;
typedef struct ssi_numeric_factorization ssi_numeric_factorization_t;

typedef struct {
  size_t size;
  ssi_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
} ssi_matrix_build_request_t;

typedef struct {
  size_t size;
  ssi_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
  int64_t leading_dimension;
  union {
    float* f32;
    double* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
    void* raw;
  } values;
} ssi_matrix_build_buffer_t;

typedef struct {
  size_t size;
  ssi_build_phase_t phase;
  ssi_sparse_graph_desc_t graph;
  ssi_sparse_format_t format;
  ssi_range_kind_t range_kind;
  union {
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_sparse_graph_build_request_t;

typedef struct {
  size_t size;
  ssi_build_phase_t phase;
  ssi_sparse_graph_desc_t graph;
  ssi_sparse_format_t format;
  ssi_range_kind_t range_kind;
  int64_t offset_capacity;
  int64_t index_capacity;
  int64_t required_offset_count;
  int64_t required_index_count;
  int64_t offset_count;
  int64_t index_count;
  union {
    int32_t* i32;
    int64_t* i64;
    void* raw;
  } offsets;
  union {
    int32_t* i32;
    int64_t* i64;
    void* raw;
  } row_indices;
  union {
    int32_t* i32;
    int64_t* i64;
    void* raw;
  } column_indices;
} ssi_sparse_graph_build_buffer_t;

typedef struct {
  size_t size;
  ssi_build_phase_t phase;
  ssi_sparse_matrix_desc_t matrix;
  ssi_sparse_format_t format;
  ssi_range_kind_t range_kind;
  union {
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_sparse_matrix_build_request_t;

typedef struct {
  size_t size;
  ssi_build_phase_t phase;
  ssi_sparse_matrix_desc_t matrix;
  ssi_sparse_format_t format;
  ssi_range_kind_t range_kind;
  int64_t value_capacity;
  int64_t required_value_count;
  int64_t value_count;
  union {
    float* f32;
    double* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
    void* raw;
  } values;
} ssi_sparse_matrix_build_buffer_t;

typedef ssi_status_t (*ssi_matrix_build_fn)(
    const ssi_matrix_build_request_t* request,
    ssi_matrix_build_buffer_t* buffer,
    void* user_data);

typedef ssi_status_t (*ssi_sparse_graph_build_fn)(
    const ssi_sparse_graph_build_request_t* request,
    ssi_sparse_graph_build_buffer_t* buffer,
    void* user_data);

typedef ssi_status_t (*ssi_sparse_matrix_build_fn)(
    const ssi_sparse_matrix_build_request_t* request,
    ssi_sparse_matrix_build_buffer_t* buffer,
    void* user_data);

typedef struct {
  size_t size;
  ssi_matrix_build_fn build;
  void* user_data;
} ssi_matrix_builder_t;

typedef struct {
  size_t size;
  ssi_sparse_graph_build_fn build;
  void* user_data;
} ssi_sparse_graph_builder_t;

typedef struct {
  size_t size;
  ssi_sparse_matrix_build_fn build;
  void* user_data;
} ssi_sparse_matrix_builder_t;

typedef struct {
  size_t size;
  ssi_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
} ssi_matrix_read_request_t;

typedef struct {
  size_t size;
  ssi_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
  int64_t leading_dimension;
  union {
    const float* f32;
    const double* f64;
    const ssi_complex64_t* c64;
    const ssi_complex128_t* c128;
    const void* raw;
  } values;
} ssi_matrix_read_buffer_t;

typedef ssi_status_t (*ssi_matrix_read_fn)(
    const ssi_matrix_read_request_t* request,
    const ssi_matrix_read_buffer_t* buffer,
    void* user_data);

typedef struct {
  size_t size;
  ssi_matrix_read_fn read;
  void* user_data;
} ssi_matrix_reader_t;

typedef struct {
  size_t size;
  uint32_t abi_version;
  uint64_t capabilities;

  ssi_status_t (*create_context)(
      const ssi_config_t* config,
      ssi_context_t** out_context);
  void (*destroy_context)(ssi_context_t** context);

  ssi_status_t (*create_matrix)(
      ssi_context_t* context,
      const ssi_matrix_desc_t* desc,
      ssi_matrix_t** out_matrix);
  ssi_status_t (*build_matrix)(
      ssi_context_t* context,
      ssi_matrix_t* matrix,
      const ssi_matrix_builder_t* builder);
  ssi_status_t (*read_matrix)(
      ssi_context_t* context,
      const ssi_matrix_t* matrix,
      const ssi_matrix_reader_t* reader);
  void (*destroy_matrix)(ssi_context_t* context, ssi_matrix_t** matrix);

  ssi_status_t (*create_sparse_graph)(
      ssi_context_t* context,
      const ssi_sparse_graph_desc_t* desc,
      ssi_sparse_graph_t** out_graph);
  ssi_status_t (*build_sparse_graph)(
      ssi_context_t* context,
      ssi_sparse_graph_t* graph,
      const ssi_sparse_graph_builder_t* builder);
  void (*destroy_sparse_graph)(ssi_context_t* context, ssi_sparse_graph_t** graph);

  ssi_status_t (*create_sparse_matrix)(
      ssi_context_t* context,
      ssi_sparse_graph_t* graph,
      const ssi_sparse_matrix_desc_t* desc,
      ssi_sparse_matrix_t** out_matrix);
  ssi_status_t (*build_sparse_matrix)(
      ssi_context_t* context,
      ssi_sparse_matrix_t* matrix,
      const ssi_sparse_matrix_builder_t* builder);
  void (*destroy_sparse_matrix)(ssi_context_t* context, ssi_sparse_matrix_t** matrix);

  ssi_status_t (*create_symbolic_factorization)(
      ssi_context_t* context,
      ssi_sparse_graph_t* graph,
      ssi_symbolic_factorization_t** out_factorization);
  void (*destroy_symbolic_factorization)(
      ssi_context_t* context,
      ssi_symbolic_factorization_t** factorization);

  ssi_status_t (*create_numeric_factorization)(
      ssi_context_t* context,
      ssi_symbolic_factorization_t* symbolic,
      ssi_sparse_matrix_t* matrix,
      ssi_numeric_factorization_t** out_factorization);
  void (*destroy_numeric_factorization)(
      ssi_context_t* context,
      ssi_numeric_factorization_t** factorization);

  ssi_status_t (*solve)(
      ssi_context_t* context,
      const ssi_numeric_factorization_t* factorization,
      const ssi_matrix_t* input,
      ssi_matrix_t* output);

  ssi_status_t (*await)(ssi_context_t* context);
  const char* (*last_error)(ssi_context_t* context);

  /*
   * Future versions may add sparse low-rank update operations for refreshing
   * numeric factorizations without recomputing symbolic analysis.
   */
  void* reserved[16];
} ssi_api_v1_t;

typedef ssi_status_t (*ssi_get_api_v1_fn)(ssi_api_v1_t* out_api, size_t out_size);

SSI_API const char* ssi_version(void);
SSI_API ssi_status_t ssi_get_api_v1(ssi_api_v1_t* out_api, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif
