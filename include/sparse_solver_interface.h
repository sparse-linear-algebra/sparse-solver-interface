#ifndef SPARSE_SOLVER_INTERFACE_H
#define SPARSE_SOLVER_INTERFACE_H

#ifdef _WIN32
  #ifdef sparse_solver_interface_EXPORTS
    #define ssi_api __declspec(dllexport)
  #else
    #define ssi_api __declspec(dllimport)
  #endif
#else
  #define ssi_api
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sparse Solver Interface (SSI)
 *
 * SSI exposes a C ABI for sparse solver implementations that can be loaded at
 * runtime. Implementations own opaque logical objects. Callers construct and
 * read those objects through builder and reader callbacks over requested ranges.
 *
 * The standard builder/reader interface is host-callable. Future extension
 * interfaces may support borrowed/native buffers so GPU-native or other
 * backend-native producers can avoid staging through host callbacks.
 *
 * TODO: Extend range descriptors for multiprocess contexts. The current core
 * descriptors assume a large directly addressable host address space for the
 * callback execution path. MPI-style pre-distributed construction, root-owned
 * input scatter, and rank-local empty logical object participation need explicit
 * representation.
 */

ssi_api const char* ssi_version(void);

typedef enum{
  ssi_success = 0,
  ssi_error = 1
} ssi_status_t;

typedef float ssi_float32_t;
typedef double ssi_float64_t;

typedef struct{
  ssi_float32_t re;
  ssi_float32_t im;
} ssi_complex64_t;

typedef struct{
  ssi_float64_t re;
  ssi_float64_t im;
} ssi_complex128_t;

typedef int32_t ssi_int32_t;
typedef int64_t ssi_int64_t;

typedef enum{
  ssi_f32 = 0,
  ssi_f64 = 1,
  ssi_c64 = 2,
  ssi_c128 = 3
} ssi_dtype_t;

typedef enum{
  ssi_i32 = 0,
  ssi_i64 = 1
} ssi_itype_t;

typedef enum{
  ssi_row_major = 0,
  ssi_col_major = 1
} ssi_matrix_order_t;

typedef enum{
  ssi_coo = 0,
  ssi_csr = 1,
  ssi_csc = 2
} ssi_sparse_format_t;

typedef enum{
  ssi_structural_property_none = 0,
  ssi_structural_property_strong_hall = 1u << 0,
  ssi_structural_property_symmetric = 1u << 1
} ssi_structural_property_t;

/*
 * Numeric property hints are independent bit flags. In SSI,
 * ssi_numeric_property_positive_definite means the quadratic form is positive
 * for every nonzero vector, equivalently that the symmetric/Hermitian part is
 * positive definite. It does not imply ssi_numeric_property_symmetric or
 * ssi_numeric_property_hermitian.
 */
typedef enum{
  ssi_numeric_property_none = 0,
  ssi_numeric_property_symmetric = 1u << 0,
  ssi_numeric_property_hermitian = 1u << 1,
  ssi_numeric_property_positive_definite = 1u << 2
} ssi_numeric_property_t;

typedef enum{
  ssi_assembly_auto = 0,
  ssi_assembly_by_row = 1,
  ssi_assembly_by_column = 2,
  ssi_assembly_by_entry = 3
} ssi_assembly_t;

typedef enum{
  ssi_range_rows = 0,
  ssi_range_columns = 1,
  ssi_range_entries = 2,
  ssi_range_dense_slice = 3
} ssi_range_kind_t;

typedef struct{
  ssi_int64_t begin;
  ssi_int64_t end;
} ssi_index_range_t;

typedef struct{
  ssi_int64_t row_begin;
  ssi_int64_t row_end;
  ssi_int64_t col_begin;
  ssi_int64_t col_end;
} ssi_matrix_range_t;

typedef struct{
  ssi_sparse_format_t format;
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_int64_t nnz;
  ssi_itype_t itype;
  ssi_structural_property_t properties;
  ssi_assembly_t preferred_assembly;
} ssi_graph_desc_t;

typedef struct{
  ssi_int64_t nnz;
  ssi_dtype_t dtype;
  ssi_numeric_property_t properties;
  ssi_assembly_t preferred_assembly;
} ssi_numeric_desc_t;

typedef struct{
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_dtype_t dtype;
  ssi_matrix_order_t preferred_order;
} ssi_dense_matrix_desc_t;

typedef struct{
  ssi_graph_desc_t graph;
  ssi_range_kind_t range_kind;
  union{
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_graph_build_request_t;

typedef struct{
  ssi_graph_desc_t graph;
  ssi_range_kind_t range_kind;
  ssi_int64_t nnz_capacity;
  ssi_int64_t nnz;
  union{
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } offsets;
  union{
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } row_ids;
  union{
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } column_ids;
} ssi_graph_build_buffer_t;

typedef struct{
  ssi_numeric_desc_t numeric;
  ssi_range_kind_t range_kind;
  union{
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_numeric_build_request_t;

typedef struct{
  ssi_numeric_desc_t numeric;
  ssi_range_kind_t range_kind;
  ssi_int64_t value_capacity;
  ssi_int64_t value_count;
  union{
    ssi_float32_t* f32;
    ssi_float64_t* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
  } values;
} ssi_numeric_build_buffer_t;

typedef struct{
  ssi_dense_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
} ssi_dense_matrix_build_request_t;

typedef struct{
  ssi_dense_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
  ssi_int64_t ld;
  union{
    ssi_float32_t* f32;
    ssi_float64_t* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
  } values;
} ssi_dense_matrix_build_buffer_t;

typedef struct{
  ssi_graph_desc_t graph;
  ssi_range_kind_t range_kind;
  union{
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_graph_read_request_t;

typedef struct{
  ssi_graph_desc_t graph;
  ssi_range_kind_t range_kind;
  ssi_int64_t nnz;
  union{
    const ssi_int32_t* i32;
    const ssi_int64_t* i64;
  } offsets;
  union{
    const ssi_int32_t* i32;
    const ssi_int64_t* i64;
  } row_ids;
  union{
    const ssi_int32_t* i32;
    const ssi_int64_t* i64;
  } column_ids;
} ssi_graph_read_buffer_t;

typedef struct{
  ssi_numeric_desc_t numeric;
  ssi_range_kind_t range_kind;
  union{
    ssi_index_range_t rows;
    ssi_index_range_t columns;
    ssi_index_range_t entries;
  } range;
} ssi_numeric_read_request_t;

typedef struct{
  ssi_numeric_desc_t numeric;
  ssi_range_kind_t range_kind;
  ssi_int64_t value_count;
  union{
    const ssi_float32_t* f32;
    const ssi_float64_t* f64;
    const ssi_complex64_t* c64;
    const ssi_complex128_t* c128;
  } values;
} ssi_numeric_read_buffer_t;

typedef struct{
  ssi_dense_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
} ssi_dense_matrix_read_request_t;

typedef struct{
  ssi_dense_matrix_desc_t matrix;
  ssi_matrix_range_t range;
  ssi_matrix_order_t order;
  ssi_int64_t ld;
  union{
    const ssi_float32_t* f32;
    const ssi_float64_t* f64;
    const ssi_complex64_t* c64;
    const ssi_complex128_t* c128;
  } values;
} ssi_dense_matrix_read_buffer_t;

typedef struct ssi_context ssi_context_t;
typedef struct ssi_logical_graph ssi_logical_graph_t;
typedef struct ssi_logical_numeric ssi_logical_numeric_t;
typedef struct ssi_logical_dense_matrix ssi_logical_dense_matrix_t;
typedef struct ssi_logical_symbolic_factorization ssi_logical_symbolic_factorization_t;
typedef struct ssi_logical_numeric_factorization ssi_logical_numeric_factorization_t;

typedef ssi_status_t (*ssi_graph_build_fn)(const ssi_graph_build_request_t* request,ssi_graph_build_buffer_t* buffer,void* user_data);
typedef ssi_status_t (*ssi_numeric_build_fn)(const ssi_numeric_build_request_t* request,ssi_numeric_build_buffer_t* buffer,void* user_data);
typedef ssi_status_t (*ssi_dense_matrix_build_fn)(const ssi_dense_matrix_build_request_t* request,ssi_dense_matrix_build_buffer_t* buffer,void* user_data);

typedef ssi_status_t (*ssi_graph_read_fn)(const ssi_graph_read_request_t* request,const ssi_graph_read_buffer_t* buffer,void* user_data);
typedef ssi_status_t (*ssi_numeric_read_fn)(const ssi_numeric_read_request_t* request,const ssi_numeric_read_buffer_t* buffer,void* user_data);
typedef ssi_status_t (*ssi_dense_matrix_read_fn)(const ssi_dense_matrix_read_request_t* request,const ssi_dense_matrix_read_buffer_t* buffer,void* user_data);

typedef struct{
  ssi_graph_build_fn build;
  void* user_data;
} ssi_graph_builder_t;

typedef struct{
  ssi_numeric_build_fn build;
  void* user_data;
} ssi_numeric_builder_t;

typedef struct{
  ssi_dense_matrix_build_fn build;
  void* user_data;
} ssi_dense_matrix_builder_t;

typedef struct{
  ssi_graph_read_fn read;
  void* user_data;
} ssi_graph_reader_t;

typedef struct{
  ssi_numeric_read_fn read;
  void* user_data;
} ssi_numeric_reader_t;

typedef struct{
  ssi_dense_matrix_read_fn read;
  void* user_data;
} ssi_dense_matrix_reader_t;

typedef struct{
  ssi_context_t* (*create_context)(void*);

  ssi_logical_graph_t* (*create_logical_graph)(ssi_context_t* context,const ssi_graph_desc_t* desc);
  ssi_logical_numeric_t* (*create_logical_numeric)(ssi_context_t* context,ssi_logical_graph_t* graph,const ssi_numeric_desc_t* desc);
  ssi_logical_dense_matrix_t* (*create_logical_dense_matrix)(ssi_context_t* context,const ssi_dense_matrix_desc_t* desc);

  ssi_status_t (*build_logical_graph)(ssi_context_t* context,ssi_logical_graph_t* graph,const ssi_graph_builder_t* builder);
  ssi_status_t (*build_logical_numeric)(ssi_context_t* context,ssi_logical_numeric_t* numeric,const ssi_numeric_builder_t* builder);
  ssi_status_t (*build_logical_dense_matrix)(ssi_context_t* context,ssi_logical_dense_matrix_t* matrix,const ssi_dense_matrix_builder_t* builder);

  ssi_status_t (*read_logical_graph)(ssi_context_t* context,const ssi_logical_graph_t* graph,const ssi_graph_reader_t* reader);
  ssi_status_t (*read_logical_numeric)(ssi_context_t* context,const ssi_logical_numeric_t* numeric,const ssi_numeric_reader_t* reader);
  ssi_status_t (*read_logical_dense_matrix)(ssi_context_t* context,const ssi_logical_dense_matrix_t* matrix,const ssi_dense_matrix_reader_t* reader);

  /*
   * Factorization lifecycle.
   *
   * TODO: Consider an interface for updating an existing numeric factorization
   * with sparse low-rank changes, e.g. a function that takes an existing
   * numeric factorization plus logical update data and produces or refreshes a
   * numeric factorization without recomputing symbolic analysis.
   */
  ssi_logical_symbolic_factorization_t* (*create_symbolic_factorization)(ssi_context_t* context,ssi_logical_graph_t* graph);
  ssi_logical_numeric_factorization_t* (*create_numeric_factorization)(ssi_context_t* context,ssi_logical_symbolic_factorization_t* symbolic,ssi_logical_numeric_t* numeric);

  ssi_status_t (*solve)(ssi_context_t* context,const ssi_logical_numeric_factorization_t* factorization,const ssi_logical_dense_matrix_t* rhs,ssi_logical_dense_matrix_t* solution);

  /*
   * Synchronize the host with work owned by the SSI context.
   */
  ssi_status_t (*await)(ssi_context_t* context);

  void (*destroy_logical_graph)(ssi_context_t* context,ssi_logical_graph_t** graph);
  void (*destroy_logical_numeric)(ssi_context_t* context,ssi_logical_numeric_t** numeric);
  void (*destroy_logical_dense_matrix)(ssi_context_t* context,ssi_logical_dense_matrix_t** matrix);
  void (*destroy_symbolic_factorization)(ssi_context_t* context,ssi_logical_symbolic_factorization_t** symbolic);
  void (*destroy_numeric_factorization)(ssi_context_t* context,ssi_logical_numeric_factorization_t** numeric);
  void (*destroy_context)(ssi_context_t**);
} ssi_api_t;

#ifdef __cplusplus
}
#endif

#endif
