#ifndef SPARSE_SOLVER_INTERFACE_C_H
#define SPARSE_SOLVER_INTERFACE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SSI_ABI_VERSION_MAJOR 0u
#define SSI_ABI_VERSION_MINOR 1u

#if defined(_WIN32)
  #define SSI_EXPORT __declspec(dllexport)
#else
  #define SSI_EXPORT __attribute__((visibility("default")))
#endif

typedef enum ssi_status_t {
  SSI_STATUS_OK = 0,
  SSI_STATUS_INVALID_ARGUMENT = 1,
  SSI_STATUS_OUT_OF_RANGE = 2,
  SSI_STATUS_UNSUPPORTED = 3,
  SSI_STATUS_EXCEPTION = 4
} ssi_status_t;

typedef enum ssi_dtype_t {
  SSI_DTYPE_FP32 = 0,
  SSI_DTYPE_FP64 = 1,
  SSI_DTYPE_C64 = 2,
  SSI_DTYPE_C128 = 3
} ssi_dtype_t;

typedef enum ssi_itype_t {
  SSI_ITYPE_I32 = 0,
  SSI_ITYPE_I64 = 1
} ssi_itype_t;

typedef enum ssi_matrix_order_t {
  SSI_MATRIX_ORDER_COL_MAJOR = 0,
  SSI_MATRIX_ORDER_ROW_MAJOR = 1
} ssi_matrix_order_t;

typedef enum ssi_graph_orientation_t {
  SSI_GRAPH_ORIENTATION_ROW = 0,
  SSI_GRAPH_ORIENTATION_COLUMN = 1
} ssi_graph_orientation_t;

typedef enum ssi_property_state_t {
  SSI_PROPERTY_STATE_UNKNOWN = 0,
  SSI_PROPERTY_STATE_KNOWN_FALSE = 1,
  SSI_PROPERTY_STATE_KNOWN_TRUE = 2
} ssi_property_state_t;

typedef enum ssi_symmetric_storage_t {
  SSI_SYMMETRIC_STORAGE_UNSYMMETRIC = 0,
  SSI_SYMMETRIC_STORAGE_FULL = 1,
  SSI_SYMMETRIC_STORAGE_LOWER = 2,
  SSI_SYMMETRIC_STORAGE_UPPER = 3
} ssi_symmetric_storage_t;

typedef struct ssi_sparse_problem_properties_t {
  int64_t nrows;
  int64_t ncols;
  ssi_graph_orientation_t orientation;
  ssi_itype_t itype;
  ssi_dtype_t dtype;
  ssi_property_state_t structurally_symmetric;
  ssi_property_state_t numerically_symmetric;
  ssi_property_state_t positive_definite;
  ssi_property_state_t negative_definite;
  ssi_property_state_t full_column_rank;
  ssi_property_state_t full_row_rank;
  ssi_property_state_t nonsingular;
  ssi_property_state_t strong_hall;
  ssi_symmetric_storage_t symmetric_storage;
} ssi_sparse_problem_properties_t;

typedef struct ssi_matrix_view_t {
  ssi_matrix_order_t order;
  ssi_dtype_t dtype;
  int64_t rbeg;
  int64_t rend;
  int64_t cbeg;
  int64_t cend;
  int64_t ld;
  void* data;
} ssi_matrix_view_t;

typedef struct ssi_graph_count_builder_t {
  ssi_graph_orientation_t orientation;
  ssi_itype_t itype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  void* counts;
} ssi_graph_count_builder_t;

typedef struct ssi_graph_edge_builder_t {
  ssi_graph_orientation_t orientation;
  ssi_itype_t itype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  const void* offsets;
  void* ids;
} ssi_graph_edge_builder_t;

typedef struct ssi_compressed_graph_view_t {
  ssi_graph_orientation_t orientation;
  ssi_itype_t itype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  const void* offsets;
  const void* ids;
} ssi_compressed_graph_view_t;

typedef struct ssi_sparse_values_view_t {
  ssi_dtype_t dtype;
  int64_t nedges;
  const void* values;
} ssi_sparse_values_view_t;

typedef struct ssi_sparse_value_builder_t {
  ssi_graph_orientation_t orientation;
  ssi_itype_t itype;
  ssi_dtype_t dtype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  const void* offsets;
  const void* ids;
  void* values;
} ssi_sparse_value_builder_t;

typedef struct ssi_context_t* ssi_context_h;
typedef struct ssi_matrix_t* ssi_matrix_h;
typedef struct ssi_sparse_problem_t* ssi_sparse_problem_h;
typedef struct ssi_graph_t* ssi_graph_h;
typedef struct ssi_sparse_matrix_t* ssi_sparse_matrix_h;
typedef struct ssi_symbolic_t* ssi_symbolic_h;
typedef struct ssi_numeric_factorization_t* ssi_numeric_factorization_h;

typedef ssi_status_t (*ssi_matrix_view_callback_t)(
  ssi_matrix_view_t* view,
  void* user_data);
typedef ssi_status_t (*ssi_const_matrix_view_callback_t)(
  const ssi_matrix_view_t* view,
  void* user_data);
typedef ssi_status_t (*ssi_graph_count_callback_t)(
  ssi_graph_count_builder_t* builder,
  void* user_data);
typedef ssi_status_t (*ssi_graph_edge_callback_t)(
  ssi_graph_edge_builder_t* builder,
  void* user_data);
typedef ssi_status_t (*ssi_sparse_value_callback_t)(
  ssi_sparse_value_builder_t* builder,
  void* user_data);
typedef ssi_status_t (*ssi_const_sparse_value_callback_t)(
  const ssi_sparse_value_builder_t* builder,
  void* user_data);

typedef struct ssi_plugin_api_t {
  uint32_t abi_version_major;
  uint32_t abi_version_minor;
  size_t struct_size;

  const char* (*last_error_message)(void);

  ssi_status_t (*create_context)(ssi_context_h* out_context);

  void (*context_release)(ssi_context_h context);
  ssi_status_t (*context_make_matrix)(
    ssi_context_h context,
    ssi_dtype_t dtype,
    ssi_matrix_h* out_matrix);
  ssi_status_t (*context_make_graph)(
    ssi_context_h context,
    ssi_itype_t itype,
    ssi_graph_h* out_graph);
  ssi_status_t (*context_make_sparse_problem)(
    ssi_context_h context,
    const ssi_sparse_problem_properties_t* properties,
    ssi_sparse_problem_h* out_problem);

  void (*matrix_release)(ssi_matrix_h matrix);
  ssi_status_t (*matrix_nrows)(ssi_matrix_h matrix,int64_t* out_nrows);
  ssi_status_t (*matrix_ncols)(ssi_matrix_h matrix,int64_t* out_ncols);
  ssi_status_t (*matrix_dtype)(ssi_matrix_h matrix,ssi_dtype_t* out_dtype);
  ssi_status_t (*matrix_preallocate)(
    ssi_matrix_h matrix,
    int64_t nrows,
    int64_t ncols);
  ssi_status_t (*matrix_borrow_matrix_view)(
    ssi_matrix_h matrix,
    const ssi_matrix_view_t* view);
  ssi_status_t (*matrix_build_from_host)(
    ssi_matrix_h matrix,
    ssi_matrix_view_callback_t builder,
    void* user_data);
  ssi_status_t (*matrix_read_to_host)(
    ssi_matrix_h matrix,
    ssi_const_matrix_view_callback_t reader,
    void* user_data);

  void (*sparse_problem_release)(ssi_sparse_problem_h problem);
  ssi_status_t (*sparse_problem_properties)(
    ssi_sparse_problem_h problem,
    ssi_sparse_problem_properties_t* out_properties);
  ssi_status_t (*sparse_problem_assert_properties)(
    ssi_sparse_problem_h problem,
    const ssi_sparse_problem_properties_t* properties);
  ssi_status_t (*sparse_problem_compute_missing_properties)(
    ssi_sparse_problem_h problem);
  ssi_status_t (*sparse_problem_make_graph)(
    ssi_sparse_problem_h problem,
    ssi_graph_h* out_graph);
  ssi_status_t (*sparse_problem_make_sparse_matrix)(
    ssi_sparse_problem_h problem,
    ssi_sparse_matrix_h* out_matrix);
  ssi_status_t (*sparse_problem_make_symbolic_analysis)(
    ssi_sparse_problem_h problem,
    ssi_symbolic_h* out_symbolic);

  void (*graph_release)(ssi_graph_h graph);
  ssi_status_t (*graph_itype)(ssi_graph_h graph,ssi_itype_t* out_itype);
  ssi_status_t (*graph_nrows)(ssi_graph_h graph,int64_t* out_nrows);
  ssi_status_t (*graph_ncols)(ssi_graph_h graph,int64_t* out_ncols);
  ssi_status_t (*graph_nedges)(ssi_graph_h graph,int64_t* out_nedges);
  ssi_status_t (*graph_build_from_host)(
    ssi_graph_h graph,
    int64_t nrows,
    int64_t ncols,
    ssi_graph_orientation_t orientation,
    ssi_graph_count_callback_t count_builder,
    void* count_user_data,
    ssi_graph_edge_callback_t edge_builder,
    void* edge_user_data);
  ssi_status_t (*graph_borrow_compressed_graph_view)(
    ssi_graph_h graph,
    const ssi_compressed_graph_view_t* view);
  ssi_status_t (*graph_make_sparse_matrix)(
    ssi_graph_h graph,
    ssi_sparse_matrix_h* out_matrix);
  ssi_status_t (*graph_make_symbolic_analysis)(
    ssi_graph_h graph,
    ssi_symbolic_h* out_symbolic);

  void (*sparse_matrix_release)(ssi_sparse_matrix_h matrix);
  ssi_status_t (*sparse_matrix_nrows)(
    ssi_sparse_matrix_h matrix,
    int64_t* out_nrows);
  ssi_status_t (*sparse_matrix_ncols)(
    ssi_sparse_matrix_h matrix,
    int64_t* out_ncols);
  ssi_status_t (*sparse_matrix_dtype)(
    ssi_sparse_matrix_h matrix,
    ssi_dtype_t* out_dtype);
  ssi_status_t (*sparse_matrix_build_from_host)(
    ssi_sparse_matrix_h matrix,
    ssi_dtype_t dtype,
    ssi_graph_orientation_t orientation,
    ssi_sparse_value_callback_t builder,
    void* user_data);
  ssi_status_t (*sparse_matrix_read_to_host)(
    ssi_sparse_matrix_h matrix,
    ssi_graph_orientation_t orientation,
    ssi_const_sparse_value_callback_t reader,
    void* user_data);
  ssi_status_t (*sparse_matrix_borrow_sparse_values_view)(
    ssi_sparse_matrix_h matrix,
    const ssi_sparse_values_view_t* view);

  void (*symbolic_release)(ssi_symbolic_h symbolic);
  ssi_status_t (*symbolic_make_numeric_factorization)(
    ssi_symbolic_h symbolic,
    ssi_sparse_matrix_h matrix,
    ssi_numeric_factorization_h* out_factorization);

  void (*numeric_factorization_release)(
    ssi_numeric_factorization_h factorization);
  ssi_status_t (*numeric_factorization_dtype)(
    ssi_numeric_factorization_h factorization,
    ssi_dtype_t* out_dtype);
  ssi_status_t (*numeric_factorization_solve)(
    ssi_numeric_factorization_h factorization,
    ssi_matrix_h rhs,
    ssi_matrix_h solution);
} ssi_plugin_api_t;

typedef ssi_status_t (*ssi_get_plugin_fn)(ssi_plugin_api_t* out_api);

SSI_EXPORT ssi_status_t ssi_get_plugin(ssi_plugin_api_t* out_api);

#ifdef __cplusplus
}
#endif

#endif
