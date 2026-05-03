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
 * Terminology:
 *   "Host"    : Calling process data with directly addressable memory.
 *   "Logical" : Data owned by an SSI context. Logical data is not directly
 *               addressable by the calling process.
 *
 * SSI splits the sparse solver interface between host-side data views and
 * logical objects owned by an SSI context. The logical address space may be
 * more complex than the host process address space, for example multiple GPUs
 * or multiple processes by way of MPI.
 *
 * TODO: Extend host-side data descriptions for multiprocess contexts. The
 * current host data model assumes a large directly addressable host address
 * space. MPI-style pre-distributed host data, root-owned input scatter, and
 * rank-local empty logical object registration need explicit representation.
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

/* Scalar types. */
typedef enum{
  ssi_f32 = 0,
  ssi_f64 = 1,
  ssi_c64 = 2,
  ssi_c128 = 3
} ssi_dtype_t;

/* Index types.*/
typedef enum {
  ssi_i32 = 0,
  ssi_i64 = 1
} ssi_itype_t;


typedef enum {
  ssi_row_major = 0,
  ssi_col_major = 1
} ssi_matrix_order_t;

typedef struct{
  ssi_matrix_order_t order;
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_int64_t ld;
  ssi_dtype_t dtype;
  union {
    ssi_float32_t* f32;
    ssi_float64_t* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
  } data;
} ssi_host_dense_matrix_t;


typedef enum {
  ssi_coo = 0,
  ssi_csr = 1,
  ssi_csc = 2
} ssi_sparse_format_t;

typedef enum {
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
typedef enum {
  ssi_numeric_property_none = 0,
  ssi_numeric_property_symmetric = 1u << 0,
  ssi_numeric_property_hermitian = 1u << 1,
  ssi_numeric_property_positive_definite = 1u << 2
} ssi_numeric_property_t;


typedef struct{
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_int64_t nnz;
  ssi_itype_t itype;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } rids;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } cids;
} ssi_coo_t;

typedef struct{
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_int64_t nnz;
  ssi_itype_t itype;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } offs;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } cids;
} ssi_csr_t;

typedef struct{
  ssi_int64_t nrows;
  ssi_int64_t ncols;
  ssi_int64_t nnz;
  ssi_itype_t itype;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } offs;
  union {
    ssi_int32_t* i32;
    ssi_int64_t* i64;
  } rids;
} ssi_csc_t;

typedef struct{
  ssi_sparse_format_t format;
  ssi_structural_property_t properties;
  union {
    ssi_coo_t coo;
    ssi_csr_t csr;
    ssi_csc_t csc;
  } graph;
} ssi_host_sparse_graph_t;

typedef struct{
  ssi_int64_t nnz;
  ssi_dtype_t dtype;
  ssi_numeric_property_t properties;
  union {
    ssi_float32_t* f32;
    ssi_float64_t* f64;
    ssi_complex64_t* c64;
    ssi_complex128_t* c128;
  } data;
} ssi_host_sparse_numeric_t;

typedef struct ssi_context ssi_context_t;
typedef struct ssi_logical_graph ssi_logical_graph_t;
typedef struct ssi_logical_numeric ssi_logical_numeric_t;
typedef struct ssi_logical_dense_matrix ssi_logical_dense_matrix_t;
typedef struct ssi_logical_symbolic_factorization ssi_logical_symbolic_factorization_t;
typedef struct ssi_logical_numeric_factorization ssi_logical_numeric_factorization_t;


/*
 * Flags whether host-side data is read, write, or readwrite from the point of
 * view of an SSI context.
 */
typedef enum{
  ssi_host_access_read_only = 0,
  ssi_host_access_write_only = 1,
  ssi_host_access_read_write = 2
} ssi_host_access_t;

typedef enum{
  ssi_host_data_dense_matrix = 0,
  ssi_host_data_sparse_graph = 1,
  ssi_host_data_sparse_numeric = 2
} ssi_host_data_kind_t;

/*
 * A wrapper around host-addressable data.
 */
typedef struct{
  ssi_host_data_kind_t kind;
  ssi_host_access_t access;
  union {
    ssi_host_dense_matrix_t* dense_matrix;
    ssi_host_sparse_graph_t* sparse_graph;
    ssi_host_sparse_numeric_t* sparse_numeric;
  } data;
} ssi_host_data_t;

typedef struct{
  ssi_context_t* (*create_context)(void*);

  /*
   * Register host-addressable data as logical data in an SSI context. An
   * implementation may copy the host data, retain a view, or create a backend
   * mirror, subject to the requested access mode.
   */
  ssi_logical_graph_t* (*register_logical_graph_from_host)(ssi_context_t* context,const ssi_host_sparse_graph_t* host_graph,ssi_host_access_t access);
  ssi_logical_numeric_t* (*register_logical_numeric_from_host)(ssi_context_t* context,ssi_logical_graph_t* graph,const ssi_host_sparse_numeric_t* host_numeric,ssi_host_access_t access);
  ssi_logical_dense_matrix_t* (*register_logical_dense_matrix_from_host)(ssi_context_t* context,const ssi_host_dense_matrix_t* host_matrix,ssi_host_access_t access);

  /*
   * Create logical data without binding it to host-addressable storage.
   */
  ssi_logical_graph_t* (*create_logical_graph)(ssi_context_t* context,ssi_sparse_format_t format,ssi_int64_t nrows,ssi_int64_t ncols,ssi_int64_t nnz,ssi_itype_t itype,ssi_structural_property_t properties);
  ssi_logical_numeric_t* (*create_logical_numeric)(ssi_context_t* context,ssi_logical_graph_t* graph,ssi_dtype_t dtype,ssi_numeric_property_t properties);
  ssi_logical_dense_matrix_t* (*create_logical_dense_matrix)(ssi_context_t* context,ssi_int64_t nrows,ssi_int64_t ncols,ssi_dtype_t dtype);

  /*
   * Copy between host-addressable data and logical data.
   */
  ssi_status_t (*copy_host_to_logical_graph)(ssi_context_t* context,ssi_logical_graph_t* logical_graph,const ssi_host_sparse_graph_t* host_graph);
  ssi_status_t (*copy_logical_graph_to_host)(ssi_context_t* context,const ssi_logical_graph_t* logical_graph,ssi_host_sparse_graph_t* host_graph);
  ssi_status_t (*copy_host_to_logical_numeric)(ssi_context_t* context,ssi_logical_numeric_t* logical_numeric,const ssi_host_sparse_numeric_t* host_numeric);
  ssi_status_t (*copy_logical_numeric_to_host)(ssi_context_t* context,const ssi_logical_numeric_t* logical_numeric,ssi_host_sparse_numeric_t* host_numeric);
  ssi_status_t (*copy_host_to_logical_dense_matrix)(ssi_context_t* context,ssi_logical_dense_matrix_t* logical_matrix,const ssi_host_dense_matrix_t* host_matrix);
  ssi_status_t (*copy_logical_dense_matrix_to_host)(ssi_context_t* context,const ssi_logical_dense_matrix_t* logical_matrix,ssi_host_dense_matrix_t* host_matrix);

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
  ssi_status_t (*refactorize_numeric_factorization)(ssi_context_t* context,ssi_logical_numeric_factorization_t* factorization);

  /*
   * Solve using logical dense matrices for right-hand sides and solutions.
   */
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
