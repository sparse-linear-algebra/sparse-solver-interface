#ifndef SPARSE_SOLVER_INTERFACE_HPP
#define SPARSE_SOLVER_INTERFACE_HPP

#include <cstdint>
#include <memory>
#include <cassert>
#include <complex>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>

namespace ssi {

enum class dtype_t : uint8_t{
  fp32,
  fp64,
  c64,
  c128
};
using float32_t = float;
using float64_t = double;
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;

template<typename T>
struct dtype_of;

template<>
struct dtype_of<float32_t>{
  static constexpr dtype_t value = dtype_t::fp32;
};

template<>
struct dtype_of<float64_t>{
  static constexpr dtype_t value = dtype_t::fp64;
};

template<>
struct dtype_of<complex64_t>{
  static constexpr dtype_t value = dtype_t::c64;
};

template<>
struct dtype_of<complex128_t>{
  static constexpr dtype_t value = dtype_t::c128;
};




enum class itype_t : uint8_t{
  i32,
  i64
};
using int32_t = std::int32_t;
using int64_t = std::int64_t;

enum class matrix_order_t : uint8_t{
  col_major,
  row_major
};

enum class graph_orientation_t : uint8_t{
  row,
  column
};

enum class property_state_t : uint8_t{
  unknown,
  known_false,
  known_true
};
using graph_property_state_t = property_state_t;

enum class graph_property_t : uint8_t{
  structurally_symmetric,
  strong_hall
};

enum class numeric_property_t : uint8_t{
  symmetric,
  positive_definite,
  negative_definite
};

struct graph_properties_t{
  graph_property_state_t structurally_symmetric = graph_property_state_t::unknown;
  graph_property_state_t strong_hall = graph_property_state_t::unknown;

  graph_property_state_t get(graph_property_t property) const{
    if(property == graph_property_t::structurally_symmetric){
      return structurally_symmetric;
    }
    if(property == graph_property_t::strong_hall){
      return strong_hall;
    }
    throw std::invalid_argument("unknown graph property");
  }

  void set(graph_property_t property,graph_property_state_t state){
    if(property == graph_property_t::structurally_symmetric){
      structurally_symmetric = state;
      return;
    }
    if(property == graph_property_t::strong_hall){
      strong_hall = state;
      return;
    }
    throw std::invalid_argument("unknown graph property");
  }
};

struct numeric_properties_t{
  property_state_t symmetric = property_state_t::unknown;
  property_state_t positive_definite = property_state_t::unknown;
  property_state_t negative_definite = property_state_t::unknown;

  property_state_t get(numeric_property_t property) const{
    if(property == numeric_property_t::symmetric){
      return symmetric;
    }
    if(property == numeric_property_t::positive_definite){
      return positive_definite;
    }
    if(property == numeric_property_t::negative_definite){
      return negative_definite;
    }
    throw std::invalid_argument("unknown numeric property");
  }

  void set(numeric_property_t property,property_state_t state){
    if(property == numeric_property_t::symmetric){
      symmetric = state;
      return;
    }
    if(property == numeric_property_t::positive_definite){
      positive_definite = state;
      return;
    }
    if(property == numeric_property_t::negative_definite){
      negative_definite = state;
      return;
    }
    throw std::invalid_argument("unknown numeric property");
  }
};

/* Defines memory placement .*/
class placement_t{
  public:
    virtual ~placement_t(){}
};

/* This represents a local slice into a global Matrix. It may not 
 * live in the same address space.*/
struct matrix_view_t{
  matrix_order_t order;
  dtype_t dtype;
  int64_t rbeg;
  int64_t rend;
  int64_t cbeg;
  int64_t cend;
  int64_t ld;
  union{
    float32_t* fp32;
    float64_t* fp64;
    complex64_t* c64;
    complex128_t* c128;
  } d;
  /* indexing occurs in the global index space.*/
  int64_t index(int64_t r,int64_t c) const{
    assert(r>=rbeg);
    assert(c>=cbeg);
    assert(r<rend);
    assert(c<cend);
    r-=rbeg;
    c-=cbeg;
    if(order == matrix_order_t::col_major){
      return r + ld * c;
    }
    assert(order == matrix_order_t::row_major);
    return c + ld*r;
  }
  template<typename T>
  T& value_mut(int64_t r,int64_t c){
    assert(dtype == dtype_of<T>::value);
    if constexpr(dtype_of<T>::value == dtype_t::fp32){
      return d.fp32[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::fp64){
      return d.fp64[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c64){
      return d.c64[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c128){
      return d.c128[index(r,c)];
    }
  }
  template<typename T>
  const T& value(int64_t r,int64_t c) const{
    assert(dtype == dtype_of<T>::value);
    if constexpr(dtype_of<T>::value == dtype_t::fp32){
      return d.fp32[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::fp64){
      return d.fp64[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c64){
      return d.c64[index(r,c)];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c128){
      return d.c128[index(r,c)];
    }
  }
};


class context_t;
class matrix_t{
  public:
    matrix_t(std::shared_ptr<context_t> context) : context_(context)  {
      if(context == nullptr) throw std::runtime_error("null context");
    }
    virtual ~matrix_t(){}
    virtual int64_t nrows() const = 0;
    virtual int64_t ncols() const = 0;
    virtual dtype_t dtype() const = 0;
    const context_t& context() const{ 
      return *context_; 
    }

    /*Make enough space but don't initialize the data in any way.*/
    virtual void preallocate(int64_t nrows,int64_t ncols) = 0;
    /*Attempts to zero-copy in a matrix view. The lifetime of the data represented by matrix_view_t must outlive this matrix_t.*/
    virtual void borrow_matrix_view(const placement_t& placement,const matrix_view_t& view) = 0;
    /*Build from ordinarily addressible memory and have the framework copy it to where it needs to go.*/
    /*The callback should be a function for initializing arbitrary slices into matrix_t.*/
    virtual void build_from_host(std::function<void(matrix_view_t&)>& builder) = 0;
    /*Build explicitly from a placement.*/
    virtual void build_from_placement(std::function<void(const placement_t&,matrix_view_t&)>& builder) = 0;

    /*Read matrix into a host-addressible view.*/
    /*Lifetime of data pointed by matrix_view_t only guaranteed to survive the callback.*/
    virtual void read_to_host(std::function<void(const matrix_view_t&)>& reader) = 0;

    /*Read matrix into placement-addressible view.*/    
    /*Lifetime of data pointed by matrix_view_t only guaranteed to survive the callback.*/
    virtual void read_to_placement(const placement_t& placement,std::function<void(const matrix_view_t&)>& reader) = 0;
  private:
    std::shared_ptr<context_t> context_;
};

struct graph_count_builder_t{
  graph_orientation_t orientation;
  itype_t itype;
  int64_t nrows;
  int64_t ncols;
  /* Rows for row-oriented builds, columns for column-oriented builds. */
  int64_t beg;
  int64_t end;
  union{
    int32_t* i32;
    int64_t* i64;
  } counts;

  int64_t extent() const{
    return end - beg;
  }

  void set_count(int64_t row_or_col,int64_t count){
    validate_member(row_or_col);
    if(count < 0){
      throw std::invalid_argument("negative graph edge count");
    }
    const int64_t local = row_or_col - beg;
    if(itype == itype_t::i32){
      if(count > std::numeric_limits<int32_t>::max()){
        throw std::out_of_range("graph edge count does not fit int32_t");
      }
      counts.i32[local] = static_cast<int32_t>(count);
      return;
    }
    if(itype == itype_t::i64){
      counts.i64[local] = count;
      return;
    }
    throw std::invalid_argument("unknown graph index type");
  }

  private:
    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("graph builder row/column is outside the active range");
      }
      if(orientation == graph_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("graph builder row is outside the graph");
        }
        return;
      }
      if(orientation == graph_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("graph builder column is outside the graph");
        }
        return;
      }
      throw std::invalid_argument("unknown graph orientation");
    }
};

struct graph_edge_builder_t{
  graph_orientation_t orientation;
  itype_t itype;
  int64_t nrows;
  int64_t ncols;
  /* Rows for row-oriented builds, columns for column-oriented builds. */
  int64_t beg;
  int64_t end;
  /* Offsets are computed by the implementation from the first pass counts.
   * Users fill ids, either directly or through set_edge(). offsets has
   * extent() + 1 entries. ids has offsets[extent()] addressable entries for
   * this slice.
   */
  union{
    const int32_t* i32;
    const int64_t* i64;
  } offsets;
  union{
    int32_t* i32;
    int64_t* i64;
  } ids;

  int64_t extent() const{
    return end - beg;
  }

  int64_t offset(int64_t local) const{
    if(local < 0 || local > extent()){
      throw std::out_of_range("graph builder offset index is outside the active range");
    }
    if(itype == itype_t::i32){
      return offsets.i32[local];
    }
    if(itype == itype_t::i64){
      return offsets.i64[local];
    }
    throw std::invalid_argument("unknown graph index type");
  }

  int64_t degree(int64_t row_or_col) const{
    validate_member(row_or_col);
    const int64_t local = row_or_col - beg;
    return offset(local + 1) - offset(local);
  }

  int64_t count(int64_t row_or_col) const{
    return degree(row_or_col);
  }

  void set_edge(int64_t row_or_col,int64_t edge_index,int64_t id){
    validate_id(id);
    const int64_t local = row_or_col - beg;
    validate_member(row_or_col);
    const int64_t row_or_col_degree = offset(local + 1) - offset(local);
    if(edge_index < 0 || edge_index >= row_or_col_degree){
      throw std::out_of_range("graph edge index is outside the row/column degree");
    }
    const int64_t position = offset(local) + edge_index;
    set_id(position,id);
  }

  private:
    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("graph builder row/column is outside the active range");
      }
      if(orientation == graph_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("graph builder row is outside the graph");
        }
        return;
      }
      if(orientation == graph_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("graph builder column is outside the graph");
        }
        return;
      }
      throw std::invalid_argument("unknown graph orientation");
    }

    void validate_id(int64_t id) const{
      if(id < 0){
        throw std::out_of_range("negative graph edge id");
      }
      if(orientation == graph_orientation_t::row){
        if(id >= ncols){
          throw std::out_of_range("graph column id is outside the graph");
        }
        return;
      }
      if(orientation == graph_orientation_t::column){
        if(id >= nrows){
          throw std::out_of_range("graph row id is outside the graph");
        }
        return;
      }
      throw std::invalid_argument("unknown graph orientation");
    }

    void set_id(int64_t position,int64_t id){
      if(itype == itype_t::i32){
        if(id > std::numeric_limits<int32_t>::max()){
          throw std::out_of_range("graph edge id does not fit int32_t");
        }
        ids.i32[position] = static_cast<int32_t>(id);
        return;
      }
      if(itype == itype_t::i64){
        ids.i64[position] = id;
        return;
      }
      throw std::invalid_argument("unknown graph index type");
    }
};

struct compressed_graph_view_t{
  graph_orientation_t orientation;
  itype_t itype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  union{
    const int32_t* i32;
    const int64_t* i64;
  } offsets;
  union{
    const int32_t* i32;
    const int64_t* i64;
  } ids;

  int64_t extent() const{
    return end - beg;
  }

  int64_t offset(int64_t local) const{
    if(local < 0 || local > extent()){
      throw std::out_of_range("compressed graph offset index is outside the active range");
    }
    if(itype == itype_t::i32){
      return offsets.i32[local];
    }
    if(itype == itype_t::i64){
      return offsets.i64[local];
    }
    throw std::invalid_argument("unknown graph index type");
  }

  int64_t degree(int64_t row_or_col) const{
    validate_member(row_or_col);
    const int64_t local = row_or_col - beg;
    return offset(local + 1) - offset(local);
  }

  int64_t edge_id(int64_t row_or_col,int64_t edge_index) const{
    validate_member(row_or_col);
    const int64_t local = row_or_col - beg;
    const int64_t row_or_col_degree = offset(local + 1) - offset(local);
    if(edge_index < 0 || edge_index >= row_or_col_degree){
      throw std::out_of_range("compressed graph edge index is outside the row/column degree");
    }
    const int64_t position = offset(local) + edge_index;
    if(itype == itype_t::i32){
      return ids.i32[position];
    }
    if(itype == itype_t::i64){
      return ids.i64[position];
    }
    throw std::invalid_argument("unknown graph index type");
  }

  private:
    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("compressed graph row/column is outside the active range");
      }
      if(orientation == graph_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("compressed graph row is outside the graph");
        }
        return;
      }
      if(orientation == graph_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("compressed graph column is outside the graph");
        }
        return;
      }
      throw std::invalid_argument("unknown graph orientation");
    }
};

class sparse_matrix_t;
class symbolic_t;
class numeric_factorization_t;
class graph_t{
  public:
    graph_t(std::shared_ptr<context_t> context) : context_(context){
      if(context == nullptr) throw std::runtime_error("null context");
    }
    virtual ~graph_t() {}
    virtual itype_t itype() const = 0;
    virtual int64_t nrows() const = 0;
    virtual int64_t ncols() const = 0;
    virtual int64_t nedges() const = 0;
    virtual graph_properties_t properties() const = 0;
    virtual void assert_property(
      graph_property_t property,
      graph_property_state_t state) = 0;
    virtual void assert_properties(const graph_properties_t& properties) = 0;
    virtual void compute_property(graph_property_t property) = 0;
    virtual void compute_properties() = 0;
    /* Build in two passes. In the first pass the user fills edge counts for
     * rows or columns in [beg,end). The implementation computes offsets and
     * presents raw id buffers in the second pass. Edge ids are global;
     * duplicates within a row/column are invalid and must be rejected by the
     * implementation before the graph is finalized. Sorted ids are not
     * required.
     */
    virtual void build_from_host(
      int64_t nrows,
      int64_t ncols,
      graph_orientation_t orientation,
      std::function<void(graph_count_builder_t&)>& count_builder,
      std::function<void(graph_edge_builder_t&)>& edge_builder) = 0;
    /* Attempts to zero-copy in a host-addressible compressed graph. The
     * offsets and ids represented by compressed_graph_view_t must outlive this
     * graph_t.
     */
    virtual void borrow_compressed_graph_view(
      const compressed_graph_view_t& view) = 0;
    const context_t& context() const{
      return *context_;
    }

    virtual std::shared_ptr<sparse_matrix_t> make_sparse_matrix() = 0;
    virtual std::shared_ptr<symbolic_t> make_symbolic_analysis() = 0;
  private:
    std::shared_ptr<context_t> context_;
};

struct sparse_values_view_t{
  dtype_t dtype;
  int64_t nedges;
  union{
    const float32_t* fp32;
    const float64_t* fp64;
    const complex64_t* c64;
    const complex128_t* c128;
  } values;

  template<typename T>
  const T& value(int64_t edge_position) const{
    assert(dtype == dtype_of<T>::value);
    if(edge_position < 0 || edge_position >= nedges){
      throw std::out_of_range("sparse value index is outside the graph edge range");
    }
    if constexpr(dtype_of<T>::value == dtype_t::fp32){
      return values.fp32[edge_position];
    }
    if constexpr(dtype_of<T>::value == dtype_t::fp64){
      return values.fp64[edge_position];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c64){
      return values.c64[edge_position];
    }
    if constexpr(dtype_of<T>::value == dtype_t::c128){
      return values.c128[edge_position];
    }
  }
};

struct sparse_value_builder_t{
  graph_orientation_t orientation;
  itype_t itype;
  dtype_t dtype;
  int64_t nrows;
  int64_t ncols;
  int64_t beg;
  int64_t end;
  union{
    const int32_t* i32;
    const int64_t* i64;
  } offsets;
  union{
    const int32_t* i32;
    const int64_t* i64;
  } ids;
  union{
    float32_t* fp32;
    float64_t* fp64;
    complex64_t* c64;
    complex128_t* c128;
  } values;

  int64_t extent() const{
    return end - beg;
  }

  int64_t offset(int64_t local) const{
    if(local < 0 || local > extent()){
      throw std::out_of_range("sparse matrix offset index is outside the active range");
    }
    if(itype == itype_t::i32){
      return offsets.i32[local];
    }
    if(itype == itype_t::i64){
      return offsets.i64[local];
    }
    throw std::invalid_argument("unknown graph index type");
  }

  int64_t degree(int64_t row_or_col) const{
    validate_member(row_or_col);
    const int64_t local = row_or_col - beg;
    return offset(local + 1) - offset(local);
  }

  int64_t edge_id(int64_t row_or_col,int64_t edge_index) const{
    return id_at(position(row_or_col,edge_index));
  }

  template<typename T>
  T& value_mut(int64_t row_or_col,int64_t edge_index){
    assert(dtype == dtype_of<T>::value);
    return value_at<T>(position(row_or_col,edge_index));
  }

  template<typename T>
  const T& value(int64_t row_or_col,int64_t edge_index) const{
    assert(dtype == dtype_of<T>::value);
    return value_at<T>(position(row_or_col,edge_index));
  }

  template<typename T>
  void set_value(int64_t row_or_col,int64_t edge_index,const T& value){
    value_mut<T>(row_or_col,edge_index) = value;
  }

  private:
    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("sparse matrix row/column is outside the active range");
      }
      if(orientation == graph_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("sparse matrix row is outside the graph");
        }
        return;
      }
      if(orientation == graph_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("sparse matrix column is outside the graph");
        }
        return;
      }
      throw std::invalid_argument("unknown graph orientation");
    }

    int64_t position(int64_t row_or_col,int64_t edge_index) const{
      validate_member(row_or_col);
      const int64_t local = row_or_col - beg;
      const int64_t row_or_col_degree = offset(local + 1) - offset(local);
      if(edge_index < 0 || edge_index >= row_or_col_degree){
        throw std::out_of_range("sparse matrix edge index is outside the row/column degree");
      }
      return offset(local) + edge_index;
    }

    int64_t id_at(int64_t position) const{
      if(itype == itype_t::i32){
        return ids.i32[position];
      }
      if(itype == itype_t::i64){
        return ids.i64[position];
      }
      throw std::invalid_argument("unknown graph index type");
    }

    template<typename T>
    T& value_at(int64_t position){
      if constexpr(dtype_of<T>::value == dtype_t::fp32){
        return values.fp32[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::fp64){
        return values.fp64[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::c64){
        return values.c64[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::c128){
        return values.c128[position];
      }
    }

    template<typename T>
    const T& value_at(int64_t position) const{
      if constexpr(dtype_of<T>::value == dtype_t::fp32){
        return values.fp32[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::fp64){
        return values.fp64[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::c64){
        return values.c64[position];
      }
      if constexpr(dtype_of<T>::value == dtype_t::c128){
        return values.c128[position];
      }
    }
};

class sparse_matrix_t{
  public:
    sparse_matrix_t(std::shared_ptr<graph_t> graph) : graph_(graph) {
      if(graph == nullptr) throw std::runtime_error("null graph");
    }
    virtual ~sparse_matrix_t() {}
    virtual int64_t nrows() const = 0;
    virtual int64_t ncols() const = 0;
    virtual dtype_t dtype() const = 0;
    virtual numeric_properties_t properties() const = 0;
    virtual void assert_property(
      numeric_property_t property,
      property_state_t state) = 0;
    virtual void assert_properties(const numeric_properties_t& properties) = 0;
    virtual void compute_property(numeric_property_t property) = 0;
    virtual void compute_properties() = 0;
    const graph_t& graph() const{
      return *graph_;
    }
    const context_t& context() const{
      return graph_->context();
    }

    virtual void build_from_host(
      dtype_t dtype,
      graph_orientation_t orientation,
      std::function<void(sparse_value_builder_t&)>& builder) = 0;
    template<typename T>
    void build_from_host(
      graph_orientation_t orientation,
      std::function<void(sparse_value_builder_t&)>& builder){
      build_from_host(dtype_of<T>::value,orientation,builder);
    }
    virtual void read_to_host(
      graph_orientation_t orientation,
      std::function<void(const sparse_value_builder_t&)>& reader) const = 0;
    /* Attempts to zero-copy in host-addressible sparse values. Values are in
     * the same edge order used when the graph's ids were supplied.
     */
    virtual void borrow_sparse_values_view(
      const sparse_values_view_t& view) = 0;

  private:
    std::shared_ptr<graph_t> graph_;
};

class symbolic_t{
  public:
    symbolic_t(std::shared_ptr<graph_t> graph) : graph_(graph) {
      if(graph == nullptr) throw std::runtime_error("null graph");
    }
    virtual ~symbolic_t() {}

    const graph_t& graph() const{
      return *graph_;
    }
    const context_t& context() const{
      return graph_->context();
    }

    virtual std::shared_ptr<numeric_factorization_t>
    make_numeric_factorization(std::shared_ptr<sparse_matrix_t> matrix) = 0;

  private:
    std::shared_ptr<graph_t> graph_;
};

class numeric_factorization_t{
  public:
    numeric_factorization_t(
      std::shared_ptr<symbolic_t> symbolic,
      std::shared_ptr<sparse_matrix_t> matrix) :
      symbolic_(symbolic),
      matrix_(matrix) {
      if(symbolic == nullptr) throw std::runtime_error("null symbolic analysis");
      if(matrix == nullptr) throw std::runtime_error("null sparse matrix");
      if(&symbolic->graph() != &matrix->graph()){
        throw std::invalid_argument("numeric factorization graph mismatch");
      }
    }
    virtual ~numeric_factorization_t() {}

    const symbolic_t& symbolic() const{
      return *symbolic_;
    }
    const sparse_matrix_t& matrix() const{
      return *matrix_;
    }
    const context_t& context() const{
      return symbolic_->context();
    }
    virtual dtype_t dtype() const = 0;
    virtual void solve(const matrix_t& rhs,matrix_t& solution) const = 0;

  private:
    std::shared_ptr<symbolic_t> symbolic_;
    std::shared_ptr<sparse_matrix_t> matrix_;
};

class context_t{
  public:
    virtual ~context_t() {}

    virtual std::shared_ptr<matrix_t> make_matrix(dtype_t dtype) = 0;
    template<typename T>
    std::shared_ptr<matrix_t> make_matrix(){
      return make_matrix(dtype_of<T>::value);
    }
    virtual std::shared_ptr<graph_t> make_graph(itype_t itype) = 0;
};


}  // namespace ssi

#endif
