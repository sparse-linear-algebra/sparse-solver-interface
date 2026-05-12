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
#include <type_traits>

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

enum class sparse_orientation_t : uint8_t{
  row,
  column
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
    matrix_t(context_t* context) : context_(context) {    
      if(context == nullptr) throw std::runtime_error("null context");
    }
    virtual ~matrix_t(){}
    virtual int64_t nrows() = 0;
    virtual int64_t ncols() = 0;
    virtual dtype_t dtype() = 0;
    const context_t& context(){ 
      return *context_; 
    }

    /*Make enough space but don't initialize the data in any way.*/
    virtual matrix_t& preallocate(int64_t nrows,int64_t ncols) = 0;
    /*Attempts to zero-copy in a matrix view. The lifetime of the data represented by matrix_view_t must outlive this matrix_t.*/
    virtual matrix_t& borrow_matrix_view(const placement_t& placement,const matrix_view_t& view) = 0;
    /*Build from ordinarily addressible memory and have the framework copy it to where it needs to go.*/
    /*The callback should be a function for initializing arbitrary slices into matrix_t.*/
    virtual matrix_t& build_from_host(std::function<void(matrix_view_t&)>& builder) = 0;
    /*Build explicitly from a placement.*/
    virtual matrix_t& build_from_placement(std::function<void(const placement_t&,matrix_view_t&)>& builder) = 0;

    /*Read matrix into a host-addressible view.*/
    /*Lifetime of data pointed by matrix_view_t only guaranteed to survive the callback.*/
    virtual matrix_t& read_to_host(std::function<void(const matrix_view_t&)>& reader) = 0;

    /*Read matrix into placement-addressible view.*/    
    /*Lifetime of data pointed by matrix_view_t only guaranteed to survive the callback.*/
    virtual matrix_t& read_to_placement(const placement_t& placement,std::function<void(const matrix_view_t&)>& reader) = 0;
  private:
    context_t* context_;
};

struct sparse_count_builder_t{
  sparse_orientation_t orientation;
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
      throw std::invalid_argument("negative sparse nonzero count");
    }
    const int64_t local = row_or_col - beg;
    if(itype == itype_t::i32){
      if(count > std::numeric_limits<int32_t>::max()){
        throw std::out_of_range("sparse nonzero count does not fit int32_t");
      }
      counts.i32[local] = static_cast<int32_t>(count);
      return;
    }
    if(itype == itype_t::i64){
      counts.i64[local] = count;
      return;
    }
    throw std::invalid_argument("unknown sparse index type");
  }

  private:
    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("sparse builder row/column is outside the active range");
      }
      if(orientation == sparse_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("sparse builder row is outside the matrix");
        }
        return;
      }
      if(orientation == sparse_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("sparse builder column is outside the matrix");
        }
        return;
      }
      throw std::invalid_argument("unknown sparse orientation");
    }
};

struct sparse_entry_builder_t{
  sparse_orientation_t orientation;
  itype_t itype;
  dtype_t dtype;
  int64_t nrows;
  int64_t ncols;
  /* Rows for row-oriented builds, columns for column-oriented builds. */
  int64_t beg;
  int64_t end;
  /* Offsets are computed by the implementation from the first pass counts.
   * Users fill ids and values, either directly or through set_entry().
   * offsets has extent() + 1 entries. ids and values have offsets[extent()]
   * addressable entries for this slice.
   */
  union{
    const int32_t* i32;
    const int64_t* i64;
  } offsets;
  union{
    int32_t* i32;
    int64_t* i64;
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
      throw std::out_of_range("sparse builder offset index is outside the active range");
    }
    if(itype == itype_t::i32){
      return offsets.i32[local];
    }
    if(itype == itype_t::i64){
      return offsets.i64[local];
    }
    throw std::invalid_argument("unknown sparse index type");
  }

  int64_t count(int64_t row_or_col) const{
    validate_member(row_or_col);
    const int64_t local = row_or_col - beg;
    return offset(local + 1) - offset(local);
  }

  template<typename T>
  void set_entry(int64_t row_or_col,int64_t entry_index,int64_t id,const T& value){
    validate_value_type<T>();
    validate_id(id);
    const int64_t local = row_or_col - beg;
    validate_member(row_or_col);
    const int64_t row_or_col_count = offset(local + 1) - offset(local);
    if(entry_index < 0 || entry_index >= row_or_col_count){
      throw std::out_of_range("sparse entry index is outside the row/column count");
    }
    const int64_t position = offset(local) + entry_index;
    set_id(position,id);
    value_mut<T>(position) = value;
  }

  private:
    template<typename>
    static constexpr bool dependent_false = false;

    void validate_member(int64_t row_or_col) const{
      if(row_or_col < beg || row_or_col >= end){
        throw std::out_of_range("sparse builder row/column is outside the active range");
      }
      if(orientation == sparse_orientation_t::row){
        if(row_or_col < 0 || row_or_col >= nrows){
          throw std::out_of_range("sparse builder row is outside the matrix");
        }
        return;
      }
      if(orientation == sparse_orientation_t::column){
        if(row_or_col < 0 || row_or_col >= ncols){
          throw std::out_of_range("sparse builder column is outside the matrix");
        }
        return;
      }
      throw std::invalid_argument("unknown sparse orientation");
    }

    void validate_id(int64_t id) const{
      if(id < 0){
        throw std::out_of_range("negative sparse entry id");
      }
      if(orientation == sparse_orientation_t::row){
        if(id >= ncols){
          throw std::out_of_range("sparse column id is outside the matrix");
        }
        return;
      }
      if(orientation == sparse_orientation_t::column){
        if(id >= nrows){
          throw std::out_of_range("sparse row id is outside the matrix");
        }
        return;
      }
      throw std::invalid_argument("unknown sparse orientation");
    }

    void set_id(int64_t position,int64_t id){
      if(itype == itype_t::i32){
        if(id > std::numeric_limits<int32_t>::max()){
          throw std::out_of_range("sparse entry id does not fit int32_t");
        }
        ids.i32[position] = static_cast<int32_t>(id);
        return;
      }
      if(itype == itype_t::i64){
        ids.i64[position] = id;
        return;
      }
      throw std::invalid_argument("unknown sparse index type");
    }

    template<typename T>
    void validate_value_type() const{
      if constexpr(std::is_same_v<T,float32_t>){
        if(dtype != dtype_t::fp32) throw std::invalid_argument("sparse value type does not match dtype");
      }else if constexpr(std::is_same_v<T,float64_t>){
        if(dtype != dtype_t::fp64) throw std::invalid_argument("sparse value type does not match dtype");
      }else if constexpr(std::is_same_v<T,complex64_t>){
        if(dtype != dtype_t::c64) throw std::invalid_argument("sparse value type does not match dtype");
      }else if constexpr(std::is_same_v<T,complex128_t>){
        if(dtype != dtype_t::c128) throw std::invalid_argument("sparse value type does not match dtype");
      }else{
        static_assert(dependent_false<T>,"unsupported sparse value type");
      }
    }

    template<typename T>
    T& value_mut(int64_t position){
      if constexpr(std::is_same_v<T,float32_t>){
        return values.fp32[position];
      }else if constexpr(std::is_same_v<T,float64_t>){
        return values.fp64[position];
      }else if constexpr(std::is_same_v<T,complex64_t>){
        return values.c64[position];
      }else if constexpr(std::is_same_v<T,complex128_t>){
        return values.c128[position];
      }else{
        static_assert(dependent_false<T>,"unsupported sparse value type");
      }
    }
};

class sparse_matrix_t{
  public:
    sparse_matrix_t(context_t* context) : context_(context){
      if(context == nullptr) throw std::runtime_error("null context");
    }
    virtual ~sparse_matrix_t() {}
    virtual itype_t itype() = 0;
    virtual dtype_t dtype() = 0;
    virtual int64_t nrows() = 0;
    virtual int64_t ncols() = 0;
    virtual int64_t nnz() = 0;
    /* Build in two passes. In the first pass the user fills nonzero counts for
     * rows or columns in [beg,end). The implementation computes offsets and
     * presents raw ids/values buffers in the second pass. Entry ids are global;
     * duplicates within a row/column are invalid and must be rejected by the
     * implementation before the matrix is finalized. Sorted ids are not
     * required.
     */
    virtual sparse_matrix_t& build_from_host(
      int64_t nrows,
      int64_t ncols,
      sparse_orientation_t orientation,
      std::function<void(sparse_count_builder_t&)>& count_builder,
      std::function<void(sparse_entry_builder_t&)>& entry_builder) = 0;
    const context_t& context(){
      return *context_;
    }
  private:
    context_t* context_;
};

class context_t{
  public:

};


}  // namespace ssi

#endif
