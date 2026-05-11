#ifndef SPARSE_SOLVER_INTERFACE_HPP
#define SPARSE_SOLVER_INTERFACE_HPP

#include <cstdint>
#include <memory>
#include <cassert>
#include <complex>

namespace ssi {

enum class DType : uint8_t{
  fp32,
  fp64,
  c64,
  c128
};
using Float32 = float;
using Float64 = double;
using Complex64 = std::complex<float>;
using Complex128 = std::complex<double>;


enum class IType : uint8_t{
  i32,
  i64
};

using Int32 = int32_t;
using Int64 = int64_t;

enum class MatrixOrder : uint8_t{
  col_major,
  row_major
};

/* Defines memory placement .*/
class Placement{
  public:
    virtual ~Placement(){}
};

/* This represents a local slice into a global Matrix. It may not 
 * live in the same address space.*/
struct MatrixView{
  MatrixOrder order;
  DType dtype;
  Int64 rbeg;
  Int64 rend;
  Int64 cbeg;
  Int64 cend;
  Int64 ld;
  union{
    Float32* fp32;
    Float64* fp64;
    Complex64* c64;
    Complex128* c128;
  };
  /* indexing occurs in the global index space.*/
  Int64 index(Int64 r,Int64 c) const{
    assert(r>=rbeg);
    assert(c>=cbeg);
    assert(r<rend);
    assert(c<cend);
    r-=rbeg;
    c-=cbeg;
    Int64 nrows = rend-rbeg;
    Int64 ncols = cend-cbeg;
    if(order == MatrixOrder::col_major){
      return r + ld * c;
    }
    return c + ld*r;
  }
  template<typename T>
  T& value(Int64 r,Int64 c){
  }

};

template<typename T,MatrixOrder Order>
class MatrixView{
  public:
    MatrixView(T* view,Int64 nrows,Int64 ncols,Int64 ld) : view_(view), nrows_(nrows), ncols_(ncols), ld_(ld) {}
    T& operator()(const Int64 r,const Int64 c){
      return view_[index_(r,c)];
    }
    const T& operator()(const Int64 r,const Int64 c) const{
      return view_[index_(r,c)];
    }
    Int64 nrows() const {return nrows_;}
    Int64 ncols() const {return ncols_;}
    Int64 ld() const {return ld_;}
    T* data() const {return view_;}
  private:
    Int64 index_(const Int64 r,const Int64 c){
      if constexpr(Order == MatrixOrder::col_major){
        return r + ld_ * c;
      }
      else{
        return c + ld_ * r;
      }
    }
    //Ownership is optional
    std::shared_ptr<T[]> owned_;
    T* view_;
    Int64 nrows_;
    Int64 ncols_;
    Int64 ld_;
};

class Context;
class Matrix{
  public:
    virtual Int64 nrows() = 0;
    virtual Int64 ncols() = 0;
    virtual DType dtype() = 0;
    virtual Context* context() = 0;
    virtual ~Matrix(){}
};

class Context{
  public:

};


}  // namespace ssi

#endif
