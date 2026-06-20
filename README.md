# sparse-solver-interface

Blank slate for a sparse solver interface design.

The public C++ header is `include/sparse_solver_interface.hpp`.

The stable binary plugin ABI is split into:

- `include/sparse_solver_interface_c.h`: plain C ABI, opaque handles, versioned function table.
- `include/sparse_solver_interface_plugin.hpp`: header-only C++ adapters for exporting an implementation and loading a plugin back as normal `ssi::context_t` objects.

## Manual Install

Install the headers into a non-system prefix:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local/sparse_solver_interface/1.0.0"
cmake --build build
cmake --install build
```

Then point another CMake project at that prefix:

```cmake
set(SPARSE_SOLVER_INTERFACE_ROOT "$ENV{HOME}/.local/sparse_solver_interface/1.0.0")

target_include_directories(
  my_solver
  PRIVATE
    "${SPARSE_SOLVER_INTERFACE_ROOT}/include"
)
```

An implementation can export a context factory with:

```cpp
#include "sparse_solver_interface_plugin.hpp"

std::shared_ptr<ssi::context_t> make_context();

SSI_EXPORT_PLUGIN(make_context)
```

A consumer can load the resulting shared object with:

```cpp
auto context = ssi::load_context_from_shared_object("libsolver_plugin.so");
```

## Writing a Plugin

Plugin authors implement the normal C++ interfaces from
`include/sparse_solver_interface.hpp`. The plugin boundary is only a packaging
layer: no STL, exceptions, virtual C++ ABI, or `std::shared_ptr` cross the shared
object boundary.

At minimum, an implementation provides a concrete `ssi::context_t` and exports a
factory for it:

```cpp
#include "sparse_solver_interface_plugin.hpp"

#include <memory>

class my_context_t final : public ssi::context_t {
public:
  ssi::support_result_t check_support(
    const ssi::sparse_problem_properties_t& properties) const override;
  std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t dtype) override;
  std::shared_ptr<ssi::graph_t> make_graph(ssi::itype_t itype) override;
  std::shared_ptr<ssi::sparse_problem_t> make_sparse_problem(
    const ssi::sparse_problem_properties_t& properties) override;
};

std::shared_ptr<ssi::context_t> make_my_context()
{
  return std::make_shared<my_context_t>();
}

SSI_EXPORT_PLUGIN(make_my_context)
```

Build that translation unit as a shared library and make sure this repository's
`include/` directory is on the include path. For example:

```sh
c++ -std=c++20 -fPIC -shared \
  -I/path/to/sparse-solver-interface/include \
  my_solver_plugin.cpp \
  -o libmy_solver_plugin.so
```

Consumers do not link against the plugin at build time. They only need these
headers, then load the binary artifact at runtime:

```cpp
#include "sparse_solver_interface_plugin.hpp"

auto context = ssi::load_context_from_shared_object("./libmy_solver_plugin.so");
ssi::sparse_problem_properties_t properties;
properties.nrows = 100;
properties.ncols = 100;
properties.itype = ssi::itype_t::i64;
properties.dtype = ssi::dtype_t::fp64;
properties.structurally_symmetric = ssi::property_state_t::known_true;
properties.numerically_symmetric = ssi::property_state_t::known_true;
properties.positive_definite = ssi::property_state_t::known_true;

auto support = context->check_support(properties);
if(!support.supported()){
  throw ssi::unsupported_error_t(support.reason);
}

auto problem = context->make_sparse_problem(properties);
auto graph = problem->make_graph();
auto matrix = problem->make_sparse_matrix();
auto rhs = context->make_matrix(ssi::dtype_t::fp64);
```

Use `sparse_problem_properties_t` as the solver-facing properties object. It
keeps shape, index type, value type, structural facts, and numeric facts in one
descriptor so each solver can choose its own Cholesky, LU, QR, least-squares, or
fallback path from matrix facts instead of from caller-supplied algorithm hints.
The lower-level graph and sparse-matrix objects no longer carry separate
property sets.

Property assertions are monotonic. Unknown facts may become known, and repeated
known facts are accepted, but known-true and known-false contradictions are
rejected. Dimensions, orientation, index type, value type, and symmetric storage
are locked once a dependent graph, sparse matrix, or symbolic analysis has been
created. Implementations should also reject impossible fact combinations such as
full column rank for a short-wide matrix, nonsingularity for a nonsquare matrix,
or definiteness without a square shape.

Use `context_t::check_support(properties)` before constructing solver-specific
objects when a client wants to choose among solvers without probing by failure.
Return `ssi::status_t::unsupported` with a short reason for valid problems that
are outside the solver's supported class. Throw `ssi::unsupported_error_t` for
the same condition when construction or execution reaches an unsupported path.

Numeric failures are separate from unsupported problems. Use typed numeric
statuses or exceptions such as `singular`, `rank_deficient`, `indefinite`,
`zero_pivot`, `breakdown`, and `not_converged` when the solver accepted the
problem class but this particular data failed. `numeric_factorization_t::solve`
returns `ssi::solve_result_t`, so direct solvers can return a minimal success
result while iterative or refining solvers can report residuals, iterations,
refinement steps, convergence, backward error, and an optional reason string.

The exported shared object must provide the `ssi_get_plugin` C symbol. The
`SSI_EXPORT_PLUGIN(...)` macro generates that symbol and fills the versioned C
function table. The loader checks the ABI major version before constructing the
C++ wrapper objects.

Borrowed-memory rules are intentionally explicit:

- SSI does not own borrowed memory and will not free it.
- Borrowed dense matrix views must outlive the matrix that borrowed them.
- Borrowed compressed graph views must outlive the graph that borrowed them.
- Borrowed sparse value views must outlive the sparse matrix and any
  factorization that may reference those values.
- Borrowed graph structure is immutable after it is borrowed.
- External mutation of borrowed values while an SSI operation is active is
  forbidden unless the implementation explicitly documents stronger guarantees.
- Input/output aliasing, including `solve(rhs,rhs)`, is unsupported unless the
  implementation explicitly documents it.
- SSI objects are not internally synchronized; callers own thread-safety for
  concurrent access and for memory shared across plugins.
- Across plugin boundaries, borrowed memory must remain addressable and
  ABI-compatible for the receiving plugin.
- Builder callback buffers are only valid for the duration of the callback.

Implementations should throw normal C++ exceptions internally. The plugin
adapter catches them at the C boundary, returns a status code, and exposes the
message through the ABI's last-error hook. Consumers using the C++ loader see
typed statuses converted back to typed C++ exceptions. Operation result structs
also carry status and reason fields for non-throwing metadata paths.
