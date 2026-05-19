# sparse-solver-interface

Blank slate for a sparse solver interface design.

The public C++ header is `include/sparse_solver_interface.hpp`.

The stable binary plugin ABI is split into:

- `include/sparse_solver_interface_c.h`: plain C ABI, opaque handles, versioned function table.
- `include/sparse_solver_interface_plugin.hpp`: header-only C++ adapters for exporting an implementation and loading a plugin back as normal `ssi::context_t` objects.

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
  std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t dtype) override;
  std::shared_ptr<ssi::graph_t> make_graph(ssi::itype_t itype) override;
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
auto graph = context->make_graph(ssi::itype_t::i64);
auto rhs = context->make_matrix(ssi::dtype_t::fp64);
```

The exported shared object must provide the `ssi_get_plugin` C symbol. The
`SSI_EXPORT_PLUGIN(...)` macro generates that symbol and fills the versioned C
function table. The loader checks the ABI major version before constructing the
C++ wrapper objects.

The lifetime rules are the same as the C++ interface:

- Borrowed dense matrix views must outlive the matrix that borrowed them.
- Borrowed compressed graph views must outlive the graph that borrowed them.
- Borrowed sparse value views must outlive the sparse matrix that borrowed them.
- Builder callback buffers are only valid for the duration of the callback.

Implementations should throw normal C++ exceptions internally. The plugin
adapter catches them at the C boundary, returns a status code, and exposes the
message through the ABI's last-error hook. Consumers using the C++ loader see
those statuses converted back to C++ exceptions.
