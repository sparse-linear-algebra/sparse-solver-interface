# sparse-solver-interface

This project defines a binary-stable C interface for sparse linear solver
implementations that can be loaded with `dlopen`/`dlsym` and swapped at runtime.

The C ABI in `include/sparse_solver_interface.h` is the only binary contract.
It uses opaque handles, fixed-width scalar types, versioned API tables, and a
single loader entry point:

```c
ssi_status_t ssi_get_api_v1(ssi_api_v1_t* out_api, size_t out_size);
```

The C++ headers under `include/ssi/` are header-only helpers. They provide a
clearer implementer-facing interface, matrix views, client-side wrappers, and an
adapter for exporting a C++ solver implementation through the C ABI without
exposing any C++ ABI across the shared-library boundary.

See `docs/implementors.md` for direct C and C++ implementation guidance.
