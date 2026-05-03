#!/usr/bin/env sh

set -eu

build_dir="${BUILD_DIR:-/tmp/sparse_solver_interface_build}"

cmake -S . -B "$build_dir" -DBUILD_TESTING=ON
cmake --build "$build_dir"
ctest --test-dir "$build_dir" --output-on-failure
