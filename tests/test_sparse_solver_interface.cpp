#include "acutest.h"
#include "sparse_solver_interface.hpp"

#include <stdexcept>
#include <vector>

static void test_sparse_count_builder_i32(void)
{
  std::vector<ssi::int32_t> counts(3,0);
  ssi::sparse_count_builder_t builder{
    ssi::sparse_orientation_t::row,
    ssi::itype_t::i32,
    5,
    7,
    1,
    4,
    {.i32 = counts.data()}
  };

  builder.set_count(1,2);
  builder.set_count(2,0);
  builder.set_count(3,4);

  TEST_CHECK(counts[0] == 2);
  TEST_CHECK(counts[1] == 0);
  TEST_CHECK(counts[2] == 4);
  TEST_EXCEPTION(builder.set_count(4,1),std::out_of_range);
  TEST_EXCEPTION(builder.set_count(1,-1),std::invalid_argument);
}

static void test_sparse_entry_builder_row_oriented(void)
{
  std::vector<ssi::int64_t> offsets{0,2,2,5};
  std::vector<ssi::int64_t> ids(5,-1);
  std::vector<ssi::float64_t> values(5,0.0);
  ssi::sparse_entry_builder_t builder{
    ssi::sparse_orientation_t::row,
    ssi::itype_t::i64,
    ssi::dtype_t::fp64,
    5,
    7,
    1,
    4,
    {.i64 = offsets.data()},
    {.i64 = ids.data()},
    {.fp64 = values.data()}
  };

  TEST_CHECK(builder.count(1) == 2);
  TEST_CHECK(builder.count(2) == 0);
  TEST_CHECK(builder.count(3) == 3);

  builder.set_entry<ssi::float64_t>(1,0,6,1.25);
  builder.set_entry<ssi::float64_t>(1,1,0,2.5);
  builder.set_entry<ssi::float64_t>(3,2,4,3.75);

  TEST_CHECK(ids[0] == 6);
  TEST_CHECK(ids[1] == 0);
  TEST_CHECK(ids[4] == 4);
  TEST_CHECK(values[0] == 1.25);
  TEST_CHECK(values[1] == 2.5);
  TEST_CHECK(values[4] == 3.75);

  TEST_EXCEPTION(builder.set_entry<ssi::float64_t>(2,0,1,1.0),std::out_of_range);
  TEST_EXCEPTION(builder.set_entry<ssi::float64_t>(1,0,7,1.0),std::out_of_range);
  TEST_EXCEPTION(builder.set_entry<ssi::float32_t>(1,0,1,1.0f),std::invalid_argument);
}

static void test_sparse_entry_builder_column_oriented(void)
{
  std::vector<ssi::int32_t> offsets{0,1,3};
  std::vector<ssi::int32_t> ids(3,-1);
  std::vector<ssi::complex64_t> values(3,{0.0f,0.0f});
  ssi::sparse_entry_builder_t builder{
    ssi::sparse_orientation_t::column,
    ssi::itype_t::i32,
    ssi::dtype_t::c64,
    4,
    6,
    2,
    4,
    {.i32 = offsets.data()},
    {.i32 = ids.data()},
    {.c64 = values.data()}
  };

  builder.set_entry<ssi::complex64_t>(2,0,3,{1.0f,2.0f});
  builder.set_entry<ssi::complex64_t>(3,1,0,{3.0f,4.0f});

  TEST_CHECK(ids[0] == 3);
  TEST_CHECK(ids[2] == 0);
  TEST_CHECK(values[0] == ssi::complex64_t(1.0f,2.0f));
  TEST_CHECK(values[2] == ssi::complex64_t(3.0f,4.0f));
  TEST_EXCEPTION(builder.set_entry<ssi::complex64_t>(2,0,4,{0.0f,0.0f}),std::out_of_range);
}

TEST_LIST = {
  { "sparse_count_builder_i32", test_sparse_count_builder_i32 },
  { "sparse_entry_builder_row_oriented", test_sparse_entry_builder_row_oriented },
  { "sparse_entry_builder_column_oriented", test_sparse_entry_builder_column_oriented },
  { 0, 0 }
};
