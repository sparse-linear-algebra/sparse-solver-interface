#include "acutest.h"
#include "sparse_solver_interface.hpp"

#include <stdexcept>
#include <vector>

static void test_graph_properties(void)
{
  ssi::graph_properties_t properties;

  TEST_CHECK(
    properties.get(ssi::graph_property_t::structurally_symmetric) ==
    ssi::graph_property_state_t::unknown);
  TEST_CHECK(
    properties.get(ssi::graph_property_t::strong_hall) ==
    ssi::graph_property_state_t::unknown);

  properties.set(
    ssi::graph_property_t::structurally_symmetric,
    ssi::graph_property_state_t::known_true);
  properties.set(
    ssi::graph_property_t::strong_hall,
    ssi::graph_property_state_t::known_false);

  TEST_CHECK(
    properties.structurally_symmetric ==
    ssi::graph_property_state_t::known_true);
  TEST_CHECK(
    properties.strong_hall ==
    ssi::graph_property_state_t::known_false);
}

static void test_graph_count_builder_i32(void)
{
  std::vector<ssi::int32_t> counts(3,0);
  ssi::graph_count_builder_t builder{
    ssi::graph_orientation_t::row,
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

static void test_graph_edge_builder_row_oriented(void)
{
  std::vector<ssi::int64_t> offsets{0,2,2,5};
  std::vector<ssi::int64_t> ids(5,-1);
  ssi::graph_edge_builder_t builder{
    ssi::graph_orientation_t::row,
    ssi::itype_t::i64,
    5,
    7,
    1,
    4,
    {.i64 = offsets.data()},
    {.i64 = ids.data()}
  };

  TEST_CHECK(builder.degree(1) == 2);
  TEST_CHECK(builder.degree(2) == 0);
  TEST_CHECK(builder.degree(3) == 3);
  TEST_CHECK(builder.count(3) == 3);

  builder.set_edge(1,0,6);
  builder.set_edge(1,1,0);
  builder.set_edge(3,2,4);

  TEST_CHECK(ids[0] == 6);
  TEST_CHECK(ids[1] == 0);
  TEST_CHECK(ids[4] == 4);

  TEST_EXCEPTION(builder.set_edge(2,0,1),std::out_of_range);
  TEST_EXCEPTION(builder.set_edge(1,0,7),std::out_of_range);
}

static void test_graph_edge_builder_column_oriented(void)
{
  std::vector<ssi::int32_t> offsets{0,1,3};
  std::vector<ssi::int32_t> ids(3,-1);
  ssi::graph_edge_builder_t builder{
    ssi::graph_orientation_t::column,
    ssi::itype_t::i32,
    4,
    6,
    2,
    4,
    {.i32 = offsets.data()},
    {.i32 = ids.data()}
  };

  builder.set_edge(2,0,3);
  builder.set_edge(3,1,0);

  TEST_CHECK(ids[0] == 3);
  TEST_CHECK(ids[2] == 0);
  TEST_EXCEPTION(builder.set_edge(2,0,4),std::out_of_range);
}

TEST_LIST = {
  { "graph_properties", test_graph_properties },
  { "graph_count_builder_i32", test_graph_count_builder_i32 },
  { "graph_edge_builder_row_oriented", test_graph_edge_builder_row_oriented },
  { "graph_edge_builder_column_oriented", test_graph_edge_builder_column_oriented },
  { 0, 0 }
};
