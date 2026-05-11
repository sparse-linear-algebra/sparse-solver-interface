#include "acutest.h"
#include "sparse_solver_interface.hpp"

static void test_blank_slate(void)
{
  TEST_CHECK(1);
}

TEST_LIST = {
  { "blank_slate", test_blank_slate },
  { 0, 0 }
};
