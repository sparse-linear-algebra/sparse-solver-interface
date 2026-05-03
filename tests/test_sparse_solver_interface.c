#include "acutest.h"
#include "sparse_solver_interface.h"

static void test_build_link_and_run(void)
{
    TEST_CHECK(ssi_version() != 0);
}

TEST_LIST = {
    { "build_link_and_run", test_build_link_and_run },
    { 0, 0 }
};
