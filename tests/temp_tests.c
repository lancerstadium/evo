
#include "sob.h"


UnitTest_fn_def(test_hello) {
    UnitTest_msg("hello world!");
    return NULL;
}



UnitTest_run(test_hello);