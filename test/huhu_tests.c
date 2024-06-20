#include <sob/sob.h>


UnitTest_fn_def(test_hello){
    UnitTest_ast(32 == 2 + 30, "hello");
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_hello);
    return NULL;
}

UnitTest_run(all_tests);