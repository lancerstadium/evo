#include <sob/sob.h>
#include <evo/task.h>

UnitTest_fn_def(test_dump_elf){
    Task(Dump)* d = Task_create(Dump, "dp-01");
    Task_run(Dump, d);
    UnitTest_msg("%s", STR_BOOL(IS_FILE("a.out")));
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_dump_elf);
    return NULL;
}

UnitTest_run(all_tests);