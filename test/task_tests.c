#include <sob/sob.h>
#include <evo/evo.h>

UnitTest_fn_def(test_dump_elf){
    Task(Dump)* d = Task_create(Dump, "dp-01");
    Task_run(Dump, d);
    UnitTest_ast(IS_FILE("out.elf"), "Not found out.elf");
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_dump_elf);
    return NULL;
}

UnitTest_run(all_tests);