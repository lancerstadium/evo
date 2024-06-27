#include <sob/sob.h>
#include <evo/evo.h>
#include <evo/task.h>

UnitTest_fn_def(test_dump_elf){
    char* path = "wuhu.elf";
    Val* elf = Val_str(path);
    Task(Dump)* d = Task_init(Dump, "dp-elf", elf);
    Task_run(Dump, d);
    UnitTest_ast(IS_FILE(path), "Not found elf file");
    return NULL;
}

UnitTest_fn_def(test_exec_rv) {
    Task(Exec)* d = Task_init(Exec, "exec-rv", NULL);
    Task_run(Exec, d);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_dump_elf);
    UnitTest_add(test_exec_rv);
    return NULL;
}

UnitTest_run(all_tests);