#include <sob/sob.h>
#include <evo/evo.h>
#include <evo/task.h>

UnitTest_fn_def(test_dump_elf){
    char* path = "wuhu.elf";
    Val* elf = Val_str(path);
    Task(Dump)* t = Task_init(Dump, "dp-elf", elf);
    Task_run(Dump, t);
    UnitTest_ast(IS_FILE(path), "Not found elf file");
    return NULL;
}

UnitTest_fn_def(test_exec_rv) {
    Task(Exec)* t = Task_init(Exec, "exec-rv", NULL);
    Task_run(Exec, t);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_dump_elf);
    UnitTest_add(test_exec_rv);
    return NULL;
}

UnitTest_run(all_tests);