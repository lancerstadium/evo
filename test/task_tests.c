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
    // Val* img = &(Val)Val_new(
    //     0x97, 0x02, 0x00, 0x00,
    //     0x23, 0x88, 0x02, 0x00,
    //     0x03, 0xc5, 0x02, 0x01,
    //     0x73, 0x00, 0x10, 0x00,
    // );
    Val* img = Val_from_u32((u32[]){
        0x00000297,
        0x00028823,
        0x0102c503,
        0x00100073,
    }, 16);

    Task(Exec)* t = Task_init(Exec, "exec-rv", img);
    Task_run(Exec, t);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_dump_elf);
    UnitTest_add(test_exec_rv);
    return NULL;
}

UnitTest_run(all_tests);