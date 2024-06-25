#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>


UnitTest_fn_def(test_cpu_display){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    char res_buf[32];
    CPUState_display(RV, cpu, res_buf);
    UnitTest_msg("%s", res_buf);
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_cpu_display);
    return NULL;
}

UnitTest_run(all_tests);