#include <sob/sob.h>
#include <evo/evo.h>
#include <isa/rv/def.h>


UnitTest_fn_def(test_cpu_display){
    CPUState(RV) * cpu = CPUState_init(RV, 56);
    char res_buf[32];
    for(size_t i = 0; i < RegMax(RV); i++) {
        CPUState_displayone(RV, cpu, res_buf, i);
        UnitTest_msg("%s", res_buf);
    }
    return NULL;
}


UnitTest_fn_def(all_tests) {
    UnitTest_add(test_cpu_display);
    return NULL;
}

UnitTest_run(all_tests);